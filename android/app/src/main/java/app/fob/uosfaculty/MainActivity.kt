package app.fob.uosfaculty

import android.Manifest
import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.net.Uri
import android.net.http.SslError
import android.os.Bundle
import android.speech.SpeechRecognizer
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.webkit.ConsoleMessage
import android.webkit.CookieManager
import android.webkit.SslErrorHandler
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import android.widget.ProgressBar
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature

/**
 * The whole app: one full-screen WebView on the hosted chatbot (BuildConfig.SERVER_URL, from android/gradle.properties).
 *
 * The website is not changed for the app. What the shell adds:
 *   - the address bar is gone and only the chatbot's own address loads inside; every other link (official pages,
 *     mailto:, tel:) opens in the app that normally handles it
 *   - voice input: Android's WebView has no Web Speech API, so [VoiceBridge] supplies one from the device recogniser
 *   - a bundled "cannot be reached" page with Try again, the system back button, a loading bar, launcher shortcuts
 *   - the guest session cookie is kept, so saved conversations and settings survive closing the app
 */
class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var progress: ProgressBar
    private val server: Uri = Uri.parse(BuildConfig.SERVER_URL)
    private var voice: VoiceBridge? = null
    private var lastAppUrl: String? = null
    private var clearHistoryWhenLoaded = false

    private val micPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        voice?.onPermissionResult(granted)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)

        webView = WebView(this)
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            visibility = View.GONE
        }
        val root = FrameLayout(this).apply {
            setBackgroundColor(ContextCompat.getColor(context, R.color.page_background))
            addView(webView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
            addView(progress, FrameLayout.LayoutParams(MATCH_PARENT, (3 * resources.displayMetrics.density).toInt(), Gravity.TOP))
        }
        setContentView(root)

        // Keep the page clear of the status bar, the navigation bar, the camera cutout and the on-screen keyboard.
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout() or WindowInsetsCompat.Type.ime()
            )
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            WindowInsetsCompat.CONSUMED
        }
        val night = resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
        WindowCompat.getInsetsController(window, root).apply {
            isAppearanceLightStatusBars = !night
            isAppearanceLightNavigationBars = !night
        }

        configureWebView()
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) webView.goBack() else finish()
            }
        })

        if (server.scheme.isNullOrEmpty() || server.host.isNullOrEmpty() || BuildConfig.SERVER_URL.contains("CHANGE-ME")) {
            Toast.makeText(this, R.string.server_not_set, Toast.LENGTH_LONG).show()
            webView.loadUrl(OFFLINE_PAGE)
        } else if (savedInstanceState == null || webView.restoreState(savedInstanceState) == null) {
            webView.loadUrl(startUrlFor(intent))
        } else {
            // Saved state can outlive an app update that changed the server address: never reopen the old address.
            val restored = webView.copyBackForwardList().currentItem?.url
            if (restored == null || !isAppUrl(Uri.parse(restored))) {
                clearHistoryWhenLoaded = true
                webView.loadUrl(startUrlFor(intent))
            }
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun configureWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = false
            allowContentAccess = false
            setSupportMultipleWindows(false)          // target="_blank" links arrive in shouldOverrideUrlLoading
            mediaPlaybackRequiresUserGesture = true
        }
        CookieManager.getInstance().apply {
            setAcceptCookie(true)
            setAcceptThirdPartyCookies(webView, false)
        }
        webView.overScrollMode = View.OVER_SCROLL_NEVER
        webView.setBackgroundColor(ContextCompat.getColor(this, R.color.page_background))

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val url = request.url
                if (url.scheme == "fobapp") {             // the bundled offline page's "Try again"
                    retry()
                    return true
                }
                if (isAppUrl(url)) return false
                if (request.isForMainFrame) openOutside(url)
                return true
            }

            override fun onPageStarted(view: WebView, url: String, favicon: Bitmap?) {
                progress.progress = 0
                progress.visibility = View.VISIBLE
                voice?.cancelFromHost()                   // a recognition started on the previous page ends with it
                if (isAppUrl(Uri.parse(url))) lastAppUrl = url
            }

            override fun onPageFinished(view: WebView, url: String) {
                progress.visibility = View.GONE
                if (clearHistoryWhenLoaded && isAppUrl(Uri.parse(url))) {
                    clearHistoryWhenLoaded = false
                    view.clearHistory()                   // back must not return to the offline page
                }
            }

            override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
                if (request.isForMainFrame) showOffline(request.url)
            }

            override fun onReceivedSslError(view: WebView, handler: SslErrorHandler, error: SslError) {
                handler.cancel()                          // never continue past a certificate problem
                if (error.url == view.url || view.url == null) showOffline(Uri.parse(error.url))
            }
        }
        webView.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView, newProgress: Int) {
                progress.progress = newProgress
            }

            override fun onConsoleMessage(message: ConsoleMessage): Boolean {
                if (BuildConfig.DEBUG) Log.d("FobWeb", "${message.messageLevel()} ${message.message()} (${message.sourceId()}:${message.lineNumber()})")
                return true
            }
        }
        // Documents the site offers as downloads open in the browser, which has a download manager and a PDF viewer.
        webView.setDownloadListener { url, _, _, _, _ -> openOutside(Uri.parse(url)) }

        installVoice()
    }

    /** Voice input needs a current WebView (message listener + document-start script) and a speech service on the
     *  device. Without them the page reports "voice input is not available", exactly as it does in Firefox. */
    private fun installVoice() {
        if (!SpeechRecognizer.isRecognitionAvailable(this) ||
            server.scheme.isNullOrEmpty() || server.host.isNullOrEmpty()
        ) return
        val origin = server.scheme + "://" + server.host + (if (server.port != -1) ":" + server.port else "")
        val bridge = VoiceBridge(this) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                voice?.onPermissionResult(true)
            } else {
                micPermission.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER) ||
            !WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)
        ) return
        try {
            // Both are limited to the chatbot's own origin: no other page can reach the microphone bridge.
            WebViewCompat.addWebMessageListener(webView, VoiceBridge.JS_OBJECT, setOf(origin), bridge)
            WebViewCompat.addDocumentStartJavaScript(webView, VoiceBridge.POLYFILL_JS, setOf(origin))
            voice = bridge
        } catch (e: IllegalArgumentException) {
            voice = null                                  // malformed origin rule: run without voice input
        }
    }

    private fun isAppUrl(url: Uri): Boolean =
        url.scheme == server.scheme && url.host.equals(server.host, ignoreCase = true) && effectivePort(url) == effectivePort(server)

    private fun effectivePort(url: Uri): Int = if (url.port != -1) url.port else if (url.scheme == "https") 443 else 80

    /** The server page for a launch intent: the start page, or the path of a launcher shortcut (fobapp://open/<path>). */
    private fun startUrlFor(intent: Intent?): String {
        val data = intent?.data
        if (data == null || data.scheme != "fobapp" || data.host != "open") return BuildConfig.SERVER_URL + "/"
        val path = data.encodedPath?.takeIf { it.startsWith("/") } ?: "/"
        val query = data.encodedQuery?.let { "?$it" } ?: ""
        return BuildConfig.SERVER_URL + path + query
    }

    private fun showOffline(failed: Uri) {
        if (isAppUrl(failed)) lastAppUrl = failed.toString()
        voice?.cancelFromHost()
        webView.loadUrl(OFFLINE_PAGE)
    }

    private fun retry() {
        clearHistoryWhenLoaded = true
        webView.loadUrl(lastAppUrl ?: (BuildConfig.SERVER_URL + "/"))
    }

    private fun openOutside(url: Uri) {
        val scheme = url.scheme ?: return
        if (scheme !in OUTSIDE_SCHEMES) return            // never hand intent:, file:, javascript: and the like to the system
        try {
            startActivity(Intent(Intent.ACTION_VIEW, url).addCategory(Intent.CATEGORY_BROWSABLE))
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(this, R.string.no_app_for_link, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (intent.data?.scheme == "fobapp") webView.loadUrl(startUrlFor(intent))
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        webView.saveState(outState)
    }

    override fun onPause() {
        voice?.cancelFromHost()
        CookieManager.getInstance().flush()               // the guest session cookie reaches disk before the app can be killed
        webView.onPause()
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        webView.onResume()
    }

    override fun onDestroy() {
        voice?.destroy()
        (webView.parent as? FrameLayout)?.removeView(webView)
        webView.destroy()
        super.onDestroy()
    }

    private companion object {
        const val OFFLINE_PAGE = "file:///android_asset/offline.html"
        val OUTSIDE_SCHEMES = setOf("https", "http", "mailto", "tel")
    }
}
