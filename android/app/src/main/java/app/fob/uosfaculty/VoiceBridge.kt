package app.fob.uosfaculty

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.webkit.WebView
import androidx.webkit.JavaScriptReplyProxy
import androidx.webkit.WebMessageCompat
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import org.json.JSONException
import org.json.JSONObject

/**
 * Voice input for the app. The website's microphone button drives the browser's Web Speech API (ui/voice.py, VOICE_JS),
 * which Android's WebView does not have. [POLYFILL_JS] defines `webkitSpeechRecognition` before the page's own scripts
 * run and forwards start / stop / abort here; this class runs the device's speech recogniser and sends back the same
 * events a browser would (start, audiostart, result, error, end). The page and its state machine are unchanged, and as in
 * the browser the transcript only lands in the message box: nothing is sent until the user presses Send.
 *
 * All methods run on the main thread (WebView message callbacks and SpeechRecognizer callbacks both do).
 */
class VoiceBridge(
    private val activity: Activity,
    private val requestMicrophone: () -> Unit,
) : WebViewCompat.WebMessageListener, RecognitionListener {

    private var recognizer: SpeechRecognizer? = null
    private var reply: JavaScriptReplyProxy? = null
    private var active = false
    private var pendingLang: String? = null

    override fun onPostMessage(view: WebView, message: WebMessageCompat, sourceOrigin: Uri, isMainFrame: Boolean, replyProxy: JavaScriptReplyProxy) {
        if (!isMainFrame) return
        val msg = try {
            JSONObject(message.data ?: return)
        } catch (e: JSONException) {
            return
        }
        when (msg.optString("cmd")) {
            "start" -> {
                if (active || pendingLang != null) return
                reply = replyProxy
                val lang = msg.optString("lang")
                pendingLang = if (LANGUAGE_TAG.matches(lang)) lang else "en-US"
                requestMicrophone()                      // answers through onPermissionResult, at once when already granted
            }
            "stop" -> if (active) recognizer?.stopListening()
            "cancel" -> if (pendingLang != null) {       // aborted while the permission dialog is still open
                pendingLang = null
                send("error", "aborted")
                send("end")
            } else {
                cancelFromHost()
            }
        }
    }

    fun onPermissionResult(granted: Boolean) {
        val lang = pendingLang ?: return
        pendingLang = null
        if (!granted) {
            send("error", "not-allowed")
            send("end")
            return
        }
        val rec = recognizer ?: SpeechRecognizer.createSpeechRecognizer(activity).also {
            it.setRecognitionListener(this)
            recognizer = it
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE, lang)
            .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            .putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            .putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, activity.packageName)
        active = true
        try {
            rec.startListening(intent)
        } catch (e: SecurityException) {
            fail("not-allowed")
        }
    }

    /** Stop a running recognition without a result: the page asked (abort), or the app left the foreground or lost the
     *  page. A pending permission request is left alone: the system dialog itself pauses the activity. */
    fun cancelFromHost() {
        if (!active) return
        recognizer?.cancel()
        fail("aborted")
    }

    fun destroy() {
        active = false
        pendingLang = null
        reply = null
        recognizer?.destroy()
        recognizer = null
    }

    private fun fail(code: String) {
        active = false
        send("error", code)
        send("end")
    }

    private fun send(kind: String, text: String = "") {
        val json = JSONObject().put("kind", kind).put("text", text).toString()
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) return
        try {
            reply?.postMessage(json)
        } catch (e: RuntimeException) {
            // the page was closed or replaced: nobody is listening any more
        }
    }

    private fun firstResult(results: Bundle?): String =
        results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.trim().orEmpty()

    // ---- RecognitionListener ----

    override fun onReadyForSpeech(params: Bundle?) {
        if (!active) return
        send("start")
        send("audiostart")
    }

    override fun onPartialResults(partialResults: Bundle?) {
        if (!active) return
        val text = firstResult(partialResults)
        if (text.isNotEmpty()) send("interim", text)
    }

    override fun onResults(results: Bundle?) {
        if (!active) return
        active = false
        val text = firstResult(results)
        if (text.isNotEmpty()) send("final", text)
        send("end")
    }

    override fun onError(error: Int) {
        if (!active) return
        fail(
            when (error) {                                // the Web Speech API error names the page already translates
                SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "no-speech"
                SpeechRecognizer.ERROR_AUDIO -> "audio-capture"
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "not-allowed"
                SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT, SpeechRecognizer.ERROR_SERVER,
                SpeechRecognizer.ERROR_SERVER_DISCONNECTED -> "network"
                SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED, SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE -> "language-not-supported"
                SpeechRecognizer.ERROR_CLIENT, SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "aborted"
                else -> "service-not-allowed"
            }
        )
    }

    override fun onBeginningOfSpeech() {}
    override fun onRmsChanged(rmsdB: Float) {}
    override fun onBufferReceived(buffer: ByteArray?) {}
    override fun onEndOfSpeech() {}
    override fun onEvent(eventType: Int, params: Bundle?) {}

    companion object {
        const val JS_OBJECT = "FobNativeVoice"

        private val LANGUAGE_TAG = Regex("^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8}){0,3}$")

        /** A minimal SpeechRecognition (the members ui/voice.py VOICE_JS uses), backed by [JS_OBJECT]. */
        val POLYFILL_JS = """
(function () {
  var bridge = window.$JS_OBJECT;
  if (!bridge || window.top !== window) return;
  // No "already defined" check: Android's WebView exposes a webkitSpeechRecognition that only ever answers
  // 'not-allowed', so the page's feature test passes and the microphone can never start. Replace it.
  var current = null;
  function fire(rec, name, ev) {
    var handler = rec['on' + name];
    if (typeof handler === 'function') { try { handler.call(rec, ev || {}); } catch (e) {} }
  }
  function Recognition() {
    this.lang = 'en-US'; this.continuous = false; this.interimResults = false; this.maxAlternatives = 1;
    this.onstart = null; this.onaudiostart = null; this.onresult = null; this.onerror = null; this.onend = null;
  }
  Recognition.prototype.start = function () {
    if (current) throw new DOMException('recognition has already started', 'InvalidStateError');
    current = this;
    bridge.postMessage(JSON.stringify({cmd: 'start', lang: String(this.lang || 'en-US')}));
  };
  Recognition.prototype.stop = function () { if (current === this) bridge.postMessage(JSON.stringify({cmd: 'stop'})); };
  Recognition.prototype.abort = function () { if (current === this) bridge.postMessage(JSON.stringify({cmd: 'cancel'})); };
  bridge.onmessage = function (event) {
    var rec = current, msg;
    if (!rec) return;
    try { msg = JSON.parse(event.data); } catch (e) { return; }
    if (msg.kind === 'start') fire(rec, 'start');
    else if (msg.kind === 'audiostart') fire(rec, 'audiostart');
    else if (msg.kind === 'interim' || msg.kind === 'final') {
      var result = [{transcript: msg.text, confidence: msg.kind === 'final' ? 1 : 0}];
      result.isFinal = msg.kind === 'final';
      fire(rec, 'result', {resultIndex: 0, results: [result]});
    }
    else if (msg.kind === 'error') fire(rec, 'error', {error: msg.text});
    else if (msg.kind === 'end') { current = null; fire(rec, 'end'); }
  };
  window.SpeechRecognition = Recognition;
  window.webkitSpeechRecognition = Recognition;
})();
""".trimIndent()
    }
}
