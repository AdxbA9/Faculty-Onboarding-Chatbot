plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val serverUrl: String = (project.findProperty("FOB_SERVER_URL") as String? ?: "").trim().trimEnd('/')

android {
    namespace = "app.fob.uosfaculty"
    compileSdk = 35

    defaultConfig {
        applicationId = "app.fob.uosfaculty"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
        buildConfigField("String", "SERVER_URL", "\"$serverUrl\"")
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        debug {
            versionNameSuffix = "-debug"
        }
        release {
            // The app is a thin shell (two Kotlin files); shrinking is not worth the reflection risk.
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

// A release that opens a placeholder or a plain-http address is never what was meant: stop the build.
tasks.matching { it.name == "assembleRelease" || it.name == "bundleRelease" }.configureEach {
    doFirst {
        require(serverUrl.startsWith("https://") && !serverUrl.contains("CHANGE-ME")) {
            "Set FOB_SERVER_URL in android/gradle.properties to the chatbot's HTTPS address before a release build (now: '$serverUrl')."
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.webkit:webkit:1.12.1")
}
