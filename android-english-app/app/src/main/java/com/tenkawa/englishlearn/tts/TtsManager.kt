package com.tenkawa.englishlearn.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

/** Thin wrapper around Android's built-in TextToSpeech engine (no network/API cost). */
class TtsManager(context: Context) {

    private var isReady = false

    private val tts: TextToSpeech = TextToSpeech(context.applicationContext) { status ->
        if (status == TextToSpeech.SUCCESS) {
            isReady = true
        }
    }

    init {
        // Applied once the engine reports SUCCESS; harmless to set before then too.
        tts.language = Locale.US
    }

    fun speak(text: String) {
        if (!isReady) return
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, text.hashCode().toString())
    }

    fun shutdown() {
        tts.stop()
        tts.shutdown()
    }
}
