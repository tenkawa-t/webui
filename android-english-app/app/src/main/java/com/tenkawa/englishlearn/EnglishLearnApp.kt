package com.tenkawa.englishlearn

import android.app.Application
import android.util.Log
import com.google.android.gms.ads.MobileAds
import com.tenkawa.englishlearn.data.WordRepository
import com.tenkawa.englishlearn.tts.TtsManager
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

private const val TAG = "EnglishLearnApp"

class EnglishLearnApp : Application() {

    private val exceptionHandler = CoroutineExceptionHandler { _, throwable ->
        Log.e(TAG, "Uncaught error in application scope", throwable)
    }

    val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.Default + exceptionHandler)

    val repository: WordRepository by lazy { WordRepository.getInstance(this) }
    val ttsManager: TtsManager by lazy { TtsManager(this) }

    override fun onCreate() {
        super.onCreate()
        try {
            MobileAds.initialize(this)
        } catch (t: Throwable) {
            Log.e(TAG, "MobileAds.initialize failed", t)
        }
        applicationScope.launch { repository.ensureSeeded() }
    }
}
