package com.tenkawa.englishlearn

import android.app.Application
import com.google.android.gms.ads.MobileAds
import com.tenkawa.englishlearn.data.WordRepository
import com.tenkawa.englishlearn.tts.TtsManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class EnglishLearnApp : Application() {

    val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val repository: WordRepository by lazy { WordRepository.getInstance(this) }
    val ttsManager: TtsManager by lazy { TtsManager(this) }

    override fun onCreate() {
        super.onCreate()
        MobileAds.initialize(this)
        applicationScope.launch { repository.ensureSeeded() }
    }
}
