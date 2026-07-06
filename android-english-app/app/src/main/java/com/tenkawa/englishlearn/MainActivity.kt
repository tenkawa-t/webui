package com.tenkawa.englishlearn

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.tenkawa.englishlearn.ui.navigation.AppNavHost
import com.tenkawa.englishlearn.ui.theme.EnglishLearnTheme

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val app = application as EnglishLearnApp

        setContent {
            EnglishLearnTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    AppNavHost(repository = app.repository, ttsManager = app.ttsManager)
                }
            }
        }
    }
}
