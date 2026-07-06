package com.tenkawa.englishlearn.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.tenkawa.englishlearn.data.WordRepository
import com.tenkawa.englishlearn.tts.TtsManager
import com.tenkawa.englishlearn.ui.home.HomeScreen
import com.tenkawa.englishlearn.ui.study.StudyScreen
import com.tenkawa.englishlearn.ui.wordlist.WordListScreen

object Routes {
    const val HOME = "home"
    const val STUDY = "study"
    const val WORD_LIST = "word_list"
}

@Composable
fun AppNavHost(
    repository: WordRepository,
    ttsManager: TtsManager,
    navController: NavHostController = rememberNavController()
) {
    NavHost(navController = navController, startDestination = Routes.HOME) {
        composable(Routes.HOME) {
            HomeScreen(
                repository = repository,
                onStartStudy = { navController.navigate(Routes.STUDY) },
                onBrowseWords = { navController.navigate(Routes.WORD_LIST) }
            )
        }
        composable(Routes.STUDY) {
            StudyScreen(
                repository = repository,
                ttsManager = ttsManager,
                onFinished = { navController.popBackStack() }
            )
        }
        composable(Routes.WORD_LIST) {
            WordListScreen(repository = repository)
        }
    }
}
