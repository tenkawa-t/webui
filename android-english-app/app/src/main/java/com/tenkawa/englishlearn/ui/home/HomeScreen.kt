package com.tenkawa.englishlearn.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.tenkawa.englishlearn.data.WordRepository
import com.tenkawa.englishlearn.ui.SimpleViewModelFactory
import com.tenkawa.englishlearn.ui.ads.BannerAd

@Composable
fun HomeScreen(
    repository: WordRepository,
    onStartStudy: () -> Unit,
    onBrowseWords: () -> Unit
) {
    val viewModel: HomeViewModel = viewModel(factory = SimpleViewModelFactory { HomeViewModel(repository) })
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Scaffold(
        bottomBar = { BannerAd() }
    ) { padding: PaddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "英単語トレーナー",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(24.dp))

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Text("今日の復習カード：${uiState.dueCount} 枚")
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("習得済み：${uiState.masteredCount} / ${uiState.totalCount} 語")
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            Button(
                onClick = onStartStudy,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (uiState.dueCount > 0) "学習を始める" else "復習カードを見る")
            }

            Spacer(modifier = Modifier.height(12.dp))

            OutlinedButton(
                onClick = onBrowseWords,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("単語一覧を見る")
            }
        }
    }
}
