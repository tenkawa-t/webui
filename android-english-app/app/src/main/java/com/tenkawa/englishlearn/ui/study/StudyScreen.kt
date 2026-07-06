package com.tenkawa.englishlearn.ui.study

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.tenkawa.englishlearn.data.WordRepository
import com.tenkawa.englishlearn.srs.LeitnerScheduler
import com.tenkawa.englishlearn.tts.TtsManager
import com.tenkawa.englishlearn.ui.SimpleViewModelFactory
import com.tenkawa.englishlearn.ui.ads.BannerAd

@Composable
fun StudyScreen(
    repository: WordRepository,
    ttsManager: TtsManager,
    onFinished: () -> Unit
) {
    val viewModel: StudyViewModel = viewModel(factory = SimpleViewModelFactory { StudyViewModel(repository) })
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(uiState.isFinished) {
        if (uiState.isFinished) onFinished()
    }

    LaunchedEffect(uiState.currentWord?.id) {
        uiState.currentWord?.let { ttsManager.speak(it.word) }
    }

    Scaffold(bottomBar = { BannerAd() }) { padding: PaddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(20.dp)
        ) {
            if (uiState.queue.isNotEmpty()) {
                LinearProgressIndicator(
                    progress = { uiState.currentIndex / uiState.queue.size.toFloat() },
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text("残り ${uiState.remaining} 枚")
            }

            Spacer(modifier = Modifier.height(16.dp))

            val word = uiState.currentWord
            if (word != null) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = word.word,
                                style = MaterialTheme.typography.headlineLarge,
                                fontWeight = FontWeight.Bold
                            )
                            IconButton(onClick = { ttsManager.speak(word.word) }) {
                                Icon(Icons.Filled.PlayArrow, contentDescription = "発音を再生")
                            }
                        }

                        if (uiState.isAnswerShown) {
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(word.meaningJa, style = MaterialTheme.typography.titleLarge)
                            Spacer(modifier = Modifier.height(24.dp))
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(word.exampleEn, style = MaterialTheme.typography.bodyLarge)
                                IconButton(onClick = { ttsManager.speak(word.exampleEn) }) {
                                    Icon(Icons.Filled.PlayArrow, contentDescription = "例文を再生")
                                }
                            }
                            Text(
                                word.exampleJa,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(20.dp))

                if (!uiState.isAnswerShown) {
                    Button(onClick = { viewModel.revealAnswer() }, modifier = Modifier.fillMaxWidth()) {
                        Text("答えを見る")
                    }
                } else {
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        GradeButton("もう一度", Color(0xFFE53935), Modifier.weight(1f)) {
                            viewModel.grade(LeitnerScheduler.Grade.AGAIN)
                        }
                        GradeButton("難しい", Color(0xFFFB8C00), Modifier.weight(1f)) {
                            viewModel.grade(LeitnerScheduler.Grade.HARD)
                        }
                        GradeButton("普通", Color(0xFF1E88E5), Modifier.weight(1f)) {
                            viewModel.grade(LeitnerScheduler.Grade.GOOD)
                        }
                        GradeButton("簡単", Color(0xFF43A047), Modifier.weight(1f)) {
                            viewModel.grade(LeitnerScheduler.Grade.EASY)
                        }
                    }
                }
            } else if (!uiState.isLoading) {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text("今日の復習カードはありません！")
                }
            }
        }
    }
}

@Composable
private fun GradeButton(label: String, color: Color, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        modifier = modifier,
        colors = ButtonDefaults.buttonColors(containerColor = color)
    ) {
        Text(label)
    }
}
