package com.tenkawa.englishlearn.ui.wordlist

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.tenkawa.englishlearn.data.WordRepository
import com.tenkawa.englishlearn.ui.SimpleViewModelFactory
import com.tenkawa.englishlearn.ui.ads.BannerAd

@Composable
fun WordListScreen(repository: WordRepository) {
    val viewModel: WordListViewModel = viewModel(factory = SimpleViewModelFactory { WordListViewModel(repository) })
    val words by viewModel.words.collectAsStateWithLifecycle()
    var query by remember { mutableStateOf("") }

    val filtered = if (query.isBlank()) words else words.filter {
        it.word.contains(query, ignoreCase = true) || it.meaningJa.contains(query)
    }

    Scaffold(bottomBar = { BannerAd() }) { padding: PaddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                label = { Text("検索") },
                modifier = Modifier.fillMaxWidth()
            )

            LazyColumn {
                items(filtered, key = { it.id }) { word ->
                    ListItem(
                        headlineContent = { Text(word.word) },
                        supportingContent = { Text(word.meaningJa) },
                        trailingContent = { Text("Box ${word.box}") }
                    )
                    HorizontalDivider()
                }
            }

            if (filtered.isEmpty()) {
                Text(
                    "単語がありません",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
    }
}
