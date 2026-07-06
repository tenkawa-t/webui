package com.tenkawa.englishlearn.ui.wordlist

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tenkawa.englishlearn.data.Word
import com.tenkawa.englishlearn.data.WordRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn

class WordListViewModel(repository: WordRepository) : ViewModel() {

    val words: StateFlow<List<Word>> = repository.observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
}
