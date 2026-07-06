package com.tenkawa.englishlearn.ui.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tenkawa.englishlearn.data.WordRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn

data class HomeUiState(
    val dueCount: Int = 0,
    val totalCount: Int = 0,
    val masteredCount: Int = 0
)

class HomeViewModel(repository: WordRepository) : ViewModel() {

    val uiState: StateFlow<HomeUiState> = combine(
        repository.observeDueCount(),
        repository.observeAll(),
        repository.observeMasteredCount()
    ) { due, all, mastered ->
        HomeUiState(dueCount = due, totalCount = all.size, masteredCount = mastered)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), HomeUiState())
}
