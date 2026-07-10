package com.tenkawa.englishlearn.ui.study

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tenkawa.englishlearn.data.Word
import com.tenkawa.englishlearn.data.WordRepository
import com.tenkawa.englishlearn.srs.LeitnerScheduler
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class StudyUiState(
    val queue: List<Word> = emptyList(),
    val currentIndex: Int = 0,
    val isAnswerShown: Boolean = false,
    val isLoading: Boolean = true,
    val isFinished: Boolean = false
) {
    val currentWord: Word? get() = queue.getOrNull(currentIndex)
    val remaining: Int get() = (queue.size - currentIndex).coerceAtLeast(0)
}

class StudyViewModel(private val repository: WordRepository) : ViewModel() {

    private val _uiState = MutableStateFlow(StudyUiState())
    val uiState: StateFlow<StudyUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val due = repository.getDueWords(limit = 20)
            _uiState.value = StudyUiState(queue = due, isLoading = false, isFinished = due.isEmpty())
        }
    }

    fun revealAnswer() {
        _uiState.value = _uiState.value.copy(isAnswerShown = true)
    }

    fun grade(grade: LeitnerScheduler.Grade) {
        val state = _uiState.value
        val word = state.currentWord ?: return
        viewModelScope.launch {
            repository.submitAnswer(word, grade)
            val nextIndex = state.currentIndex + 1
            _uiState.value = state.copy(
                currentIndex = nextIndex,
                isAnswerShown = false,
                isFinished = nextIndex >= state.queue.size
            )
        }
    }
}
