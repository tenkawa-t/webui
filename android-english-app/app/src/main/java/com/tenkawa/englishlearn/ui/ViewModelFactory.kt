package com.tenkawa.englishlearn.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider

/** Minimal generic factory to avoid pulling in a DI framework for a single-repository app. */
class SimpleViewModelFactory<T : ViewModel>(private val create: () -> T) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <VM : ViewModel> create(modelClass: Class<VM>): VM = create() as VM
}
