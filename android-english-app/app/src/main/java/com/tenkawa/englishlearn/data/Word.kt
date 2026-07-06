package com.tenkawa.englishlearn.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "words")
data class Word(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val word: String,
    val meaningJa: String,
    val exampleEn: String,
    val exampleJa: String,
    val category: String,
    // Leitner box: 1 (new/hardest) .. 5 (mastered)
    val box: Int = 1,
    val nextReviewAt: Long = 0L,
    val correctStreak: Int = 0
)
