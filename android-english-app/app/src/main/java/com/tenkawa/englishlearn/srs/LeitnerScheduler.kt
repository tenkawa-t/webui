package com.tenkawa.englishlearn.srs

import com.tenkawa.englishlearn.data.Word
import java.util.concurrent.TimeUnit

/**
 * Simple 5-box Leitner spaced-repetition system.
 * Box 1 = new/hardest (review daily), Box 5 = mastered (review every 16 days).
 */
object LeitnerScheduler {

    enum class Grade { AGAIN, HARD, GOOD, EASY }

    private val boxIntervalDays = mapOf(
        1 to 0L,   // due immediately / same day
        2 to 1L,
        3 to 3L,
        4 to 7L,
        5 to 16L
    )

    private const val MIN_BOX = 1
    private const val MAX_BOX = 5

    fun applyGrade(word: Word, grade: Grade): Word {
        val newBox = when (grade) {
            Grade.AGAIN -> MIN_BOX
            Grade.HARD -> (word.box - 1).coerceAtLeast(MIN_BOX)
            Grade.GOOD -> (word.box + 1).coerceAtMost(MAX_BOX)
            Grade.EASY -> (word.box + 2).coerceAtMost(MAX_BOX)
        }
        val newStreak = if (grade == Grade.AGAIN) 0 else word.correctStreak + 1
        val intervalDays = boxIntervalDays[newBox] ?: 0L
        val nextReview = System.currentTimeMillis() + TimeUnit.DAYS.toMillis(intervalDays)
        return word.copy(box = newBox, nextReviewAt = nextReview, correctStreak = newStreak)
    }
}
