package com.tenkawa.englishlearn.data

import android.content.Context
import com.tenkawa.englishlearn.srs.LeitnerScheduler
import kotlinx.coroutines.flow.Flow

class WordRepository(private val context: Context, private val dao: WordDao) {

    suspend fun ensureSeeded() {
        if (dao.count() == 0) {
            dao.insertAll(SeedDataLoader.loadSeedWords(context))
        }
    }

    fun observeAll(): Flow<List<Word>> = dao.observeAll()

    fun observeDueCount(now: Long = System.currentTimeMillis()): Flow<Int> = dao.observeDueCount(now)

    fun observeMasteredCount(): Flow<Int> = dao.observeMasteredCount()

    suspend fun getDueWords(limit: Int = 20, now: Long = System.currentTimeMillis()): List<Word> =
        dao.getDueWords(now, limit)

    suspend fun submitAnswer(word: Word, grade: LeitnerScheduler.Grade) {
        dao.update(LeitnerScheduler.applyGrade(word, grade))
    }

    companion object {
        @Volatile
        private var instance: WordRepository? = null

        fun getInstance(context: Context): WordRepository {
            return instance ?: synchronized(this) {
                instance ?: WordRepository(
                    context.applicationContext,
                    AppDatabase.getInstance(context).wordDao()
                ).also { instance = it }
            }
        }
    }
}
