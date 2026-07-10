package com.tenkawa.englishlearn.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface WordDao {

    @Query("SELECT COUNT(*) FROM words")
    suspend fun count(): Int

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAll(words: List<Word>)

    @Update
    suspend fun update(word: Word)

    @Query("SELECT * FROM words ORDER BY word ASC")
    fun observeAll(): Flow<List<Word>>

    @Query("SELECT * FROM words WHERE nextReviewAt <= :now ORDER BY nextReviewAt ASC LIMIT :limit")
    suspend fun getDueWords(now: Long, limit: Int): List<Word>

    @Query("SELECT COUNT(*) FROM words WHERE nextReviewAt <= :now")
    fun observeDueCount(now: Long): Flow<Int>

    @Query("SELECT COUNT(*) FROM words WHERE box >= 5")
    fun observeMasteredCount(): Flow<Int>
}
