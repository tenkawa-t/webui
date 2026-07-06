package com.tenkawa.englishlearn.data

import android.content.Context
import org.json.JSONObject

object SeedDataLoader {

    /** Loads app/src/main/assets/words.json into a list of [Word] rows with box=1 (due now). */
    fun loadSeedWords(context: Context): List<Word> {
        val json = context.assets.open("words.json").bufferedReader().use { it.readText() }
        val root = JSONObject(json)
        val array = root.getJSONArray("words")
        return (0 until array.length()).map { i ->
            val entry = array.getJSONObject(i)
            Word(
                word = entry.getString("word"),
                meaningJa = entry.getString("meaningJa"),
                exampleEn = entry.getString("exampleEn"),
                exampleJa = entry.getString("exampleJa"),
                category = entry.getString("category"),
                box = 1,
                nextReviewAt = 0L
            )
        }
    }
}
