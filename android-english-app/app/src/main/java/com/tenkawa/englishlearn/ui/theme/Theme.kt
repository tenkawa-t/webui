package com.tenkawa.englishlearn.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Blue = Color(0xFF1565C0)
private val BlueDark = Color(0xFF90CAF9)

private val LightColors = lightColorScheme(
    primary = Blue,
    secondary = Color(0xFF00897B)
)

private val DarkColors = darkColorScheme(
    primary = BlueDark,
    secondary = Color(0xFF4DB6AC)
)

@Composable
fun EnglishLearnTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colors = if (darkTheme) DarkColors else LightColors
    MaterialTheme(colorScheme = colors, content = content)
}
