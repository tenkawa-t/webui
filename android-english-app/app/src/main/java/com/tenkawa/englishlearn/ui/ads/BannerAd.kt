package com.tenkawa.englishlearn.ui.ads

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.google.android.gms.ads.AdRequest
import com.google.android.gms.ads.AdSize
import com.google.android.gms.ads.AdView

/**
 * Displays a banner ad.
 *
 * NOTE: [TEST_BANNER_UNIT_ID] is Google's public test ad unit ID. Replace it with your own
 * AdMob banner ad unit ID (and the app ID in AndroidManifest.xml) before publishing.
 */
const val TEST_BANNER_UNIT_ID = "ca-app-pub-3940256099942544/6300978111"

@Composable
fun BannerAd(modifier: Modifier = Modifier, adUnitId: String = TEST_BANNER_UNIT_ID) {
    AndroidView(
        modifier = modifier.fillMaxWidth(),
        factory = { context ->
            AdView(context).apply {
                setAdSize(AdSize.BANNER)
                this.adUnitId = adUnitId
                loadAd(AdRequest.Builder().build())
            }
        }
    )
}
