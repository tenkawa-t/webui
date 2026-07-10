package com.tenkawa.englishlearn.ui.ads

import android.util.Log
import android.view.View
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.google.android.gms.ads.AdRequest
import com.google.android.gms.ads.AdSize
import com.google.android.gms.ads.AdView

private const val TAG = "BannerAd"

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
            try {
                AdView(context).apply {
                    setAdSize(AdSize.BANNER)
                    this.adUnitId = adUnitId
                    loadAd(AdRequest.Builder().build())
                }
            } catch (t: Throwable) {
                // A missing/broken Google Play services install shouldn't take the whole app down.
                Log.e(TAG, "Failed to create banner ad", t)
                View(context)
            }
        }
    )
}
