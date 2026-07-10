# 英単語トレーナー (English Word Trainer)

副業アプリのMVP。Kotlin + Jetpack Composeで作った、間隔反復（ライトナー式）の
英単語フラッシュカード学習アプリです。単語・例文はAndroid標準のTTS（音声合成）
で読み上げられ、AdMobバナー広告で収益化する構成になっています。

## 機能

- **単語フラッシュカード**: 単語 → 意味・例文の順に表示するカード学習
- **ライトナー式間隔反復 (SRS)**: 回答の自己評価（もう一度/難しい/普通/簡単）に
  応じてBox 1〜5を上下させ、次回復習日を自動計算 (`srs/LeitnerScheduler.kt`)
- **例文・リスニング**: 各単語に例文（英語・日本語訳）を収録し、Android標準の
  `TextToSpeech` で単語・例文を読み上げ（外部API不要・追加コストなし）
- **単語一覧・検索画面**: 収録済み87語を検索・閲覧
- **AdMobバナー広告**: ホーム／学習／一覧画面下部に表示

## 技術スタック

- Kotlin, Jetpack Compose, Material3
- Room（単語データの永続化。初回起動時に`assets/words.json`からシード）
- Navigation Compose
- Google Mobile Ads SDK (AdMob)
- Android標準 `TextToSpeech`（無料・追加APIキー不要）

DI用フレームワーク（Hilt等）はMVPの規模には過剰なため使わず、
`EnglishLearnApp`（Applicationクラス）でRepository/TtsManagerをシングルトンとして保持する
シンプルな構成にしています。

## ディレクトリ構成

```
app/src/main/java/com/tenkawa/englishlearn/
├── data/        Room Entity/DAO/Database/Repository, JSONシードローダー
├── srs/         ライトナー式スケジューラ
├── tts/         TextToSpeechラッパー
├── ui/
│   ├── home/      ホーム画面（学習開始・進捗表示）
│   ├── study/     フラッシュカード学習画面
│   ├── wordlist/  単語一覧・検索画面
│   ├── ads/       AdMobバナーComposable
│   ├── theme/     Compose テーマ
│   └── navigation/ NavHost
├── EnglishLearnApp.kt  Applicationクラス（DB初期化・広告初期化）
└── MainActivity.kt
app/src/main/assets/words.json  収録単語データ（日常/ビジネス/旅行/感情/動詞、87語）
```

## ビルド方法

このプロジェクトにはGradle Wrapper（`gradlew`）を含めていません。開発に使った
サンドボックス環境ではAndroidのMavenリポジトリ（dl.google.com）とGradle配布サーバー
（services.gradle.org）の両方がネットワークポリシーでブロックされており、
wrapperの生成もビルド検証もできなかったためです。ソースコードは手動で入念に
レビュー済みですが、実機でのビルド確認はまだ行えていません。

1. [Android Studio](https://developer.android.com/studio) をインストール
2. `android-english-app/` フォルダを開く
   （Gradle Wrapperが無い場合、Android StudioがWrapperの生成を提案するので従う）
3. Gradle Sync完了後、実機またはエミュレータで実行（Run ▶️）

初回起動時にビルドエラーが出た場合は、依存ライブラリのバージョン
（AGP 8.5.0 / Kotlin 1.9.24 / Compose BOM 2024.06.00 など、`build.gradle.kts`参照）を
Android Studioが提案する最新の組み合わせに更新してください。

## 本番リリース前に必ずやること

現在、AdMobは**Googleのテスト用ID**を使っています。このまま公開すると
広告が表示されなかったり、テストトラフィック扱いになり収益が発生しません。

1. [AdMob](https://admob.google.com/) でアプリを登録し、アプリIDとバナー広告ユニットIDを取得
2. `app/src/main/AndroidManifest.xml` の `com.google.android.gms.ads.APPLICATION_ID` を
   自分のAdMobアプリIDに差し替え
3. `app/src/main/java/com/tenkawa/englishlearn/ui/ads/BannerAd.kt` の
   `TEST_BANNER_UNIT_ID` を自分のバナー広告ユニットIDに差し替え
4. `app/build.gradle.kts` の `applicationId` を自分のパッケージ名に変更（Google Play公開時は
   一度公開すると変更できないため注意）
5. 署名用のリリースキーストアを作成し、`signingConfigs` を設定
6. Google Playデベロッパーアカウント（登録料が必要）でアプリを公開

## 今後の拡張アイデア

- 単語データの追加（TOEIC頻出語、カテゴリ別コース化など）
- 学習ストリーク（連続学習日数）や通知によるリマインド
- 買い切り/サブスクでの広告非表示オプション（収益源の多様化）
- 発音の自己録音・比較機能
