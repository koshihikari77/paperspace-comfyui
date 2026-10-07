# Wan INT8標準化（2026-10-07）

Paperspaceの標準をnative INT8 ConvRotへ変更。

- start.ipynb: DOWNLOAD_WAN_MODELS=True、MODEL_ROOT=/storage/ComfyUI/models、SageAttention使用。
- ダウンローダーの未指定グループおよびbootstrapのWan取得グループをwan22-int8-coreへ変更。
- bootstrap: shared CUDA12カーネル検証は既定有効。既存のH3用ローカルビルドを共用。
- koshi-custom-nodes/wan_nativeに3ノードを実装。計測フックは含まない。
- 標準API/UIはworkflows/wan22_int8_i2v_api.json / wan22_int8_i2v.json。
- Notebookの追加LoRAプリセットをactive APIに反映。High追加LoRAはprestepにも適用。UIにも同じ選択を反映。
- agent-harnessのWanスキル・生成スクリプトを更新。調査記録はarchiveに分離。
- 検証用5重みを/tmpから/storage/ComfyUI/modelsへ移動し、重複コピーを残さない。41.48GBの永続モデル配置。既存VAE/upscaler/RIFEを再利用。

## 検証

通常ポート6006をキュー空の状態で再起動し、通常の全カスタムノード・監視を有効にした状態で実行。
カメラLoRA .8、seed22222、prestep1+High3+Low3、NAG/FETA、色合わせ、x2拡大、RIFE49 x4。
サーバーhistory実行時間316.027秒（初回、ノードキャッシュなし）。前回の限定ノード検証サーバー232.221秒とは実行環境・モデル配置先が異なり、差の寄与は未分離。
完成MP4は1024×1024、321枚、64fps。FFmpeg全デコード成功。
ファイル: /notebooks/wan22-int8-end-to-end/installed/installed_validation_00001.mp4
証跡: /notebooks/wan22-int8-end-to-end/installed-history.json

ブラウザーでAPIグラフからUIを作成し、UIを再ロードしてAPIへ戻した結果が全29ノード・全入力で一致（IDのコロン正規化と表示タイトルを除く）。
NAGはpositive-only / negative-only / 両順序のbatchでテスト成功。追加LoRAのHigh/Low振り分けとprestep接続、標準6重みの存在、Notebook/Python構文を確認。
旧skillのuser_invocable/argumentsフロントマターは既存互換用に保持。skill-creator汎用validatorはこの既存拡張を許可しないため、その2項目を除いた一時コピーで本文・必須項目を検証。

通常6006は稼働中。旧FP8モデルや旧ワークフローの削除は行っていない。
通常制作の全LoRAの品質確認、新規Docker環境でのCUDA12ローカルカーネル再構築の検証は含まない。
