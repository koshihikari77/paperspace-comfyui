# Wan2.2 INT8 ConvRot：A6000実機検証（2026-10-07）

> 最新追試: [prestep付き7stepを維持したINT8候補を3seedで確認](wan22-int8-prestep-2026-10-07.md)。前段の「prestep付きは採用見送り」「4stepを候補」は更新済み。以下は当時の検証記録。

追試: [従来機能の互換検証](wan22-int8-equivalence-2026-10-07.md)。native 4stepへのNAG・追加LoRA・refine・拡大/補間を確認。以下は第一段階の同条件速度比較記録です。

対象：[paperspace-comfyui Issue #12](https://github.com/koshihikari77/paperspace-comfyui/issues/12)。

## 結論

**ComfyUI標準ノードのWan2.2 I2V＋専用LightX2V 4step＋SageAttentionでは、INT8 ConvRotが実速度向上に効いた。** 同じ画像・設定・3 seedで、連続生成の総時間中央値はFP8 **43.09秒 → INT8 33.08秒（23.2%短縮、約1.30倍）**。サンプラーノードの合計中央値は **37.18秒 → 27.34秒（26.5%短縮）**。

ただし、**今までのWanVideoWrapper 7step構成の重みを差し替えただけの結果ではない**。現行WrapperにはこのINT8 ConvRot形式を解釈する経路が見当たらず、標準ノード側で対照実験を行った。既存NAG/FETA/prestep構成を標準ノードへ移した試作は色ノイズが残り、そのままの本番置換には採用しない。

採用候補は、下記の**独立した4step I2Vワークフロー**。従来の7stepワークフローは変更していない。ユーザーによる動画の採用判定は未実施。

## 同条件比較

| 設定 | FP8 / INT8共通 |
|---|---|
| GPU | RTX A6000 48GB、Ampere SM86 |
| 出力 | 512×512、81フレーム、16fps、5.0625秒、無音 |
| 入力 | Issue #3と同じ銀髪・緑の服の人物画像。中央クロップ |
| Prompt | 2歩前へ進み、カメラに向き、右手を上げる。人物・服・部屋・照明・カメラを維持 |
| Seed | 比較11111 / 22222 / 33333。準備実行44444は集計から除外 |
| Sampler / scheduler | Euler / simple |
| Steps / expert切替 | 4step、High 0–2 → Low 2–4 |
| CFG / shift | 1 / 5 |
| LoRA | Wan2.2 I2V LightX2V 4steps v1、High/Low専用、各strength 1 |
| Attention | SageAttention 1.0.6、`--use-sage-attention` |
| T5 | 既存UMT5 XXL BF16、量子化なし |
| VAE | 残存する`wan_2.1_vae.safetensors`、両条件同一 |
| NAG / FETA / prestep | なし |
| BlockSwap | WrapperのBlockSwapノードなし。ComfyUI標準の動的VRAM管理 |
| Upscale / RIFE | なし |
| ノードキャッシュ | 通常のRAM pressure cache。モデル、T5、入力条件を再利用。サンプラーとデコードは毎回実行 |
| 計算精度 | native loaderの自動選択、両方manual cast FP16。T5/VAE BF16 |

FP8とINT8で変更した計算用パラメータは**High/Lowの2モデル名のみ**。出力ファイル名の違いを除いた機械的差分は[paired-workflow-diff.json](../wan22-int8-results/paired-workflow-diff.json)。

### 3 seedの実測

| Seed | FP8総時間 | INT8総時間 | FP8 sampler合計 | INT8 sampler合計 |
|---|---:|---:|---:|---:|
| 11111 | 42.34秒 | 32.11秒 | 36.49秒 | 26.71秒 |
| 22222 | 43.09秒 | 33.08秒 | 37.18秒 | 27.34秒 |
| 33333 | 43.29秒 | 33.90秒 | 37.31秒 | 27.54秒 |
| **中央値** | **43.09秒** | **33.08秒** | **37.18秒** | **27.34秒** |

| 項目 | FP8 | INT8 |
|---|---:|---:|
| 観測ピークGPUメモリ | 31,102MiB（30.37GiB） | 31,626MiB（30.88GiB） |
| デコード中央値 | 4.31秒 | 4.27秒 |
| プロセスRSS最大（3回中） | 約13.34GiB | 約13.41GiB |
| CUDA INT8 linear呼出 / 動画 | 0 | **1,600回** |
| INT8 linearの他backend呼出 / 動画 | 0 | **0回** |

INT8のGPUメモリは**約0.51GiB多かった**。今回は省VRAM目的ではなく速度目的の候補。FP8も8bitのため、INT8に替えて重みが半分になるわけではない。High/Low合計のファイル容量もFP8約28.59GB、INT8約29.07GB。

### 計測の意味・限界

- 総時間はAPI投入後から履歴完了を確認するまで。1秒間隔ポーリングにより最大約1秒の検出遅延を含む。キューは空の状態で1本ずつ実行した。
- ノード時間は前後でCUDA同期。Sampler時間には必要なモデル転送・LoRA適用・初期化を含む。純粋な行列演算時間ではない。
- `nvidia-smi`を約1秒間隔で取得したGPU全体の観測ピーク。短いピークは取り逃す可能性がある。通常サーバーはアイドルでモデルをアンロードした状態。
- キャッシュされた13ノードにサンプラー・デコード・保存は含まれない。INT8は毎回1,600呼出を記録し、生成が省略されていないことを確認した。
- 順序はFP8 3回→INT8 3回。順序を反転した試験はしていない。3 seedは同じ入力画像であり、幅広いシーンの品質保証ではない。
- 初期探索は`--cache-none`。その4step試験ではFP8 108.17秒、INT8 92.49秒だった。T5やモデルの読み込みが入り、上の連続生成とは別条件。
- 通常cacheの準備実行はFP8 126.76秒、INT8 62.89秒。ただしINT8の準備時はT5/入力がすでにキャッシュ済みなので、この2値でcold速度倍率を計算しない。OSページキャッシュを強制消去したcold試験でもない。

## CUDA INT8 / SageAttention / LoRA

開始時に`verify_minimax_kernel.py`のGEMM・ConvRot単体テストを実行し成功。既存のCUDA12用ローカルkernelと、プロセス用`libcublasLt.so`エイリアスを使った。

```bash
LD_LIBRARY_PATH=/storage/ComfyUI/.h3-cuda/lib
H3_LOCAL_CUDA12_KERNELS=1
```

生成中はComfy Kitchenのbackend選択を記録。全3動画で`comfy_kitchen.backends.cuda.int8_linear`が各1,600回、他のINT8 linear backendは0回。モデル全層がINT8という意味ではなく、量子化対象外の層は浮動小数点のまま。

SageAttentionを有効にしたサーバーで完走し、最終測定ログにSageAttentionエラーによるfallbackは見られなかった。Wan2.2専用LightX2Vは各expertに400 patchesが付き、INT8経路を保ったまま動作した。すべての任意LoRAに互換性があるという検証ではない。

## 品質と比較動画

**6動画すべて512×512・81f・16fpsをffprobeで確認し、全フレームのFFmpegデコードが成功。** 3 seedそれぞれ、先頭・20・40・60・80フレームの静止画を比較した。

- 両方式で銀髪、緑の服、部屋の壁・柱・箱、前進と手上げが保たれた。
- 最終4step構成の抽出フレームでは、試作7stepで出た色ノイズや、旧LoRAの4step試験で出た大きな残像は見られなかった。
- 同じseedでも、手の位置・腕の軌道・体の向き・髪の動きは異なる。INT8を画素一致の置き換えとは扱わない。
- 顔や指の細かな形、全編再生時の小さなちらつき、複数シーンでの採用率は未判定。静止画確認と全フレームの機械的デコードは、全編を人が視聴した品質審査とは異なる。

横並び動画は**左FP8、右INT8**：

- [seed11111](../wan22-int8-results/videos/compare_seed11111_left-fp8_right-int8.mp4)
- [seed22222](../wan22-int8-results/videos/compare_seed22222_left-fp8_right-int8.mp4)
- [seed33333](../wan22-int8-results/videos/compare_seed33333_left-fp8_right-int8.mp4)

元の7step Wrapper：[今回のbaseline動画](../wan22-int8-results/videos/wrapper_cold_00001.mp4)。

## 既存7step構成との関係・失敗の記録

既存baselineはFP8＋SageAttention＋旧LightX2V、pre 1＋High 3＋Low 3、CFG 3.5/2/1/1/1/1/1、shift8、NAG 11/2.5/0.25、FETA1、BlockSwap20。旧VAEは削除済みなので、残存VAEを使って今回再測定した。初回308.36秒、GPUピーク12,052MiB。これはcoldの1本であり、最終native4stepのwarm33秒との倍率をINT8の効果として示してはいけない。

| 試験 | 結果・判断 |
|---|---|
| 既存Wrapper FP8 | 7step動画を完走。従来構成のbaselineとして保存 |
| native 7step、初期T5 adapter | FP8/INT8とも画像が大きく崩れた。比較から除外 |
| T5 adapterを修正したnative INT8 7step | 人物は復元されたが色付きノイズが残った。不採用 |
| native INT8 4step、旧LoRA strength1、NAG/FETA/prestepなし | 色ノイズは消えたが大きな残像。不採用 |
| native INT8 4step、Wan2.2専用LoRA | 最終候補。FP8対照と各3 seedを測定 |

T5 adapterはWrapperの可変長埋め込みを標準CONDITIONINGへ渡すもの。最初はWrapper内で行われる512トークンへのゼロpaddingが欠けていたため、修正した。重み自体は変えていない。

7step側の残る色ノイズについては、NAG/FETA、LoRA強度、sampler/prestepなどを個別に戻す切り分けまではしていない。**NAGやINT8 LoRAが単独原因と断定しない。** WrapperのschedulerはFlowDPMSolverMultistepScheduler、試作nativeはdpmpp_2m_sde/simple。CFG再現の分割境界でsolver履歴も異なる。したがってnative試作を元の完全移植とは呼ばない。

## 推奨・残項目

- **速度重視の新しいI2V候補として、標準ノード＋INT8 ConvRot High/Low＋Wan2.2用LightX2V 4step＋SageAttentionを使う価値がある。** Issueの速度目安「sampling 20%以上短縮」を今回の対照試験では満たした。
- まず横並び動画で好みと品質を確認する。採用可能な1本あたりのGPU時間は、人の採用判定・再試行数を取っていないため未評価。
- NAG/FETA/prestepを含む従来7stepを維持したい場合は、現時点ではWrapper FP8を維持。重みを替えるだけの移行はしない。
- T5 INT8、追加の画風・動作LoRA、解像度変更、長尺、BlockSwap量の掃引、RIFE/upscale仕上げは今回未検証。
- 本番ワークフロー、Notebookのダウンロード設定、通常サーバーの起動設定は変更していない。

## 再現・保存場所

実測一式：[`wan22-int8-results`](../wan22-int8-results/)。

- [FP8 API workflow](../wan22-int8-results/official4_fp8.json) / [INT8 API workflow](../wan22-int8-results/official4_int8.json)
- [全測定CSV](../wan22-int8-results/benchmark.csv)（失敗映像もプロセス上はsuccessなので、品質判断は本文・ラベルを参照）
- [ノード時間・INT8 backendログ](../wan22-int8-results/node-metrics.jsonl)
- [環境](../wan22-int8-results/environment.json) / [生成モデル・T5等のrevision/hash](../wan22-int8-results/model-manifest.json) / [専用LoRAのrevision/hash](../wan22-int8-results/native-lora-manifest.json)
- [入力/VAEのSHA-256](../wan22-int8-results/input-vae.sha256)
- [再現用起動スクリプト](../wan22-int8-results/reproduce.sh) / [実行スクリプト](../wan22-int8-results/run.py) / [3 seedのbatch](../wan22-int8-results/batch.py)

環境：ComfyUI0.37.0、torch2.14.0+cu126、comfy-kitchen0.2.35、SageAttention1.0.6、Triton3.8.0、driver550.144.03。RAMのcgroup上限40,896MiB。ソースはGit情報が残っていないディレクトリのため、関連実装のファイルhashを保存した。

検証専用サーバー6007は停止。**今回だけ取得したモデル8ファイル・72.22GBは検証後に削除**した。再実行時はmanifestから再取得が必要。通常サーバー6006は停止していない。動画・グラフ・数値・ログは残した。GitHub Issue本文・コメントへの投稿、commit/pushはこの実施では行っていない。

一次情報：[INT8配布元](https://huggingface.co/Winnougan/Wan2.2-INT8-Convrot)、[Comfy-OrgのWan2.2配布](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)、[既存T5/LoRA配布元](https://huggingface.co/Kijai/WanVideo_comfy)。
