# Wan2.2 INT8: 従来ワークフロー相当の機能検証

> 最新追試: [prestep付き7stepを維持したINT8候補を3seedで確認](wan22-int8-prestep-2026-10-07.md)。前段の「prestep付きは採用見送り」「4stepを候補」は更新済み。以下は当時の検証記録。

検証日: 2026-10-07。Issue: https://github.com/koshihikari77/paperspace-comfyui/issues/12

## 結論

**native INT8 4stepを土台にすれば、I2V・追加LoRA・NAG・FETA・色合わせ/追加refine・2倍upscale・4倍補間という主要機能を組める。**
旧7stepの数値の再現ではなく、この4step構成を次の採用候補にする。
通常環境のworkflowは未変更。全制作LoRA・全入力の品質保証まで終わった、という意味ではない。

| 機能 | 今回の判定 | 根拠・制限 |
|---|---|---|
| I2V + High/Low INT8 + LightX2V + Sage | 確認済み | CUDA INT8 linear実行を計数 |
| NAG | 確認済み | 4step CFG1、3seedで選択フレームに大きな破綻なし |
| FETA | 作用を確認 | 強度1は無効時と画素一致、5で変化。画質向上の保証ではない |
| 追加LoRAと加速LoRAの併用 | 1種類で確認 | カメラLoRA 0/.8で明確な出力変化。全LoRA互換は未検証 |
| 色合わせ + 追加refine | 完走 | 同一seed33333、Low側4step / denoise .15 |
| 2倍upscale → 4倍RIFE | 完走 | 1024×1024 / 321f / 64fpsを保存、全フレームdecode成功 |
| 加速LoRAなしprestep付き7step | 採用見送り | seed22222で背景破綻、他seed成功だけで安定扱いしない |
| Wrapper BlockSwap20のそのまま移植 | 対象外 | native動的メモリ管理を使用。低VRAM適性は未検証 |

## 目的と基準

ユーザー指定は「ノードや数値が全く同じでなくても、同等のことができればよい」。
Floyo WanVideoWrapper 7stepを主な参照元とし、別の `wan22.json` にある色合わせ・追加refineも確認する。
今回の検証は、同じ絵になる保証や全ての制作LoRAの互換保証ではない。

前段の純粋な速度比較は [INT8 benchmark](wan22-int8-benchmark-2026-10-07.md) を参照。
その4step同条件比較では、FP8からINT8への変更でtotal中央値43.093→33.082秒。
この23.2%短縮を、NAGや後処理を追加した構成全体の短縮率としては扱わない。

## 検証構成

- RTX A6000 48GB、driver 550.144.03、ComfyUI 0.37.0、torch 2.14.0+cu126。
- comfy-kitchen 0.2.35、SageAttention 1.0.6、CUDA12用の既設INT8 kernel。
- High/Low: Winnougan Wan2.2 INT8 ConvRot。VAE: Wan2.1 VAE。
- Wan2.2用LightX2V high/low 4step LoRAを各strength1。
- native UNETLoader / KSamplerAdvanced。T5のみ既存WanVideoWrapper BF16形式を再利用し、512tokenまでzero paddingする変換ノードを使用。
- 512×512、81frame、16fps、無音、同じ入力画像。乱数種11111 / 22222 / 33333。
- 主候補: Euler / simple、4step (High2 + Low2)、CFG1、shift5。
- NAG: scale11 / alpha.25 / tau2.5。FETA: weight1（効果確認のみ5も試験）。
- 通常RAM pressure cache。明示的なWrapper BlockSwap20は使わず、native側の動的メモリ管理。

全manifest、API形式workflow、history、ノード時間、動画は `../wan22-int8-equivalence/`。
再現上の注意は同ディレクトリのREADMEを参照。

## 従来設定の直移植ができなかった箇所

### 強制BF16 + NAG

`--bf16-unet` を付けた実験では、nativeモデルのmanual castがFP16なのにNAG側contextがBF16となり、AttentionでHalf/BFloat16不一致エラーになった。
フラグを外して自動精度に戻すと、同NAGノードは動作した。今回のエラーを「INT8ではNAG非対応」と解釈してはいけない。

### prestep付き7step

加速LoRAなしprestep1 + High3 + Low3、shift8をnativeで試した。
元に近いCFG3.5→2→1とNAGを組み合わせるとseed11111で背景が割れるような映像になった。
NAGを外して同CFGを維持した1本は正常、NAGを残して全CFG1にした1本も正常だった。
ただし全CFG1を3seedに広げるとseed22222で背景が大きく破綻。11111 / 33333は選択フレームで重大破綻なし。
従って「CFG1にすれば7stepも安定」とは結論しない。7step移植は不採用、4stepを機能追加の土台にする。
APIでsuccessだったことと、品質が正常だったことは区別した。

### FETAの効果

同seedのデコード済み動画を全画素比較すると、FETA1は無効時と完全一致した（PSNR inf）。NAG併用でも同じ。
FETA5では動画が変化した（PSNR約20.27dB）。選択フレーム上も動きの違いがある。
つまりノードは作用し得るが、この入力で旧strength1を残すだけでは効果は確認できない。
5が一般に最適、画質が向上した、という意味ではない。

## 評価の範囲

動画の全フレームをFFmpegでデコードしてエラーを確認し、各動画の先頭・約1/4・中央・約3/4・末尾の5枚を目視する。
これは全フレームを人が連続再生して、ちらつきや動きの自然さまで承認した評価ではない。
今回の追加LoRA試験はカメラLoRA1種類であり、普段の全LoRAの品質・互換性を保証しない。
長尺・異なる縦横比・複数入力・低VRAM運用は今回の範囲外。

## 生成＋拡大＋補間の実測

`full_finish-workflow.json` は4step＋NAG＋FETA1から最後まで実行したもの。
seed33333、512×512 / 81frame / 16fpsから、2x-AnimeSharpV4 Fast RCAN→RIFE49 multiplier4で仕上げた。
ffprobeで確認した最終出力は **1024×1024 / 321frame / 64fps / 5.015625秒**。
元動画は81/16=5.0625秒。RIFEが終端間を補間するため約0.047秒短い。
ノードのログには324framesと出るが、実ファイルは321frameなので実ファイルを採用する。

| 処理 | 秒 |
|---|---:|
| T5 loader | 2.99 |
| T5 encodeノード（重みロードを含む） | 55.08 |
| I2V conditioning | 5.71 |
| High samplingノード（初期化を含む） | 16.06 |
| Low samplingノード（初期化を含む） | 27.16 |
| VAE decode | 5.61 |
| 元動画保存 | 1.13 |
| upscale model load | 2.24 |
| 2倍upscale | 16.21 |
| RIFE 4倍 | 46.22 |
| 最終動画保存 | 12.80 |
| **ComfyUI開始→完了** | **193.48** |

各ノード外にもメモリ解放等があるので、ノード合計をtotalと混同しない。
この実行はcacheで省略された生成を仕上げただけではなく、生成も再実行した。ただし全てを完全coldに揃えた速度benchmarkでもない。
仕上げノード3個（拡大・RIFE・最終保存）の合計は75.24秒。upscale model load込み77.47秒。

nvidia-smiの約1秒間隔観測ではGPU使用量最大40330MiB（他サーバーのidle contextも含む）。
今回の仕上げ構成は「INT8だからVRAMが少なくて済む」という結果ではない。
サーバーRSS最大約18.1GiB。CPU offload・GPU cacheの状態で変わる。

4step＋NAG＋FETA1はseed11111 / 22222 / 33333の3本で、選択した5フレームに7step seed22222のような大きな破綻は見られなかった。
seed22222の生成単独はComfyUI開始→完了32.83秒。
runner側205.53秒は前の仕上げ処理の待ち行列を含むため、生成時間には使わない。

## 色合わせ・追加refine

`refine-workflow.json`、seed33333で完走。4step生成→ColorMatch MKL strength.5→VAE encode→Lowモデルdenoise .15 / 4step→decodeという構成。
出力は512×512 / 81frame / 16fps。選択フレームでは人物・室内背景が保たれた。
処理を追加できることの確認であり、refine前より画質が良いという優劣判定ではない。

ComfyUI実行時間86.15秒。生成High13.83 + Low13.97秒、元decode3.90秒、ColorMatch19.12秒、再encode2.47秒、refine27.52秒、再decode4.01秒、保存.91秒。
runnerの119.65秒は待ち行列を含む。
refineもCUDA INT8 linear 1920回、Triton等へのINT8 linear fallback記録なし。
ColorMatch内部にcomplex→real変換warningが出たが例外はなく完了。warningはログに保存。

## 追加LoRAと加速LoRAの同時利用

`UnifiedHorusRA/CameraRotation-Wan2.2_2.1-I2V` の `wan22-video13-CameraRotation-16-sel-2.safetensors` をHigh側に追加。
同じ画像・prompt・seed11111でstrength0と.8だけを変更（保存名以外）。
加速LoRA各1、NAG、FETA1も併用した。

LoRAなしはカメラが横に回り込むような出力、.8では画面全体が大きく傾くロールが出た。
追加LoRAの作用は明確だが、「slow orbit」というpromptを厳密に満たしたとは評価しない。強い回転が欲しくなければ別途strength調整が必要。
両者のPSNRは約16.27dBで画素変化あり。API成功、選択フレームで大きな色ノイズ・崩壊なし、全フレームdecode成功。
`.8` の実行45.48秒。`0` は71.93秒だがロード/cache条件が異なるため、LoRA追加が高速化したとは解釈しない。
両者ともCUDA INT8 linear 1920回。未適用LoRA keyの警告なし。
モデルrevision/hash/原作者情報はartifact内のcamera manifest/sourceを参照。

## 記録と運用上の扱い

- API失敗1件（強制BF16 + NAG）も保存。API成功の映像破綻も隠さず保存。
- 保存動画17本は全てFFmpeg全フレームdecode成功。失敗生成を含む実験記録であり、17本全てが品質合格ではない。
- `timings.csv` はqueue待ちを除くサーバー開始→完了。各runはcache条件が混在し、全行をそのまま速度比較しない。
- 全ての記録されたINT8 linear dispatchはCUDA backend。他backendへのINT8 linear dispatchは0。全演算がINT8という意味ではない。
- 一度RSS監視が子プロセスも拾って停止したが、サーバー生成は完了した。該当seed22222のhistoryを回収し、時間はサーバーイベントから算出。監視は最古のサーバーPIDを選ぶよう修正。
- 再現用モデルは一時ディスク上に取得。ハッシュ・workflow・動画・ログを保存し、一時モデルを片付ける。通常6006のモデルや設定は変更しない。

次に本番へ組み込むなら、`finish.json` の4step構成を基本にし、追加LoRAとrefineを必要時だけ使う。
prestep7と強制BF16は今回の推奨構成に含めない。FETA1は今回効果を確認できなかったため、既定で必須にはしない。
機能互換の確認はここまで完了。実際に使用する各LoRAでの出力採用判断と、本番Notebook/UIへの統合は別作業。

検証後処理完了: 6007停止、一時モデル7ファイル計43,934,129,512 bytes（43.93GB）削除。6006はidleのまま維持。動画・workflow・ログは保存済み。

Issue追記: https://github.com/koshihikari77/paperspace-comfyui/issues/12#issuecomment-6032291897
