# EasyWan22 構成整理

## 概要

EasyWan22 は `Wan2.2 I2V` の最小 workflow ではない。実態は「動画生成ワークベンチ」で、生成本体の前後に大量の制御と後処理が乗っている。

全体は次の3層で見ると分かりやすい。

```text
Setup / Runtime
  ├─ ComfyUI 本体
  ├─ custom nodes
  └─ Python / torch / attention 周辺

Assets / Models
  ├─ diffusion models
  ├─ VAE / text encoders
  ├─ LoRA bundles / preset LoRAs
  ├─ detectors
  └─ upscalers

Workflow
  ├─ workflow-core
  ├─ prompt
  ├─ model
  └─ postprocess
```

関連ファイル:

- 元セットアップ: [EasyWan22/EasyWan22/Setup.bat](/notebooks/EasyWan22/EasyWan22/Setup.bat:1)
- 元ダウンロード定義: [EasyWan22/Download/Default.bat](/notebooks/EasyWan22/Download/Default.bat:1)
- 元 workflow: [EasyWan22/EasyWan22/Workflow/00-I2v_ImageToVideo.json](/notebooks/EasyWan22/EasyWan22/Workflow/00-I2v_ImageToVideo.json:1)
- 現在の downloader: [scripts/download_easywan22.py](/notebooks/scripts/download_easywan22.py:1)
- notebook 起動入口: [start.ipynb](/notebooks/start.ipynb:1)

## 0. Setup と Asset 層

### Setup 層

Windows 側の `Setup.bat` は、workflow を壊さずに動かすための固定セットアップに近い。

- ComfyUI バージョン固定
  - 既定は `v0.3.55`
- ComfyUI-Manager 固定
  - 既定は `3.35`
- PyTorch / Triton / SageAttention の組み合わせ固定
- `transformers`, `PyYAML`, `opencv`, `stringzilla` など補助依存の固定
- custom node 群の clone / update

意味としては「新しい環境へ追従する」より「workflow が壊れないように依存を止める」寄り。

### Asset 層

`Default.bat` が workflow 前提のモデル群を集める入口。カテゴリは次のとおり。

- diffusion models
  - `FastMix`
  - `Base`
- text encoders
  - `Qwen2.5_3B_instruct_bf16`
  - `umt5_xxl_fp8_e4m3fn_scaled`
  - `umt5-xxl-enc-fp8_e4m3fn`
- VAE / VAE approx
- detector
  - `ultralytics/segm/*`
- upscaler
- fast LoRA / Wan21Fast
- bundle LoRA
  - `Nashikone-I2v`
  - `Nashikone-I2vWan21`
- preset LoRA
  - `Nsfw`
  - `NsfwWan21`
  - `Sfw`

### 配布元

元の EasyWan22 は Hugging Face と Civitai を混ぜている。

- public HF
  - Base / FastMix / VAE / text encoder / 一部 LoRA / bundle
- Civitai
  - `Nsfw`, `NsfwWan21`, `Sfw` の多く
  - detector の一部

### この repo での現在の運用

- public HF から取れるものは public HF から取る
- Civitai 由来は private mirror repo `korokoro77/paperspace_models_mirror` に寄せてそこから取る
- `SmoothMIX` は private mirror から外して public HF `Babaladen/SmoothMIX` から取る

配置先:

- `/app/models`
  - 主モデル置き場
- `/storage/ComfyUI/models`
  - detector や Civitai 系 workflow assets

## 1. workflow-core

### workflow-core の役割

workflow-core は「入力画像から Wan2.2 動画を1本出すまで」の最短経路。

```text
Start Image / End Image
  -> 画像サイズと動画長の決定
  -> image embeds 作成
  -> sampler 実行
  -> generated frames
```

### 対応する主な group

- `ImageInput, 画像入力`
- `VideoOutput, 動画出力`
- `Generate, 生成`
- `GeneratedVideo, 生成した動画`
- `SkipPostProcess`

### 主要入力

- 開始画像
- 終了画像
  - 使う構成では補間や start/end 指定に利用
- 秒数
- fps
- 画面サイズ

### 内部でやっていること

1. 開始画像を読む
2. 必要なら終了画像も読む
3. サイズを workflow 内ルールで丸める
   - 長辺、面積、16n など
4. 秒数と base fps からフレーム数を作る
   - `4n + 1` になるように調整
5. 画像から `WANVIDIMAGE_EMBEDS` を作る
6. sampler に投入して動画フレーム列を得る

### core の特徴

- 「最小 I2V」だけならここが本体
- ただし EasyWan22 では、この core に prompt 自動化、LoRA preset、後処理が密結合している

### core で直接効くモデル

- VAE
  - `Wan2_1_VAE_bf16.safetensors`
- text encoders
  - `umt5-xxl-enc-fp8_e4m3fn.safetensors`
  - `Qwen2.5_3B_instruct_bf16.safetensors`
- diffusion models
  - `Wan22-I2V-FastMix_v10-H/L-Q4_K_M.gguf`
  - `Wan2.2-I2V-A14B-HighNoise/LowNoise-Q4_K_M.gguf`

## 2. prompt

### prompt 層の役割

EasyWan22 の prompt は「文字列をそのまま sampler に入れる」だけではない。

```text
PositiveInput
  + TranslateInput
  + Preset Prompt
  + LoRA Trigger
  + Qwen rewrite
  -> final positive

NegativeInput
  -> final negative

final positive / negative
  -> UMT5 encode
```

### 対応する主な group

- `Prompt, プロンプト`
- `Preset, プリセット`

### 入力の種類

- `PositiveInput`
  - 翻訳対象外の trigger や明示 prompt
- `TranslateInput`
  - 日本語入力
- `NegativePromptInput`
- `Preset-Prompt`
  - workflow が preset から差し込む文字列

### 内部でやっていること

1. PositiveInput と TranslateInput を合成
2. preset 由来 prompt を差し込む
3. LoRA 側が返す trigger を差し込む
4. Qwen で書き換えまたは補助
5. negative prompt を別系統で整理
6. 最終 positive / negative を UMT5 で encode

### prompt 層の特徴

- prompt engineering が workflow 内に埋め込まれている
- LoRA 選択が prompt にも効いている
- 生成品質の差が prompt 文字列だけでなく preset 制御や Qwen 由来で変わる

### prompt 関連モデル

- `Qwen2.5_3B_instruct_bf16.safetensors`
- `umt5-xxl-enc-fp8_e4m3fn.safetensors`
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`

## 3. model

### model 層の役割

model 層は「何の diffusion model と LoRA で、どのメモリ戦略で回すか」をまとめて決める。

```text
Model Select
  ├─ FastMix High/Low
  ├─ Base High/Low
  ├─ compile settings
  ├─ block swap
  └─ LoRA chains
        ├─ Fast
        ├─ Wan21Fast
        ├─ Nashikone
        ├─ Nsfw
        ├─ NsfwWan21
        └─ Sfw
```

### 対応する主な group

- `Model, モデル`
- `Preset, プリセット`

### 主な責務

- FastMix と Base の切り替え
- High noise / Low noise の切り替え
- LoRA 多段適用
- torch compile の有無
- block swap による VRAM 制御
- refiner 側モデルの切り替え

### LoRA の見方

LoRA は用途ごとにかなり分かれている。

- `Fast`
  - speed / few-step 強化
- `Wan21Fast`
  - 軽量寄り / motion 補助
- `Nashikone-I2v`
  - preset bundle
- `Nashikone-I2vWan21`
  - preset bundle
- `Nsfw`
  - 行為 / motion / body action preset
- `NsfwWan21`
  - Wan2.1 系補助
- `Sfw`
  - dance / presentation / clean motion

### model 層の特徴

- 生成モデル選択と LoRA preset が UI 上で強く結合している
- prompt 層とも trigger 文字列でつながっている
- 実行時の見た目以上に「生成ロジック」がここへ寄っている

## 4. postprocess

### postprocess 層の役割

EasyWan22 の後処理はかなり広い。単なる save 前の加工ではなく、別の動画処理パイプラインになっている。

```text
generated frames
  -> refiner
  -> detailer
  -> mosaic
  -> trim / fade / append
  -> color adjust
  -> upscale
  -> VFI
  -> label
  -> save webp/mp4/endframe
```

### 対応する主な group

- `Refiner, リファイナ`
- `Detailer, ディティーラ`
- `PostProcess, 後処理`
- `AutoMosaic`
- `PointMosaic`
- `MaskMosaic`
- `MosaicWork, モザイク作業`
- `AppendVideo, 動画の連結`
- `VideoFrameInterpolation (VFI), フレーム補間`
- `AddLabel, ラベル追加`
- `SaveWebp`
- `SaveMp4`
- `SaveEndFrame`

### 各機能の意味

- Refiner
  - 一度出した動画を追加描画して改善
- Detailer
  - 顔や局所の描き直し
- Mosaic
  - detector, point, mask で複数経路
- Append / RepeatFade / Trim
  - 動画編集的な処理
- Color match / correction
  - 色味調整
- Upscale
  - 解像感を上げる
- VFI
  - フレーム補間
- Label
  - 共有用の文字列付与
- Save
  - WebP, MP4, end frame 保存

### postprocess で使う補助モデル

- `2x-AnimeSharpV4_Fast_RCAN_PU.safetensors`
- `AnzhcFace-v3-640-seg.pt`
- `AnzhcBreasts-v1-1024-seg.pt`
- `AnzhcEyes-seg.pt`
- `AnzhcHeadHair-seg.pt`
- `99coins_anime_girl_face_m_seg.pt`
- `ntd11_anime_nsfw_segm_v4_all.pt`
- `nipples_v2_yolov11s-seg.pt`
- `PitEyeDetailer-v2-seg.pt`
- `PitHandDetailer-v2-Test-v9c.pt`

### postprocess 層の特徴

- ここだけで別 workflow を作れるくらい大きい
- EasyWan22 の「使い勝手」はかなりこの層から来ている
- 一方で、壊れやすさもここが大きい

## 5. workflow 全体を図で見る

```text
[ImageInput]
   |
   v
[workflow-core]
   |
   +--> [prompt]
   |       |
   |       v
   +--> [model]
           |
           v
      [Wan2.2 Generate]
           |
           v
      [postprocess]
           |
           +--> SaveGeneratedVideo
           +--> SaveWebp
           +--> SaveMp4
           +--> SaveEndFrame
```

もう少し実務寄りに見るとこうなる。

```text
入力画像
  -> サイズ/秒数/フレーム数
  -> prompt 構築
  -> model / LoRA / compile / block swap
  -> Wan sampler
  -> refiner / detailer
  -> mosaic / color / append / upscale / VFI / label
  -> 保存
```

## 6. custom node 依存

workflow が巨大なのは、実質 custom node の集合で組まれているからでもある。

主要依存:

- `ComfyUI-WanVideoWrapper`
- `rgthree-comfy`
- `comfyui-impact-pack`
- `comfyui-kjnodes`
- `comfyui-videohelpersuite`
- `comfyui_essentials`
- `comfyui-custom-scripts`
- `comfyui-mxtoolkit`
- `pysssss`

見えている node の傾向:

- `GetNode`: 137
- `SetNode`: 92
- `MarkdownNote`: 24
- `SimpleMathInt+`: 19
- `Fast Groups Muter (rgthree)`: 15
- `mxSlider`: 15
- `ImpactSwitch`: 11

つまり EasyWan22 は「大きな workflow」というより、「大量の UI 制御 node と shared state でできたアプリ」に近い。

## 7. この repo での現在の扱い

残している workflow:

- 元 API export: [workflows/easywan22_api.json](/notebooks/workflows/easywan22_api.json:1)
- 修正版 API prompt: [workflows/easywan22_api_fixed.json](/notebooks/workflows/easywan22_api_fixed.json:1)
- 修正版 editor workflow: [workflows/easywan22_editor_fixed.json](/notebooks/workflows/easywan22_editor_fixed.json:1)
- ComfyUI 配置: [easywan22_editor_fixed.json](/storage/ComfyUI/user/default/workflows/easywan22_editor_fixed.json:1)

fixed 版を作った理由:

- 古い node 名
- frontend 依存の missing node
- `SimpleMath+` の旧 input 形式
- editor workflow と API prompt の差
- Windows path

なので fixed 版は「意味が完全に一致する保証」ではなく、まず「ComfyUI 上で回せるところまで補修した版」。

## 8. notebook との関係

[start.ipynb](/notebooks/start.ipynb:1) は現在こういう責務に寄っている。

1. runtime 前提確認
2. HF auth 確認
3. base repo 同期
4. eye lora 同期
5. SmoothMIX 同期
6. EasyWan22 動画 workflow assets の追加 download
7. `extra_model_paths.yaml` 再生成
8. ComfyUI 起動

Notebook 側は workflow 本体を直す場所ではなく、「必要 asset が揃った ComfyUI runtime を作る場所」。

## 9. 現状の判断

EasyWan22 を理解する上で重要なのは次。

1. これは `Wan2.2 I2V` の最小 workflow ではない
2. core, prompt, model, postprocess が強く結合している
3. 実行が通ることと、意図どおりの動画が出ることは別

現状の repo 側では、

- downloader と notebook 導線はかなり整理済み
- private mirror repo も構築済み
- API 実行までの補修はある程度済み
- ただし workflow の意味的な正しさと画質はまだ要検証

という段階にある。

## 10. 次にやるなら

安全な進め方は次。

1. `workflow-core` だけの最小 Wan2.2 I2V を安定させる
2. その後 `prompt` を戻す
3. 次に `model` の preset / LoRA を戻す
4. 最後に `postprocess` を段階的に戻す

一気に EasyWan22 全体を直すより、この順の方が構造的に正しい。
