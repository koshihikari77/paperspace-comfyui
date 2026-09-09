# Paperspace 用 ComfyUI コンテナ

Paperspace の永続ストレージにある `ComfyUI` をそのまま使うための軽量イメージです。  
イメージには `ComfyUI` 本体やその依存を入れず、JupyterLab と Hugging Face CLI の `hf` だけを入れます。

## 方針

- `ComfyUI` 本体は `/storage/ComfyUI` に置く
- モデルはイメージに焼かない
- モデル取得のロジックは GitHub 側の `scripts/` に置く
- Notebook は GitHub から clone した repo 内の `start.ipynb` を使う
- 取得先は `/app/models`
- `ComfyUI` からは `/storage/ComfyUI/extra_model_paths.yaml` で `/app/models` を参照する
- Hugging Face repo のトップレベル構成をそのまま参照する
- `image` では `loras` のみ取得する
- Wan 2.2 動画は Floyo 本番 workflow に必要なモデルだけを取得する

## 含まれるもの

- Ubuntu 22.04
- CUDA 12.1 runtime
- Python 3.10
- JupyterLab
- Hugging Face CLI `hf`
- PyYAML

## 含まれないもの

- `ComfyUI` 本体
- `ComfyUI` の Python 依存
- モデルファイル
- GCS 関連セットアップ

## ビルド

`podman` 前提です。

```bash
podman build -f docker/Dockerfile -t yourname/comfyui-paperspace:latest .
```

ラッパースクリプトを使う場合:

```bash
chmod +x docker/build-and-push.sh
./docker/build-and-push.sh yourname/comfyui-paperspace:latest
```

プッシュまで行う場合:

```bash
./docker/build-and-push.sh yourname/comfyui-paperspace:latest --push
```

## Paperspace での使い方

1. Paperspace Notebook のカスタムコンテナにこのイメージを指定する
2. Notebook 上でこのリポジトリを clone する
3. `HF_HOME=/storage/.cache/huggingface` を使って一度だけ `hf auth login` する
4. 永続ストレージ上に `ComfyUI` を `/storage/ComfyUI` で置く
5. clone した repo 内の [`hf-repo.yaml`](/mnt/c/Users/inada/obsidian/base/03_projects/paperspace-comfyui/hf-repo.yaml) を必要なら編集する
6. clone した repo 内の [`start.ipynb`](/mnt/c/Users/inada/obsidian/base/03_projects/paperspace-comfyui/start.ipynb) を開く
7. 先頭セルのダウンロード設定を用途に合わせて変更し、3セルを順番に実行する
8. ComfyUI へのアクセスは `https://tensorboard-$PAPERSPACE_FQDN` を使う

Notebook は「設定とURL」「ダウンロード／準備」「ComfyUI起動」の3セルです。
実処理は `scripts/bootstrap_comfyui.py` が `prepare` / `start` のフェーズ別に行います。

- `hf` と Hugging Face ログイン状態の確認
- repo 設定の読み込み
- 画像用 LoRA と、Floyo 安定版 Wan 2.2 workflow に必要なモデルだけを `/app/models` へ同期
- `/storage/ComfyUI/extra_model_paths.yaml` をバックアップして再生成
- `/storage/ComfyUI/main.py` を `6006` でバックグラウンド起動（起動済みなら再利用）
- Paperspace の `tensorboard-$PAPERSPACE_FQDN` 形式の URL を表示

準備セルの表示はフラグ単位のライブ進捗バー、`running` / `complete` / `skipped` / `failed`、
および項目ごとの経過・所要時間に限定します。割合を取得できない処理では、誤った％の代わりに
往復するインジケーターを表示します。
Hugging Faceなどの詳細な進捗は `/storage/ComfyUI/user/logs/bootstrap.log` に保存されます。
そのため、どのダウンロードまで完了したかはNotebook上ですぐ分かり、長い転送ログは通常表示されません。

### ControlNet AnyTest v4

準備フェーズは公開HFの [AnyTest v4](https://huggingface.co/2vXpSwA7/iroiro-lora/blob/main/test_controlnet2/CN-anytest_v4-marged.safetensors)
を毎回確認し、未配置なら `/app/models/controlnet/SDXL/CN-anytest_v4-marged.safetensors` に取得します。
既存ファイルは再ダウンロードしません（`--force` 指定時を除く）。
Notebookの追加フラグは不要で、進捗は `DOWNLOAD_CONTROLNET_ANYTEST_V4` に表示されます。
公開モデルのため、この取得自体にprivate HFミラーの認証は不要です。

既存の `extra_model_paths.yaml` のcontrolnet設定を使うため、ControlNetLoaderでの名前は
`SDXL/CN-anytest_v4-marged.safetensors` です。jobの変更は不要です。
`/app` が初期化されても、次の準備フェーズで復元されます。

単独取得: `python scripts/download_easywan22.py --group controlnet-anytest-v4`

配布サイズ: 2,502,139,104 bytes。
SHA-256: `807aa29189c10660dff77a5bbfcf5cf39d60f7780199db36db36a9096e11ace7`。

## クラッシュ調査ログ

bootstrapのprepare/startで独立した監視プロセスが起動し、
`/notebooks/logs/runtime/` に約10秒間隔で記録します。
ComfyUIが終了しても監視は継続します。同じログディレクトリでは二重起動しません。
Notebookの変更やComfyUIの再起動は不要です。

- `metrics.jsonl`: コンテナRAM使用量・上限・OOMカウンター・メモリpressure、
  ホストmeminfo、主要プロセスのRSS/PID/OOMスコア、GPU使用率・VRAM・温度・電力、
  GPUプロセス、ディスク空き容量、ComfyUI疎通とキュー件数
- `events.jsonl`: 監視開始（boot IDとPID）、OOMカウンター／疎通の変化、
  カーネルログの末尾（60秒ごと、取得可能な場合）、収集エラー
- `comfyui.jsonl`: `/storage/ComfyUI/user/logs/comfyui.log` と `bootstrap.log` の追記分。
  監視開始時は各ファイルの末尾最大1MiBも取り込みます。

時刻はUTC。各ログは10MiB×最大5世代（`.1`～`.4`が過去分）、合計約150MiBに制限。
毎回flush/fsyncし、再起動後も追記します。保存世代を超えた古いログは自動削除されます。
Git対象外です。監視は環境変数・コマンドライン・画像・動画・キューのpromptを保存しませんが、
ComfyUI自体のログにpromptやパス等が含まれることがあるので、共有前に確認してください。

手動起動: `python scripts/collect_runtime_logs.py --start`

`koshi-custom-nodes` の `runtime_diagnostics` 拡張を有効にすると、追加の
`nodes.jsonl` にWan sampler/decode・RealESRGAN・ColorMatch・RIFEの
ノード開始/終了と実行中約1秒ごとのRAM/VRAMを記録します。
prompt IDとnode IDで照合可能です。画像内容やprompt本文は記録しません。
このファイルも最大約50MiBでローテーション（既存ログとの合計約200MiB）。
初回の拡張読み込みにはComfyUIの再起動が必要です。

`memory.current`はページキャッシュも含むため、上限近くでも即OOMとは限りません。
`memory.events`の`oom_kill`増加、プロセス消失、直前の例外を時刻で照合します。
CUDA OOMとホストRAM OOMは別です。監視プロセス自体が停止した場合やVMごと落ちた場合、
最後の約10秒以上が残らない可能性があります。コンテナから`dmesg`が読めない場合は
権限エラーを記録します。VM停止理由やホストGPUドライバ障害の確定には
Paperspace側イベントログが必要な場合があります。

## Repo Config

標準パスは clone した repo 内の [`hf-repo.yaml`](/mnt/c/Users/inada/obsidian/base/03_projects/paperspace-comfyui/hf-repo.yaml) です。  
既定では `korokoro77/paperspace_models` を参照します。別 repo を使うならこのファイルだけ直します。

```yaml
version: 1
repo: korokoro77/paperspace_models
revision: main
```

repo のトップレベルで認識するディレクトリ:

- `checkpoints`
- `loras`
- `vae`
- `clip`
- `clip_vision`
- `controlnet`
- `embeddings`
- `upscale_models`
- `text_encoders`
- `ipadapter`
- `diffusion_models`
- `unet`

取得ルール:

- `image`: `loras` 配下だけ取得
- `video`: `loras` 以外の対応ディレクトリを取得
- `revision`: 省略時は `main`

## Wan 2.2 安定版

動画の通常起動は `scripts/download_easywan22.py` の `floyo-wan22-core` と、private mirrorの
`loras/Nsfw/` 一式を取得します。
SmoothMIX、GGUF、旧 EasyWan22 一式、Clip Vision、別系統の LightX2V は通常起動では取得しません。

取得するのは Wan 2.2 I2V High/Low FP8 Scaled、BF16 UMT5、WanVideoWrapper VAE、
rank64 LightX2V、RealESRGAN x2と、Wan 2.2 NSFW LoRA一式です。
配置先はすべて `/app/models` で、`extra_model_paths.yaml` から参照します。

workflow は `workflows/floyo_wanvideowrapper_i2v.json` です。
起動時には先頭のLoRAプリセットを反映したコピーを
`/app/workflows/floyo_wanvideowrapper_i2v_active.json` に生成します。
RIFEだけはカスタムノードの仕様により `/storage/ComfyUI/custom_nodes/comfyui-frame-interpolation/ckpts/rife`
で管理され、`rife47.pth` が初回使用時に自動取得されます。

Floyo workflow が指定する SageAttention は永続 venv に固定します。

```bash
/storage/ComfyUI/.venv/bin/python -m pip install -r /notebooks/comfyui-requirements.txt
```

## Notebook の設定値

Notebook の先頭セルで次を変更できます。

- `DOWNLOAD_IMAGE_LORAS`: private mirrorの画像LoRA一式（既定は `False`）
- `DOWNLOAD_EYE_LORAS`: Eye LoRA一式を独立して選択
- `DOWNLOAD_WAN22_MODELS`
- `DOWNLOAD_WAN22_NSFW_LORAS`: private mirrorの `loras/Nsfw/` 一式（既定は `True`）
- `WAN22_LORA_PRESETS`: active workflowで使うHigh/Lowペア。リスト先頭を使用
- `WAN22_DOWNLOAD_MAX_WORKERS`
- `FORCE_DOWNLOAD`
- `COMFYUI_PORT`
- `COMFYUI_ARGS`
- `COMFYUI_PYTHON`
- `HF_HOME`

`DOWNLOAD_WAN22_NSFW_LORAS=True` の場合、`WAN22_LORA_PRESETS` はダウンロード対象を絞らず、
active workflowで使用するペアだけを選びます。全体取得を無効にした場合は、選択したHigh/Lowペアだけを取得します。

初回だけ次を実行して Hugging Face のログイン情報を永続化します。

```bash
export HF_HOME=/storage/.cache/huggingface
hf auth login
```

`COMFYUI_PYTHON` を空のままにした場合は、次の順で探索します。

1. `/storage/ComfyUI/.venv/bin/python`
2. `/storage/ComfyUI/venv/bin/python`
3. コンテナ内の `python`

永続ストレージ側に依存入りの仮想環境があるなら、それを使うのが前提です。

Paperspace Notebook では `6006` を `tensorboard-$PAPERSPACE_FQDN` で公開する前提が扱いやすいので、Notebook の既定ポートも `6006` にしています。

## Scripts

Notebook から呼ぶスクリプトは、このリポジトリの `scripts/` にあります。

- `check_runtime.py`
- `check_hf_auth.py`
- `sync_hf_repo.py`
- `write_extra_model_paths.py`
- `run_comfyui.py`
- `bootstrap_comfyui.py`

## 備考

- `/app/models` はコンテナローカルなので、Notebook セッションごとに必要なモデルを再同期します
- `DOWNLOAD_IMAGE_LORAS=False` でも、`DOWNLOAD_WAN22_NSFW_LORAS=True` ならWan 2.2 NSFW LoRA一式は取得します
- `extra_model_paths.yaml` が既にある場合は `extra_model_paths.yaml.bak.paperspace-comfyui` に退避してから上書きします
- 準備結果は `DOWNLOAD_IMAGE_LORAS`、`DOWNLOAD_EYE_LORAS`、`DOWNLOAD_WAN22_MODELS`、`DOWNLOAD_WAN22_NSFW_LORAS`、`WAN22_LORA_PRESETS` などの固定形式で出力します
- 起動結果は `COMFYUI_PROCESS`、`COMFYUI_STATUS`、`COMFYUI_URL`、`COMFYUI_WORKFLOW`、`COMFYUI_LOG` の固定形式で出力します
- 初回起動はLoRAの索引作成などで時間がかかるため最大300秒待機し、それを超えてもプロセスが生きていれば例外にせず `COMFYUI_STATUS=starting` を返します
- 旧 README にあった GCS 前提の運用はこの構成では使いません
