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
7. 先頭セルのダウンロード設定を用途に合わせて変更し、2セルを順番に実行する
8. ComfyUI へのアクセスは `https://tensorboard-$PAPERSPACE_FQDN` を使う

Notebook は設定とURL表示だけを持ち、実処理は `scripts/bootstrap_comfyui.py` が行います。

- `hf` と Hugging Face ログイン状態の確認
- repo 設定の読み込み
- 画像用 LoRA と、Floyo 安定版 Wan 2.2 workflow に必要なモデルだけを `/app/models` へ同期
- `/storage/ComfyUI/extra_model_paths.yaml` をバックアップして再生成
- `/storage/ComfyUI/main.py` を `6006` でバックグラウンド起動（起動済みなら再利用）
- Paperspace の `tensorboard-$PAPERSPACE_FQDN` 形式の URL を表示

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

動画の通常起動は `scripts/download_easywan22.py` の `floyo-wan22-core` と、選択した
High/Low motion LoRAだけを取得します。
SmoothMIX、GGUF、旧 EasyWan22 一式、Clip Vision、別系統の LightX2V は通常起動では取得しません。

取得するのは Wan 2.2 I2V High/Low FP8 Scaled、BF16 UMT5、WanVideoWrapper VAE、
rank64 LightX2V、RealESRGAN x2と、`WAN22_LORA_PRESETS` で選んだHigh/Low LoRAだけです。
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
- `WAN22_LORA_PRESETS`: `-H` / `-L` より前の名前を列挙。空リストならmotion LoRAなし
- `WAN22_DOWNLOAD_MAX_WORKERS`
- `FORCE_DOWNLOAD`
- `COMFYUI_PORT`
- `COMFYUI_ARGS`
- `COMFYUI_PYTHON`
- `HF_HOME`

例えば `WAN22_LORA_PRESETS = ["DeepthroatBlowjob_v10"]` は、private mirrorの
`DeepthroatBlowjob_v10-H.safetensors` と `DeepthroatBlowjob_v10-L.safetensors` だけを取得します。
複数指定した場合は全ペアを取得し、active workflowにはリスト先頭のペアを設定します。
この指定はHigh/Lowが揃っているプリセット用です。

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
- `DOWNLOAD_IMAGE_LORAS=False` ならprivate mirrorの `loras/` 全体は同期せず、Eye LoRAと選択したWan LoRAだけを取得します
- `extra_model_paths.yaml` が既にある場合は `extra_model_paths.yaml.bak.paperspace-comfyui` に退避してから上書きします
- 起動結果は `COMFYUI_STATUS`、`COMFYUI_URL`、`COMFYUI_WORKFLOW`、`COMFYUI_LOG` の固定形式で出力します
- 旧 README にあった GCS 前提の運用はこの構成では使いません
