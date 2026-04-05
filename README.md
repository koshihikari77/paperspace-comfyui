# Paperspace 用 ComfyUI コンテナ

Paperspace の永続ストレージにある `ComfyUI` をそのまま使うための軽量イメージです。  
イメージには `ComfyUI` 本体やその依存を入れず、JupyterLab と Hugging Face CLI の `hf` だけを入れます。

## 方針

- `ComfyUI` 本体は `/storage/ComfyUI` に置く
- モデルはイメージに焼かない
- モデルは Notebook のセルから `hf download` で取得する
- 取得先は `/app/models`
- `ComfyUI` からは `/storage/ComfyUI/extra_model_paths.yaml` で `/app/models` を参照する
- モデルセットは `image`, `video` などのカテゴリで切り替える

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
2. `HF_TOKEN` を Paperspace Secrets か環境変数で渡す
3. 永続ストレージ上に `ComfyUI` を `/storage/ComfyUI` で置く
4. [`hf-models.example.yaml`](/mnt/c/Users/inada/obsidian/base/03_projects/paperspace-comfyui/hf-models.example.yaml) を参考に `/storage/ComfyUI/hf-models.yaml` を作る
5. Notebook 起動後に [`start.ipynb`](/mnt/c/Users/inada/obsidian/base/03_projects/paperspace-comfyui/start.ipynb) を開く
6. 先頭セルの `MODEL_CATEGORY` を `image` や `video` に変更して順番に実行する
7. ComfyUI へのアクセスは `https://tensorboard-$PAPERSPACE_FQDN` を使う

Notebook は次を行います。

- `hf` と `HF_TOKEN` の確認
- manifest の読み込み
- 指定カテゴリのモデルだけ `/app/models` へ同期
- `/storage/ComfyUI/extra_model_paths.yaml` をバックアップして再生成
- `/storage/ComfyUI/main.py` を `6006` で起動
- Paperspace の `tensorboard-$PAPERSPACE_FQDN` 形式の URL を表示

## Manifest

標準パスは `/storage/ComfyUI/hf-models.yaml` です。  
カテゴリごとに private repo のモデル一覧を定義します。

```yaml
version: 1
categories:
  image:
    - repo: your-org/private-models
      source_path: checkpoints/sd_xl_base_1.0.safetensors
      target_subdir: checkpoints
    - repo: your-org/private-models
      source_path: loras/style.safetensors
      target_subdir: loras
  video:
    - repo: your-org/private-models
      source_path: diffusion_models/video-model.safetensors
      target_subdir: diffusion_models
```

使える `target_subdir`:

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

追加キー:

- `filename`: 保存名を変えたいときに指定
- `revision`: 省略時は `main`

## Notebook の設定値

Notebook の先頭セルで次を変更できます。

- `MODEL_CATEGORY`
- `FORCE_DOWNLOAD`
- `COMFYUI_PORT`
- `COMFYUI_ARGS`
- `COMFYUI_PYTHON`

`COMFYUI_PYTHON` を空のままにした場合は、次の順で探索します。

1. `/storage/ComfyUI/.venv/bin/python`
2. `/storage/ComfyUI/venv/bin/python`
3. コンテナ内の `python`

永続ストレージ側に依存入りの仮想環境があるなら、それを使うのが前提です。

Paperspace Notebook では `6006` を `tensorboard-$PAPERSPACE_FQDN` で公開する前提が扱いやすいので、Notebook の既定ポートも `6006` にしています。

## 備考

- `/app/models` はコンテナローカルなので、Notebook セッションごとに必要なモデルを再同期します
- `extra_model_paths.yaml` が既にある場合は `extra_model_paths.yaml.bak.paperspace-comfyui` に退避してから上書きします
- 旧 README にあった GCS 前提の運用はこの構成では使いません
