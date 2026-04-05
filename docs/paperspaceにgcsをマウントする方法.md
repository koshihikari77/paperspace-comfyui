以下、**Paperspace Gradient Notebook に GCP（GCS）をマウントして `/datasets/gcp` で使えるようにする手順（成功したやつだけ）**まとめ。

---

## 1) GCS 側：S3互換用のアクセスキー（HMAC）を作る

1. Google Cloud Console → **Cloud Storage** → **設定** → **相互運用性**
2. **HMAC キー**で「キーを作成」
3. 対象（サービスアカウント推奨）を選んで作成
4. **アクセスキーID** と **シークレット** を控える（シークレットは基本あとで見れない）
---

## 2) Paperspace 側：Secrets に登録

1. Paperspace（Gradient）コンソール → **Secrets**
2. 下記2つを登録

   * `GCP_S3_ACCESS_KEY`（アクセスキーID）
   * `GCP_S3_SECRET_KEY`（シークレット）

---

## 3) Paperspace 側：Data Source（S3互換）として GCS を追加してマウント

1. Notebook の画面で **Data Sources**（または Datasets/Storage 周りの設定）へ
2. **S3 Compatible** を選ぶ
3. 設定（目安）

   * **Bucket**: `comfyui-models-neon-fiber-397403`
   * **Endpoint**: `https://storage.googleapis.com`
   * **Access Key / Secret**: さっきの Secrets を指定
   * マウント先：`/datasets/gcp`（フォルダ名を `gcp` に）
4. **Mount** 実行

確認：

```bash
ls -la /datasets/gcp | head
```

---

## 4) ComfyUI にモデルパスとして追加（extra_model_paths.yaml）

`/datasets/gcp` 直下が `checkpoints/`, `loras/`, `vae/`… になってる前提で例：

```yaml
gcp_models:
  base_path: /datasets/gcp
  checkpoints: checkpoints/
  loras: loras/
  vae: vae/
  clip: clip/
  clip_vision: clip_vision/
  controlnet: controlnet/
  embeddings: embeddings/
  upscale_models: upscale_models/
  text_encoders: text_encoders/
  ipadapter: ipadapter/
  diffusion_models: diffusion_models/
  unet: unet/
```

（設置場所はあなたの環境だと `/storage/ComfyUI/extra_model_paths.yaml`）

ComfyUI 再起動で反映。

---

これで「Paperspace側のローカル容量をほぼ増やさずに、GCS上のモデルを `/datasets/gcp` 経由で ComfyUI から参照」できる運用になります。
