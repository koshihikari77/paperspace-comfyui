# Wan2.2 / EasyWan22 Refiner Notes

## 結論

- 今の `wan_22_easywan_povmissionary_base_automosaic_refiner.json` に入っているものは、
  EasyWan22 の `RefinerSampler` を移植したものではない。
- 実体は `wan_22` 系 native workflow の中にある
  `decode -> color match -> re-encode -> KSamplerAdvanced -> decode`
  という二段目再サンプル枝。
- したがって、これは「native の second pass」であって、EasyWan22 の wrapper refiner とは別物。

## 今の workflow 側で起きていること

対象: `wan_22_easywan_povmissionary_base_automosaic_refiner.json`

本生成:
1. `KSamplerAdvanced` id `20` で high-noise 側を前半ステップ実行
2. `KSamplerAdvanced` id `21` で low-noise 側を後半ステップ実行
3. `VAEDecodeTiled` id `22` で画像へ戻す

二段目枝:
1. `ColorMatch` id `35`
2. `VAEEncode` id `40`
3. `KSamplerAdvanced` id `32`
4. `VAEDecodeTiled` id `33`
5. `ImageUpscaleWithModel` id `28`
6. `RIFE VFI` id `30`

このため順序は `refine-ish second pass -> upscale -> VFI` にはなっている。

## EasyWan22 の本来の Refiner

EasyWan22 の refiner は別系統の subgraph 2つで構成される。

- `RefinerImageEmbeds`
- `RefinerSampler`

### RefinerImageEmbeds の入力
- `vae: WANVAE`
- `image: IMAGE`
- `upscale_model: UPSCALE_MODEL`
- `image_1: IMAGE`
- `image_2: IMAGE`
- 各種 boolean / int

### RefinerSampler の入力
- `image: IMAGE`
- `model: WANVIDEOMODEL`
- `vae: WANVAE`
- `samples: LATENT`
- `image_embeds: WANVIDIMAGE_EMBEDS`
- `text_embeds: WANVIDEOTEXTEMBEDS`
- `steps`
- `total steps`
- `shift`
- `seed`
- `additional seed`
- end-image や postprocess 用の boolean

### 重要点
- EasyWan22 refiner は wrapper 系型
  - `WANVIDEOMODEL`
  - `WANVAE`
  - `WANVIDIMAGE_EMBEDS`
  - `WANVIDEOTEXTEMBEDS`
- `wan_22` 側は native 系型
  - `MODEL`
  - `VAE`
  - `CONDITIONING`
  - `LATENT`

この型の差があるので、EasyWan22 の refiner をそのまま `wan_22` に刺すことはできない。

## 何が必要か

EasyWan22 に近い refiner をやるなら、native 側で以下を設計し直す必要がある。

- 二段目 sampler の条件付けをどうするか
- 画像再投入前に start/end image や clip vision 的な情報をどう使うか
- seed / sigma / shift / total steps を本生成とどう分けるか
- color match を refiner 前に置くか後に置くか

## 現時点の判断

- 今の `..._refiner.json` は「動く可能性のある二段目仕上げ版」としては使える
- ただし EasyWan22 の refiner 移植版と呼ぶのは不正確
- 次にやるなら、native second pass をちゃんと設計して別 workflow にするのが正しい
