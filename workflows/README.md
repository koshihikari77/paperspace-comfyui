# Wan 2.2 workflows

`floyo_wanvideowrapper_i2v.json` is the stable WanVideoWrapper API workflow
copied from the `agent-harness` Floyo production reference.

Its required ComfyUI models are defined by the `floyo-wan22-stable` group in
`scripts/download_easywan22.py`. Standard models are downloaded to
`/app/models` for each Paperspace session and exposed through
`/storage/ComfyUI/extra_model_paths.yaml`.

RIFE is the exception: `comfyui-frame-interpolation` manages its checkpoint in
its own directory under `/storage/ComfyUI/custom_nodes` and downloads
`rife47.pth` on first use.
