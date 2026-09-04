# Wan 2.2 workflows

`floyo_wanvideowrapper_i2v.json` is the stable WanVideoWrapper API workflow
copied from the `agent-harness` Floyo production reference.

Its required core models are defined by the `floyo-wan22-core` group in
`scripts/download_easywan22.py`. The notebook adds only the High/Low motion
LoRA pairs named in `WAN22_LORA_PRESETS`. Standard models are downloaded to
`/app/models` for each Paperspace session and exposed through
`/storage/ComfyUI/extra_model_paths.yaml`.

The first selected LoRA pair is also written into the ephemeral workflow
`/app/workflows/floyo_wanvideowrapper_i2v_active.json`; this checked-in
reference remains unchanged.

RIFE is the exception: `comfyui-frame-interpolation` manages its checkpoint in
its own directory under `/storage/ComfyUI/custom_nodes` and downloads
`rife47.pth` on first use.
