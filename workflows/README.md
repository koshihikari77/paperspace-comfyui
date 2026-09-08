# Wan 2.2 workflows

`floyo_wanvideowrapper_i2v.json` is the stable WanVideoWrapper API workflow
copied from the `agent-harness` Floyo production reference.

Its required core models are defined by the `floyo-wan22-core` group in
`scripts/download_easywan22.py`. By default, the notebook also downloads the
complete private-mirror `loras/Nsfw` bundle; `WAN22_LORA_PRESETS` selects the
High/Low pair used by the active workflow. Standard models are downloaded to
`/app/models` for each Paperspace session and exposed through
`/storage/ComfyUI/extra_model_paths.yaml`.

The first selected LoRA pair is also written into the ephemeral workflow
`/app/workflows/floyo_wanvideowrapper_i2v_active.json`; this checked-in
reference remains unchanged.

BubbleButt K3NK v1.0 is mirrored as `BubbleButt_v10-H.safetensors` and
`BubbleButt_v10-L.safetensors` under `loras/Nsfw`. The default full-bundle
download includes both. To fetch only this pair:

```sh
python scripts/download_easywan22.py --group wan22-bubblebutt
```

The existing `--wan22-lora-preset BubbleButt_v10` option also supports this pair.
Adding the assets does not change the selected workflow preset.
See [source metadata and hashes](../docs/bubblebutt-model.md).

RIFE is the exception: `comfyui-frame-interpolation` manages its checkpoint in
its own directory under `/storage/ComfyUI/custom_nodes` and downloads
`rife47.pth` on first use.
