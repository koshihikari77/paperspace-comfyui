# Seed-VC / RVC voice conversion（2026-10-17、A6000）

対象：[Issue #14（Seed-VC）](https://github.com/koshihikari77/paperspace-comfyui/issues/14)、[Issue #15（RVC）](https://github.com/koshihikari77/paperspace-comfyui/issues/15)。

## 結論（暫定。聴感評価は未実施）

- **Seed-VC は普通のセリフ・笑い・声の出るため息なら変換できる。息切れや囁きのような無声の音は、ほぼ変換されない。**
  - セリフ：変換先との話者類似度 0.97〜0.98、かな CER 0、元音声とのピッチ相関 0.8〜0.99。
  - 笑い・ため息：類似度 0.90〜0.96。
  - 息切れ・囁き：類似度 0.43〜0.51 で、変換前（0.35〜0.54）とほとんど変わらない。4kHz 以上のエネルギー（息成分の目安）は元の 14〜38% に減る。
  - 「非言語音は素材から変換する」用途、特に吐息系には、そのままでは向かない。
- ComfyUI から使える（下記）。1本 1.5〜3秒、VRAM 約3GB。
- RVC は Applio の専用環境で学習まで動いた。変換と比較は未実施（学習が 277/300 epoch で中断）。

## テスト条件

- **声はすべて合成音声**。drama3-small に参照音声を渡さず、seed と英語の声の説明で作った架空の声を使った。実在の人物の声は使っていない。
  - 変換先：`tgt_f`（明るい若い女性）と `tgt_m`（落ち着いた男性）。
  - 変換元：別の合成話者（柔らかい女性）。セリフ2本（日常・喜び）と、非言語4本（ため息・笑い・息切れ・息を呑む＋囁き）。
- **条件**：`f0_condition` オン（44.1kHz モデル）で steps 25 と 50、オフ（22kHz 会話モデル）で steps 25。`inference_cfg_rate` 0.7、`auto_f0_adjust` オン。
- **テイク**：各2回。ノードに seed の入力がないので、テイクは単に回し直したもの。ComfyUI のキャッシュを避けるため、テイクごとに入力ファイルを分けた。
- **指標**（`scripts/vc_eval/score_vc.py`）：WavLM-SV 話者類似度（変換先・変換元）、F0 相関、長さ比、4kHz 以上のエネルギー比、セリフのかな CER（Whisper large-v3-turbo）。

## 結果（Seed-VC、72本）

| 条件 | 素材 | 本数 | 変換先との類似度 | 変換元との類似度 | 変換前の類似度 | F0 相関 | 高域比 | かな CER | 秒/本 |
|---|---|---|---|---|---|---|---|---|---|
| f0 オン・25 steps | セリフ | 8 | 0.973 | 0.634 | 0.631 | 0.905 | 0.80 | 0 | 2.0 |
| f0 オン・25 steps | 非言語 | 16 | 0.697 | 0.715 | 0.519 | 0.481 | 0.53 | - | 2.0 |
| f0 オン・50 steps | セリフ | 8 | 0.975 | 0.634 | 0.631 | 0.886 | 0.83 | 0 | 2.7 |
| f0 オン・50 steps | 非言語 | 16 | 0.702 | 0.733 | 0.519 | 0.473 | 0.50 | - | 2.9 |
| f0 オフ・25 steps | セリフ | 8 | 0.978 | 0.641 | 0.631 | 0.797 | 1.08 | 0 | 1.5 |
| f0 オフ・25 steps | 非言語 | 16 | 0.716 | 0.752 | 0.519 | 0.309 | 0.57 | - | 1.5 |

非言語の内訳（f0 オン・25 steps）：

| 素材 | 変換先との類似度 | 変換元との類似度 | 変換前 | 高域比 |
|---|---|---|---|---|
| 笑い | 0.955 | 0.600 | 0.573 | 1.09 |
| ため息 | 0.895 | 0.744 | 0.620 | 0.50 |
| 息切れ | 0.430 | 0.823 | 0.537 | 0.38 |
| 息を呑む＋囁き | 0.510 | 0.693 | 0.347 | 0.14 |

- WavLM の類似度は、短い息だけの音では当てにならない可能性がある。一方、高域比が下がっているのは測定上はっきりしているので、息が消えている点は確か。聴いて確認すること。
- steps を 25 から 50 に上げても差はほとんどない。f0 オンのほうがピッチの流れがよく残る。
- 単独で測った性能：モデル読み込み 16秒、VRAM 2.7GB 常駐でピーク 3.0GB、7秒の音声で 1.3〜4.3秒。

## ComfyUI への導入

- ノード：[billwuhao/ComfyUI_Seed-VC](https://github.com/billwuhao/ComfyUI_Seed-VC)（commit 02c0cb8b、Apache-2.0）。本家 [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) はアーカイブ済み。
- 導入は `scripts/install_tts_nodes.sh`、モデルは `scripts/download_tts.py --group seed-vc`（3.38GB）。`start.ipynb` の `TTS_GROUPS` の既定に入っている。
- 追加した依存は librosa 1.0.0・munch・docstring-parser だけ。requirements.txt にある funasr・hydra・sounddevice 等は import されていないので入れていない。torch と comfy-kitchen は変更なし。
- モデルは `ComfyUI/models/TTS/` 直下から読む作りなので、そこを `/app/models/TTS` へのリンクにした。
- 同梱の BigVGAN は huggingface_hub 1.x で読み込めない（`proxies` / `resume_download` 引数）。`patches/comfyui-seedvc-hfhub1.patch` で引数を省略可能にし、導入時に自動で当てる。
- ワークフロー：
  - 画面用 `workflows/seedvc_convert.json`。サイドバーのワークフロー一覧にも登録している。
  - API 用 `workflows/seedvc_convert_api.json`。

## RVC（#15、途中）

- **専用環境にした理由**：Applio 3.6.6 は torch 2.11.0・transformers 5.13.1・numpy 2.4.6 などを固定していて、ComfyUI の venv（torch 2.14、numpy 2.5.2）に入れると環境が壊れる。issue でも分離を求めている。
- **導入**：`scripts/install_rvc.sh`。`/storage/tts-runtime/applio` に専用 venv（Python 3.12、torch 2.11.0+cu126）を作る。事前学習（v2 48k・RMVPE・FCPE・ContentVec）は `download_tts.py --group rvc` で `/app/models/rvc/` に入る（0.82GB）。
- **学習データ**：合成話者 tgt_f の声を固定するため、tgt_f の音声を参照にして drama3-small で作った（`scripts/vc_eval/make_rvc_dataset.py`）。
  - 108本・約7.5分。セリフ 88本、非言語 20本。
  - tgt_f の音声との類似度は平均 0.951。0.85 未満は短い非言語の7本だけ。
- **学習**：v2・48k・RMVPE・batch 8。1 epoch 約7〜15秒で、GPU 使用量は学習前より最大約7GB 増えた。
  - 100 / 200 epoch の重みは `applio/logs/vc_eval_tgt_f/` に保存済み。
  - セッション終了で 277 epoch で止まったため、300 epoch の重みと index はまだない。
- **ComfyUI から使う場合**：Applio の `core.py infer` を別プロセスで呼ぶ薄いノードにするのが安全（未実装）。
- **残作業**：index を作る → `scripts/vc_eval/run_rvc.py` で #14 と同じ素材を変換 → `score_vc.py` で採点 → Seed-VC と比べる。

## 成果物

- 入力・出力・採点結果：`/notebooks/workspace/vc_eval/`（`inputs/`、`log.jsonl`、`scores.csv`、`summary.md`、`review.html`）。音声は Git に入れていない。
- 変換結果：`/storage/ComfyUI/output/vc_eval/`。
- スクリプト：`scripts/vc_eval/`（生成と変換は `run_seedvc.py`、採点は `score_vc.py`、比較ページは `review_vc.py`、RVC 用は `make_rvc_dataset.py` と `run_rvc.py`）。

## 再現手順

```bash
bash scripts/install_tts_nodes.sh
python scripts/download_tts.py --group drama3-small --group seed-vc
# ComfyUI 起動後
cd scripts/vc_eval
python run_seedvc.py prepare && python run_seedvc.py convert
/storage/tts-runtime/Irodori-TTS/.venv/bin/python score_vc.py
/storage/tts-runtime/Irodori-TTS/.venv/bin/python review_vc.py
```
