# drama3-small と Irodori-TTS v4.1：A6000 日本語TTS比較（2026-10-16）

対象：[paperspace-comfyui Issue #13](https://github.com/koshihikari77/paperspace-comfyui/issues/13)。

## 結論（暫定。聴感評価は未実施）

**通常の制作は Irodori-TTS v4.1 を第一候補にする。drama3-small は、聴いて演技が明らかに上回る場面に限って、再生成込みで併用候補とする。**

- 自動指標で判定した失敗率（同一台本24セリフ×2声×3seed、30秒参照）は、**drama3-small が 12.5%（18/144）、Irodori は Small・MF・Anime とも 0%（0/144）**。
- drama3-small の失敗は2つに集中した。
  - **漢字の読み**：東雲・流鏑馬・八幡宮 で 5/6。
  - **途中で台詞を見失う**：短い文や疑問文の後に続く行で、後半が脱落または無音化する。演技切替の行で 8/18。タグを外したプレーン文でも 6/18 発生したので、タグの書き方の問題ではない。
- 速度（単独実行、30秒参照、ウォーム状態）は、**Irodori-Small が 5.5倍速、MF が 17倍速、drama3-small が 1.8倍速**。drama3 のストリーミングは初音まで約1.0秒。
- 約50秒の長文では、どちらも内容の欠落はなく、声質のドリフトも小さかった。8秒窓ごとの話者類似度の最小値は 0.93〜0.97。
- **演技の質（囁き・笑い・感情がそれらしく聞こえるか）は自動指標では測れない。** drama3 の強みとされる点はここなので、`review.html` で聴いて判定する必要がある。上の結論はこの点を含んでいない。

## 環境

| 項目 | 値 |
|---|---|
| GPU / driver | RTX A6000 48GB（SM86）/ 550.144.03（CUDA 12.4） |
| drama3-small | `pentacoxian-dev/drama3-small-GGUF` @ `4a3078e8`、`drama3-small-IQ3_XXS.gguf`（9.74 GiB） |
| audio.cpp | `JarodMica/audio.cpp` @ `9277495a`。どのブランチにも含まれないため SHA 指定で fetch。`drama3-small-audiocpp.patch`（sha256 `cb71f86c…`）を適用し、CUDA 12.4.131 の nvcc で sm_86 ビルド |
| Irodori | `Aratako/Irodori-TTS` @ `89f9d8fb`、専用 uv venv（Python 3.10、torch 2.10.0+cu128、transformers 5.12.1）。fp32、既定40 step（MF は4 step） |
| モデル | v4.1-Small `2b28324d` / v4.1-Small-MF `ccc78f5d` / v4.1-Anime `6b259f5b`、codec Semantic-DACVAE-Japanese-32dim `47376ee2` |
| 採点 | Whisper large-v3-turbo `41f01f3f`（温度フォールバックあり）、WavLM-base-plus-sv `feb593a6` |
| 配置 | モデルはすべて `/app/models/tts`（HF_HOME も同じ場所。セッションごとに再取得）。実行環境（パッチ済み audio.cpp のソースとバイナリ、Irodori の venv）は `/storage/tts-runtime` に置き、再起動後も残す（計約10.5GB）。評価時点では ComfyUI 本体と venv は変更していない（2026-10-17 に ComfyUI へ組み込み。下記） |

再現手順は `scripts/tts_eval/setup.sh` にまとめた。CUDA コンパイラの apt 導入、モデル取得、パッチ照合、ビルド、venv 作成までを含む。イメージには CUDA の runtime しか入っておらず、ビルド時だけ nvcc と cuFFT・cuRAND 等の dev パッケージを apt で追加した。ビルド済みのバイナリが参照するのはイメージ内の `/usr/local/cuda-12.4/lib64` だけなので、2回目以降は `setup.sh` がビルドと venv 作成を飛ばし、モデル取得だけを行う。

## テスト条件

- **台本**：`scripts/tts_eval/script_ja.json` の24セリフ。内訳は日常3、感情5（喜・怒・悲・戸惑い・照れ）、囁き2、笑い・ため息3、セリフ途中の切替3、長文3（約20秒）、漢字・固有名詞・数字・英語混じり・電話番号5。
- **演技指示**：モデルごとに個別に書いた。drama3 は `[tags]`・英語のスタイル指示・`<whisper>` スパン、Irodori は絵文字。caption は使っていない。
- **サンプリング**：drama3 は作者推奨の temperature 0.7 / top_p 0.7 / top_k 30、pause fix（`max_pause_frames=10` と `silence_codes.json`）、`cpu_expert_layers=0`、`context_length=4096`、`max_tokens=645`。Irodori は既定値（CFG text 3 / speaker 5）。
- **参照音声**：2声 × 約10秒・約30秒。両モデルに同じ音声を渡し、Irodori には分割クリップ（推奨形式）、drama3 には連結した wav と書き起こしを渡した。
  - `jsut_f`：JSUT basic5000 の女性1名（4501〜4530番）。**非商用研究用途に限る**。
  - `cv_m`：Common Voice 17 ja（CC0）の男性1名。連続する同属性の行のうち、WavLM 類似度 0.9 超のクリップだけを使った（`prepare_refs.py`）。
- **テイク数**：1セリフ×1モデルにつき seed 0/1/2 の3テイク。全件を記録し、選別はしていない。
- **失敗の自動判定**（`score.py`）は次のどれか。
  - かな CER が 0.25 超
  - 文中の無音が3秒超
  - 1かなあたり0.6秒超（暴走）
  - ピークが −35dBFS 未満（ほぼ無音）
  - 同じセリフの Irodori-Small の長さの中央値の1.8倍超（末尾にゴミ音が付く）
  - タグを音読している
- **注意**：品質用の全テイクは drama3 と Irodori を並列で生成したので、そこに記録された速度は使っていない。速度とVRAM は `bench.sh` で1モデルずつ単独で測り直した。

## 結果：品質（自動指標、全テイク）

| model | refs | 失敗 | 失敗率 | かなCER | 話者類似度 | seed間一致 |
|---|---|---:|---:|---:|---:|---:|
| drama3-small | 30s | 18/144 | 12.5% | 0.092 | 0.931 | 0.933 |
| drama3-small | 10s | 31/144 | 21.5% | 0.152 | 0.883 | 0.877 |
| drama3-small（pause fixなし、seed0） | 30s | 13/48 | 27.1% | 0.198 | 0.881 | – |
| Irodori v4.1-Small | 30s | 0/144 | 0% | 0.019 | 0.962 | 0.978 |
| Irodori v4.1-Small | 10s | 0/144 | 0% | 0.019 | 0.945 | 0.974 |
| Irodori v4.1-Small-MF | 30s | 0/144 | 0% | 0.030 | 0.963 | 0.974 |
| Irodori v4.1-Anime | 30s | 0/144 | 0% | 0.015 | 0.966 | 0.980 |

話者類似度はどの条件でも、その声の30秒参照と比べた値。

**再生成の手間**：drama3 の30秒参照では、1テイク目で採用できる率が 83%。3テイク以内に一度も通らなかったセリフは 48中1（k01 の漢字読み）。10秒参照にすると1テイク目の採用率は 73% に下がり、0/3 のセリフが 4 に増える。Irodori はすべて1テイク目で通った。

**drama3-small の失敗内訳（30s）**：カテゴリ別では、切替行 8/18、漢字 5/6、喜び・悲しみ・照れ・ため息・「笑い→ため息」の行が各 1/6。怒り・戸惑い・笑い始まり・数字・英語混じり・囁き・長文は 0。失敗の典型は次の3つ。

- (a) 十数秒の −40〜−57dBFS のつぶやきだけが出て、別人のような声になる（話者類似度 約0.4、Whisper は「ご視聴ありがとうございました」と幻覚）。無音コードに当たらないため pause fix をすり抜け、長さガードで止まる。
- (b) 最初の文を言ったところで終わる、または後半が無音のまま続く。
- (c) 漢字の誤読。

**pause fix は有効**：外すと失敗率が 12.5% → 27%（seed0 同士の比較）になる。作者報告のとおり。

**参照の長さ**：drama3 は10秒参照で悪化が大きい（失敗 12.5% → 21.5%、類似度 0.93 → 0.88）。Irodori は類似度が少し下がるだけ（0.962 → 0.945）。

**参照テキスト**：audio.cpp では `reference_text` なしで参照音声を渡すとエラーになる（`Drama 3 reference audio requires reference_text`）。drama3 には書き起こしが必須。Irodori は参照テキストを使わない。

## 切替行の切り分け（drama3、30s、2声×3seed）

| プロンプト | 失敗 |
|---|---:|
| 元の書き方（`……[tag]`） | 8/18 |
| タグを文頭に置き、`……` を外す | 8/18 |
| タグなしのプレーン文 | 6/18 |
| Irodori-Small、同じプレーン文 | 0/18 |

→ 原因はタグではなく、「わかった、もういいよ。……って、言うと思った？」のような**短い文の後に続く行**で drama3 が位置を見失うことにある。作者の Limitations にある「途中で位置を見失うと pause fix がテイクを閉じ、残りは飛ぶ」と同じ現象。

## 結果：速度・メモリ（単独実行、`bench.sh`）

対象は d01（短文）、l01（長文約19秒）、n03（タグ付き）、k03（英語混じり）×3seed で、作者の速度測定セットと同じ構成。最初の1件はコールドとして除外。xRT は 音声秒 ÷ 処理秒。

| 構成 | 全体 xRT | 短文 xRT | 長文 xRT | 短文 wall | 長文 wall | GPU ピーク | ロード |
|---|---:|---:|---:|---:|---:|---:|---:|
| drama3-small offline 30s参照 | 1.86 | 1.61 | 2.02 | 3.9 s | 9.6 s | 13.3 GB | 11 s |
| drama3-small offline 10s参照 | 2.11 | 2.06 | 2.17 | – | – | 13.1 GB | 10 s |
| drama3-small streaming 30s参照 | 1.66（初音 **1.01 s**） | 1.48 | 1.78 | 4.2 s | 10.9 s | 11.1 GB | 11 s |
| Irodori-Small fp32 30s参照 | 5.71 | 4.65 | 6.93 | 1.3 s | 2.7 s | 8.2 GB | 13 s |
| Irodori-Small bf16 30s参照 | 6.73 | 4.92 | 10.9 | 1.3 s | 1.9 s | 7.8 GB | 11 s |
| Irodori-Small fp32 10s参照 | 6.81 | 5.57 | 8.20 | 1.2 s | 2.4 s | 8.3 GB | 14 s |
| Irodori-Small-MF（4 step） | 16.5 | 9.1 | 29.7 | 0.8 s | 0.6 s | 8.3 GB | 12 s |
| Irodori-Anime fp32 | 4.67 | 3.04 | 6.63 | 2.0 s | 3.0 s | 8.0 GB | 11 s |

- どちらも、毎リクエストで参照音声をエンコードし直した時間を含む。Irodori の参照エンコードは 0.1〜0.4秒。
- drama3 の値は作者の RTX 4090 の値（全体1.6倍、長文2.0倍）とほぼ同じ。ストリーミングの出力は offline と同一だった（同 seed で CER が一致）。
- GPU ピークは nvidia-smi で測った GPU 全体の使用量。drama3 は長い出力ほど増える（作者の記載では30〜40秒の出力で 13〜15GB）。
- ホスト RAM のピークは、drama3 サーバーが約1.3GB、Irodori プロセスが約5.3GB。

## 長文（約50秒、1段落7文、`longform.py`）

| engine | 方式 | 音声 | wall | かなCER | 類似度 先頭→末尾（最小） |
|---|---|---:|---:|---:|---|
| drama3-small | 1リクエスト（audio.cpp が自動でチャンク分割） | 44〜54 s | 21〜26 s | 0〜0.057 | 0.96→0.95〜0.97（0.93〜0.97） |
| Irodori-Small | 1文ずつ7リクエストを連結 | 51〜52 s | 18〜22 s | 0〜0.02 | 0.95〜0.98→0.95〜0.98（0.95〜0.97） |
| Irodori-Anime | 同上 | 51 s | 17〜23 s | 0 | 0.96〜0.98→0.95〜0.98（0.95〜0.97） |

- drama3 の男声では、文中に1.7〜2.2秒の間が入った。
- Irodori は1回の出力が30秒までなので、長文は文単位に分けることになる。文間の自然さは聴いて確認する必要がある。

## ComfyUI への組み込み（2026-10-17 追記：導入済み）

当初は「別プロセスのサーバーを HTTP で呼ぶ」案だったが、両モデルとも ComfyUI の venv で直接動くノードがあったので、そちらを入れた。手順は `scripts/install_tts_nodes.sh`。

- **drama3-small：作者の [ComfyUI-drama3-small](https://github.com/pentaCoxian/ComfyUI-drama3-small)（commit c7137d2c）。audio.cpp は不要。**
  - 同じ GGUF を純 PyTorch と Triton カーネルで動かす。pause fix と長文の分割も入っている。
  - 追加の依存はない。コードのライセンスは Fish Audio Community License。
  - 公開から1日、commit は1つだけ（導入時点）。
- **Irodori Anime：[comfy-IrodoriTTS](https://github.com/kjranyone/comfy-IrodoriTTS)（kjranyone、commit 60b57245、MIT）。**
  - 作者の Aratako による公式 ComfyUI ノードはない。
  - dacvae の宣言どおりに依存を入れると、CUDA 13 版の torchaudio と protobuf 3.19 が入ってしまう。そこで実際に import されるものだけを `--no-deps` で入れた。
  - torchaudio 2.11 と torchcodec 0.17 は cu126 版を使った。torch の stable ABI を使うので 2.14 でも動く。
  - コーデックは node の `data/` に落ちる作りなので、`/app/models/irodori/` へのリンクにした。
- **ほかの候補**
  - ComfyUI 公式の Fish Audio ノードは、有料のクラウド API（Drama 3 本体）を呼ぶもので、ローカル実行ではない。
  - PedroMarinhoDev/ComfyUI-Drama3 などはフルサイズの Drama 3（24〜38GB）用で、drama3-small には使えない。
  - jupo-ai 版の Irodori ノードは transformers 5 未満を要求するので、ComfyUI の venv とぶつかる。
- **動作確認**（ComfyUI 0.41.0、cv_m 30秒参照、6セリフ × seed 0/1、Whisper で採点）
  - **Irodori Anime**：公式実装と秒数が一致し、話者類似度の差も ±0.003 以内。
  - **drama3-small**：12本とも、評価で使った失敗判定（CER 0.25 以下）を満たした。12本だけなので、評価で見えた失敗率（12.5%）が変わったかどうかは判断できない。
  - 速度（ウォーム状態、1本あたり）：drama3 は 5〜12秒、Irodori は 2〜4秒。
  - 出力は `/storage/ComfyUI/output/tts_check/`、採点は `workspace/tts_comfy_check/scores.json`。

## ライセンス（原文で確認）

**drama3-small：Fish Audio Community License v1.0（2026-10-13版）**

- 研究・非商用は無償。商用は、本人と関連会社の年間総売上が USD 50万未満の間だけ可（III(a)）。
- 生成音声の所有は利用者に帰属する（IV(c)(iii)）。
- 自分で作る作品の台詞やナレーション、受託で完成音声だけを納品する場合は Hosted Service に当たらない（III(c)(1)(2)）。
- 第三者に TTS や声のクローンを提供するサービスは、売上に関係なく別途契約が必要。
- 製品やサイトで配布・利用する場合は「Built with Fish Audio」の表示義務がある（IV(a)）。
- 本人の同意がない声の模倣、実在人物の声での性的コンテンツ、ウォーターマークの除去は禁止（IV(b)）。
- 出力を音声生成モデルの学習・蒸留に使うことも禁止（IV(b)）。
- 売上が閾値を超えたら30日以内に通知し、90日の猶予後は別途契約が必要（III(b)）。

**Irodori v4.1-Small / MF：MIT と倫理的制限**（本人の同意がない声のクローン、ディープフェイクや誤情報の禁止）。

- 出力には SilentCipher のウォーターマークが自動で入る。
- **v4.1-Anime も MIT だが、学習に使った「アニメ調音声データ」の出所が書かれていない。** 商用作品で使う前に確認が必要。

**コード**：audio.cpp は Apache-2.0、Irodori-TTS は MIT。

**今回の参照音声**：JSUT は非商用研究用途のみで再配布不可、Common Voice は CC0。**評価専用で、作品には使わない。** 制作で使う声は、本人の許可があるもの、または自作の声で別途用意する。

## 限界・未実施

- **聴感評価（自然さ、演技の追従、照れや囁きの質）は未実施。** `review.html` で行う。自動指標はタグの音読は検出できるが、演技が「それらしい」かは判定できない。
- かな CER は Whisper と pykakasi の読みに依存する。数字や表記ゆれで差が出る（例：電話番号の行は全モデルで約0.07）。
- 話者は2声で、どちらもアニメ声優ではない。声優風の参照では結果が変わりうる。
- drama3 の失敗判定に、Irodori-Small の長さを基準にした `overlong` を使っている。30s の結果では、`overlong` だけで失敗になったテイクは0件で、いずれも内容の誤り（content）と同時に出ている。
- Irodori の caption 指示と、drama3 の複数話者（`<|speaker:N|>`）や `previous_takes` による部分修正は試していない。

## 成果物（`/notebooks/workspace/tts_eval/`、計約1.8GB。zip：`/notebooks/tts_eval_20261016.zip`、1.3GB。Git には入れていない）

- `takes/<model>/<voice>_<refs>/<line>_s<seed>.wav` と `log_<refs>.jsonl`：全テイクの音声、プロンプト、設定、時間、VRAM
- `scores.csv`（1テイク1行、書き起こし付き）、`summary.md`
- `review.html`：音量を揃えた 48kHz コピー（`review_audio/`）で試聴して比べるページ。JupyterLab で右クリック →「Open in New Browser Tab」で開く
- `bench/`：単独実行の速度測定（同じ形式）
- `longform/`：長文試験と `scores.json`
- `ablation/`：切替行の切り分けと、その `review.html`
- `refs/`：参照音声（**JSUT 由来の音声は再配布しない**）
- `evidence/drama3-small-noreftext/`：参照テキストなしでエラーになった記録

## 再現

```bash
scripts/tts_eval/setup.sh                       # models -> /app/models/tts; build+venv -> /storage/tts-runtime (first time only)
PY=/storage/tts-runtime/Irodori-TTS/.venv/bin/python
export HF_HOME=/app/models/tts/hf
cd scripts/tts_eval
$PY prepare_refs.py                             # refs -> /notebooks/workspace/tts_eval/refs
$PY run_drama3.py --refs 30s; $PY run_drama3.py --refs 10s
$PY run_drama3.py --refs 30s --no-pause-fix --seeds 0
for m in small mf anime; do $PY run_irodori.py --model $m --refs 30s; done
$PY run_irodori.py --model small --refs 10s
$PY longform.py --engine drama3; $PY longform.py --engine irodori-small; $PY longform.py --engine score
./bench.sh                                      # GPU 上で単独実行すること
$PY score.py; TTS_EVAL_DIR=/notebooks/workspace/tts_eval/bench $PY score.py
$PY review_page.py
```
