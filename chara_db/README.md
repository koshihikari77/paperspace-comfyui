# chara_db

Nijisanji / Hololive のキャラ LoRA 台帳（Illustrious のみ、1 キャラ 1 本）。
実体は HF private mirror（`hf-repo.yaml` の repo）の `loras/nijisanji/` と `loras/hololive/` に置き、
`start.ipynb` の `CHARA_LORAS` で指定したキャラだけを `/app/models/loras/<franchise>/` に取得する。
`DOWNLOAD_IMAGE_LORAS` の一括同期からはこの 2 フォルダを除外している。

## ファイル

| ファイル | 内容 |
|---|---|
| `loras.yaml` | 選定結果。キャラ id → ファイル、HF パス、Civitai の出典、トリガーワード、別候補（`alternates`） |
| `catalog/<franchise>.csv` | 対象キャラの Illustrious LoRA 候補すべて（ダウンロード数・公開日・URL） |
| `rosters/<franchise>.json` | 英語版 Wikipedia から作った名簿。所属・現役/元・性別・対象可否 |
| `overrides.yaml` | 手で直す設定（除外する所属、男性、表記ゆれ、選定の固定、既存ファイル） |

## 対象

- Nijisanji: JP・ID の女性ライバー（現役・元）。EN / KR / 中国・台湾・インドは除外
- Hololive: JP・DEV_IS・ID・holoAN・mekPark の女性。English / China / Holostars は除外
- 性別は名簿に無いので `overrides.yaml` の `male` で除外している。抜けがあれば追記する

## 選定ルール

1. Civitai の Illustrious 版 LoRA だけを候補にする（タグ・作品名・キャラ名で検索）
2. 名前の一致に加えて、作品名（nijisanji / hololive など）か VTuber の言及を要求する
   （同名の別作品キャラを避ける）。言及が無い場合は、日本語フルネームか長いローマ字フルネームの一致だけを採る
3. 衣装 1 着専用の名前（outfit / 水着 / 誕生日 など）より、汎用版（複数衣装を含むものなど）を優先する
4. 最多ダウンロード数の 50% 以上のものから、最新の版を選ぶ
5. `overrides.yaml` の `pin`（Civitai の modelId）があればそれを使う
6. ミラーに同じ SHA256 のファイルがあれば、その名前をそのまま使う（改名しない）
7. 選ばれなかった既存のミラーファイルは `alternates` として残す

## 使い方

```bash
# 一覧（id / 日本語名 / ファイル）
python scripts/chara_db.py list

# キャラ指定で取得（id、日本語名、"nijisanji/*" が使える）。start.ipynb の CHARA_LORAS と同じ
python scripts/chara_db.py fetch --chara 椎名唯華 hololive/houshou_marine
python scripts/chara_db.py fetch --chara 椎名唯華 --with-alternates   # 既存の別候補も

# 台帳の更新（名簿 → Civitai 収集 → 選定 → ミラーへ追加）
python scripts/chara_db.py roster
CIVITAI_API_KEY=... python scripts/chara_db.py crawl      # 30〜40 分。キャッシュは workspace/chara_db_cache/
python scripts/chara_db.py build
CIVITAI_API_KEY=... python scripts/chara_db.py upload     # 10 本ずつ 1 コミット
```

選定を変えたいときは `overrides.yaml` の `pin` にキャラ id と Civitai の modelId を書き、`build` → `upload`。
