# クラッシュ診断（2026-09-26強化）

現行ログは `/notebooks/logs/runtime/notebooks/<PAPERSPACE_NOTEBOOK_ID>/`。
以前の `/notebooks/logs/runtime/` のログは残す。Notebookごとにロック・状態を分離する。

## 取得する証拠

- `process.jsonl`: 起動ラッパーが子ComfyUIの終了コード、終了シグナル、転送したシグナル、終了前後のOOMカウンターを保存。SIGKILLだけでOOMと断定しない。
- `comfyui.log`（既存の `/storage/ComfyUI/user/logs/`）: `PYTHONFAULTHANDLER=1` によりPythonが処理できるSIGSEGV/SIGABRT等のスタックも出力。SIGKILLは捕捉不可。
- `fast_memory.jsonl`: HTTP・GPUコマンドから独立したプロセスで毎秒、コンテナと参照可能な親cgroupのOOM・メモリ圧力、ホストの空きRAMを記録。親のOOMは他プロセスの可能性もある。
- `file_memory.jsonl`: 独立プロセスが10秒ごとにComfyUIの `/proc/<pid>/smaps` を読み、ファイルパス別のRSS・PSS・匿名化されたページ量を集計。Wan High/Low・T5・LoRA等のモデルファイルは常駐0でも記録、その他はRSS 1MiB以上を記録し小さいファイルは合計のみ保存。モデル内容は読み込まない。RSSには共有ページの重複があり、PSSは共有分を按分する。file_backed_rssはRSSからAnonymousを引いた値。マップを外した後のOSキャッシュはファイル別に帰属できないため、cgroupのfile/file_mapped合計も併記する。次回起動のincidentレポートに最終サンプルを含める。
- `events.jsonl`: 既存情報に加え毎分GPUの温度上限・スロットリング理由・ECC情報を保存。カーネルログは権限不足なら取得失敗も記録。
- `gpu_events.jsonl`: NVMLでGPUのXid通知を直接購読。カーネルログ権限は不要。この環境のA6000で購読成功を実測済み。通知があればXid番号と時刻を即flush/fsyncする。過去のエラーを遡って取得する機能ではない。
- `nodes.jsonl`: debugモードで対象ノード実行中も毎秒RAM/VRAM記録。既存のmemory_debug/http_debugも有効になるため、診断のCPU/I/O負荷が加わる。
- `memory_debug.jsonl`: CPUノード出力の容量、1MiB以上のテンソルの出力元・生存時間・キャッシュ外での生存をweakrefで追跡する。生成中10秒ごと、終了時には一部のPython参照元の型も記録。強制GC・解放実験は行わない。生存＝リークとは断定せず、観測上限・省略数も残す。次回起動のincidentレポートに直前の追跡サンプルを含める。
- `lifecycle.jsonl` / `lifecycle-state.json`: 生存時刻・終了シグナル・前回の終了記録の有無。

各JSONLは10MiB×5世代。debugログを含め通常最大約550MiB/Notebook（incidentsと既存comfyui.logは別）。Notebookを切り替えると別枠になる。
生成物、prompt本文、環境変数全体、APIキーは新しい記録には保存しない。既存ComfyUIログの本文には含まれる場合がある。

## 起動

`start.ipynb` の起動セル／`bootstrap_comfyui.py --phase start` は監視付き起動を使う。
すでにComfyUIが起動済みなら再利用するため、終了コード監視は次の起動からになる。
監視用ラッパーは異常終了しても生成を自動再実行しない。
`/tmp/paperspace-comfyui-6006.pid` はラッパーPID。実ComfyUI PIDは `supervisor-state.json` に保存する。

## 外部監視（未配置）

外部PCは使用しない運用とした。以下は任意の手段として残すが、現在の起動・監視の前提条件ではない。

同じ環境内の監視はコンテナ停止・ホスト障害・ストレージ停止と同時に止まり得る。
別PC/サーバーに `scripts/watch_remote_runtime.py`、`collect_runtime_logs.py`、
`runtime_lifecycle.py` を同じディレクトリへコピーし、既存のSSH接続を使って実行する:

```bash
python watch_remote_runtime.py --ssh-target YOUR_SSH_ALIAS --notebook-id YOUR_NOTEBOOK_ID
```

SSHは鍵認証・BatchMode・タイムアウト付き。接続先のホスト鍵確認を無効化しない。
ログは外部PCの `paperspace-external-logs/external.jsonl` に保存する。
これは遠隔の生存確認であり、ネットワーク障害とPaperspaceによる停止の理由を単独では区別できない。
外部PCの時計同期が必要。Notebook内部から同じNotebookを監視しようとすると拒否する。

## 未取得の情報／完了条件

`coverage.json` に現時点の取得状況を明示する。
`python scripts/check_crash_diagnostics.py` で実プロセス・最新ログ・ノード読み込みを再確認し更新する。
終了コード0はローカル監視の稼働を示すだけで、全原因の特定が可能という意味ではない。
起動セルはテンソル追跡を含む全ローカル項目を検査し、不足があれば失敗を表示する（実行中の生成を停止しない）。
`hooks-state.json` は実PID・プロセス起動時刻・読み込み時のソースSHA256を保存する。
ファイルを書き換えただけの旧プロセス、PID再利用、過去のJSONLマーカーを有効化済みと扱わない。
`coverage.json` は監視中も更新し、欠落項目が変化すると `events.jsonl` に記録する。
監視プロセスの例外は `collector-launch.log` に保存する。
`platform-state.json` は現在のサービスアカウントによるPod参照の実測結果を記録する。
403を取得成功として扱わず、Paperspace自動停止設定や停止理由が接続済みとも扱わない。
外部監視は接続先未提供のため未配置。Paperspace Notebookの正式な停止理由・自動停止設定も未接続。
公開Core Machine APIをNotebook APIの代わりに使わない。
原因特定の全経路が揃ったとは扱わない。

監視の次回起動時、前回の最終生存時刻付近の証拠を `incidents/<時刻>/evidence.json` へ自動保存する。
Notebook IDが変わった場合も、直近20セッションのうち記録が途絶した過去Notebookを確認する。
証拠には監視の稼働状況、テンソル追跡の読み込み状態、最終RAM/VRAMと基盤情報の取得可否を含める。
手動確認は `python scripts/diagnose_runtime.py`。別Notebookの記録は `--log-dir` で指定する。
終了コード・Xid・OOMカウンター増加を列挙し、SIGKILL単独や記録途絶を停止理由にすり替えない。
GPUの通知処理は https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlEvents.html に基づく。

完全な停止原因の調査には、ユーザーがアクセスできるPaperspace管理画面の記録、
またはPaperspaceサポートが提供する停止理由・ホストのOOM/Xid/障害記録が必要。
手動の停止予定時刻は `runtime_lifecycle.py --shutdown-at '<timezone付き日時>'` で記録可能。
予定時刻との一致だけで自動停止を確定しない。

## 検証

`python -m unittest discover -s tests -v`
試験用の子プロセスで異常終了・SIGKILL・SIGABRT・SIGTERMを発生させ、
終了コード・fatal traceback・シグナル転送の記録を確認する。本番ComfyUIを殺す試験は行わない。
