# Wan2.2 INT8: prestepを維持するための追試

Issue: https://github.com/koshihikari77/paperspace-comfyui/issues/12
作業記録: `/notebooks/wan22-int8-prestep/`。

## 結果

**prestepを省かずにINT8の7step生成を再現できた。** 元の加速LoRA（High3/Low1）・NAG・FETAも維持した候補を3seedで実行し、全81フレームの縮小一覧と代表5フレームを確認。前回の背景破綻は再発しなかった。
元Wrapper FP8とは2seedで比較。人物・背景・動作の流れは近いが、末尾の腕の位置などは異なる。全用途で完全同品質を保証するものではない。
採用候補を4step省略版から **`wrapper_solver_legacy.json`（prestep付き7step）** に変更する。

## 目的

元のprestep付きworkflowにできるだけ近い出力品質を保つ。ノード構成の完全一致は不要だが、prestepを省いて代替完成とはしない。
前回の「native4stepで主要機能確認完了」という結論は、prestepを重要とする用途の代替完成を意味しない。

## 原因の切り分け

### NAGだけを変更した対照

前回色ノイズ・背景破綻が出た `prestep7` seed11111と、今回 `euler_safe_s11111` を比較。
保存先以外のgraph差分はHigh/Lowの `WanVideoNAG` → `WanNAGBranchSafe` の2箇所だけ。
prestep1＋High3＋Low3、CFG3.5/2/1、Euler/simple、shift8、公式Wan2.2 LightX2V各1は維持。
今回は選択フレームで元の背景破綻が消え、人物・服・室内を保って動いた。
したがって、少なくともこの失敗例は「INT8でprestepを使うと壊れる」ことが原因ではなく、NAGの移植経路を修正して改善した。

KJ NAG defaultはbatch数で正負を推測する。batch1は正、batch2は正→負と仮定。
native samplerは正負の区別を `transformer_options['cond_or_uncond']` で渡しており、バッチ数だけでは負のみの呼び出しや逆順を区別できない。
修正版はこのフラグを見て正側だけNAG、負側は通常cross attentionを使う。
元Wrapper自身も `nag_context if not is_uncond else None` としている。

NAG式の係数・正規化式はKJ実装をそのまま利用。toy attentionで正のみ・負のみ・正負順・負正順のテストに合格。
実際のサンプル品質改善の証拠はgraph差分、前後動画、historyとして保存。
全seed・全破綻の唯一の原因とまでは断定しない。

### Wrapper samplerの再利用

前回native7stepは、元のFlowDPMSolverと異なるEuler/simpleを使い、High途中でCFGを切り替えるためsamplerも分割していた。
今回の `WanWrapperNativeSampler` は元Wrapperの `get_scheduler` / `FlowDPMSolverMultistepScheduler` を直接呼ぶ。

- pre 0→1 / High 1→4 / Low 4→7。
- Highの3step内はCFG2→1→1に変えるがsolverを途中で作り直さない。
- Wrapperと同様CPU generator stateを段間で引き継ぐ。
- 最初のGaussian noiseをWrapperと同様に扱う。
- solverのfloat sigmaとモデル評価のinteger timestepを区別。
- INT8モデル・LoRA・メモリ管理はnativeを利用。

Wrapper sigmaは `[0.9999844, .9795743, .9523611, .9142628, .8571160, .7618730, .5713928, 0]`、モデル時刻は `[999,979,952,914,857,761,571]`。
前回native simpleは初期sigma1.0で、後続にも小差があった。
これらは一括してWrapperに合わせたため、改善をsigmaの差だけの効果とはしない。

## 最も元に近づけた候補

`wrapper_solver_legacy.json`:

| 項目 | 設定 |
|---|---|
| モデル | Wan2.2 I2V High/Low INT8 ConvRot |
| prestep | High、加速LoRAなし、1step、CFG3.5 |
| High | 3step、CFG2→1→1、元のWan2.1 LightX2V強度3 |
| Low | 3step、CFG1、同LightX2V強度1 |
| sampler / shift | Wrapper Flow DPM++ SDE / 8 |
| NAG | 11 / .25 / 2.5、正負判別修正版 |
| FETA | 強度1 |
| attention | SageAttention |
| 入出力 | 同一画像・prompt、512×512、81f、16fps、無音 |

用途別LoRAなしの非露骨な入力で品質と実装を切り分けた。全ての制作LoRAや長尺での品質を保証する検証ではない。
元の保存JSONの480×480から、比較用に512×512へ揃えている。Wrapper対照も同じ512×512。
元Wrapper対照はFP8保存/BF16計算/BlockSwap20、native側はINT8保存/FP16計算/native動的メモリ管理。そのため実行時間の比率をINT8単独の効果とは呼ばない。

## 再現方法と範囲

`model-manifest.json` と `fp8-manifest.json` にrevision・SHA256・保存先。
`download.py` / `download_fp8.py` で取得し、`bash start-test-server.sh` で検証用6007を起動。
`python run.py wrapper_solver_legacy.json label seed` でAPI workflowを投入。
追加ノードは `prestep_nodes.py`、T5変換と診断hookは `audit_hook.py`。

ComfyUI本体・WanVideoWrapper・KJNodesの既存ファイルは変更していない。通常6006のworkflowも変更していない。
研究用ノードは今回のWan2.2 14B / batch1向けに検証したもので、全モデル・全batchへの一般的な修正として導入済みではない。

## 実測結果

全runは512×512・81frame・16fps、拡大/RIFEなし。時間はqueue待ちを除くComfyUI開始→完了。

| 構成 | seed | 秒 | 評価 |
|---|---:|---:|---|
| 前回Euler7 + NAG判別修正のみ | 11111 | 69.87 | 前回破綻が解消。保存名以外の変更はNAG2ノードのみ |
| 同上 | 22222 | 198.94 | 正常。起動後初回、T5等ロード込み |
| Wrapper solver + NAG修正 + 公式Wan2.2 LoRA1/1 | 22222 | 73.00 | 正常 |
| **Wrapper solver + NAG修正 + 元LoRA3/1** | **11111** | **113.73** | 正常。Wrapper実行後のnative再ロードあり |
| **同上** | **22222** | **81.47** | 正常 |
| **同上** | **33333** | **75.46** | 正常 |
| 元Wrapper FP8 / BF16 / BlockSwap20 | 22222 | 288.59 | 比較用、model load等を含む |

候補seed22222のサンプラーノード: pre14.13 + High30.26 + Low31.64 =76.03秒。
候補3本ともCUDA INT8 linearを3680回、他INT8 backend呼び出し0と記録。
Wrapper対照のサンプラーノードはpre86.42 + High80.05 + Low93.80秒（重みロード/offload込み）。
キャッシュ・計算精度・メモリ管理も違うため、288.59/81.47という比率をINT8単独の速度向上とは扱わない。今回の目的はprestepと品質の再現。

## 品質の確認

- 候補3本は全81フレームの縮小一覧で、背景の割れ、全画面色ノイズ、人物の消失の再発なし。
- 元Wrapperと同seed11111/22222の横並び動画を作成。11111の対照は第一段階で保存した同条件Wrapper動画、22222は今回再生成。
- 同じ歩行・旋回・腕上げの流れを保ち、服色・室内構図も近い。後半の腕の開き方や手の位置、細部には差がある。
- seed22222の全フレームSSIMは0.945176。ただし背景面積が大きく、SSIMは顔・手・動きの品質を保証する数値ではない。
- 7実行すべてAPI成功、全動画FFmpeg decode成功。
- 全フレームの画像一覧による確認であり、等速連続再生によるちらつきの最終承認や、制作LoRAを含む全条件の承認ではない。

比較動画:
- `wan22-int8-prestep/videos/compare_wrapper_int8_s11111.mp4`
- `wan22-int8-prestep/videos/compare_wrapper_int8_s22222.mp4`

いずれも左Wrapper FP8、右prestep付きnative INT8。

## 残る差分・注意

旧Wan2.1 LightX2VをWan2.2に適用すると、未適用keyのwarningが出る。今回の290種類は全て `k_img` / `v_img` / `norm_k_img` / `img_emb` 系で、FP8/INT8両Wan2.2 checkpointにその枝がないことをheaderで確認。
警告は隠さず `legacy-lora-unloaded-keys.json` に記録。LoRAの全keyが適用されたとは説明しない。

今回再検証したのはprestepを含む生成部分。追加の制作LoRA、拡大・補間・refineを全部同時に積んだ新候補の通し試験はしていない。
前回の拡大・補間等の単独動作確認と区別する。通常workflowへの切替やNotebook/UI統合もまだ行っていない。
候補だけを再現する際は `python download.py --candidate` で必要4重みを取得できる（既存VAEは別途必要）。

Issue追記: https://github.com/koshihikari77/paperspace-comfyui/issues/12#issuecomment-6033124796

検証後処理: 6007停止、一時モデル8ファイル計72,216,767,496 bytes削除。通常6006はidleで維持。動画・コード・manifest・ログは保存済み。
