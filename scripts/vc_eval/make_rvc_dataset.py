"""RVC training data for the synthetic target voice tgt_f (issue #15): drama3-small keeps the voice by
taking tgt_f's own clip as reference audio. Mostly plain speech plus some voiced/breathy non-verbal lines,
so the model is not trained on non-verbal sounds alone.

  python make_rvc_dataset.py   (eval venv, for soundfile)  ->  OUT/rvc_dataset/tgt_f/*.flac, OUT/rvc_dataset/tgt_f.jsonl
"""
from __future__ import annotations

import json
import shutil

import soundfile as sf

from run_seedvc import COMFY, OUT, TARGETS, output_path, post

SPEECH = [
    "朝の駅はいつも混んでいて、座れたことはほとんどありません。",
    "今日のお昼ごはんは、近くのパン屋さんで買ったサンドイッチにしました。",
    "週末は天気がよさそうなので、少し遠くの公園まで歩いてみようと思います。",
    "この本、最後の章で一気に印象が変わるから、ぜひ最後まで読んでみてね。",
    "すみません、この電車は空港まで行きますか？",
    "明日の会議は十時からなので、資料は九時半までに送ってください。",
    "子どものころ、夏休みになると毎年おばあちゃんの家に泊まりに行っていました。",
    "新しいカフェができたらしいよ。今度一緒に行ってみない？",
    "雨が降りそうだから、念のため傘を持っていったほうがいいと思う。",
    "このアプリ、設定画面がちょっとわかりにくいんですよね。",
    "冷蔵庫に昨日のカレーが残っているから、夕飯はそれでいいかな。",
    "えっと、たしかこの角を曲がったところに郵便局があったはずなんだけど。",
    "ありがとうございます。おかげさまで、無事に引っ越しが終わりました。",
    "猫がキーボードの上に乗ってきて、全然作業が進まないんです。",
    "来月から、毎朝三十分だけ早起きして散歩をすることにしました。",
    "このボタンを押すと、画面の右上に通知が表示されます。",
    "ねえ、聞いて。さっき駅でずっと会いたかった人に偶然会ったの！",
    "ごめんね、返事が遅くなっちゃった。今ちょうど仕事が終わったところ。",
    "図書館は月曜日がお休みなので、行くなら火曜日以降がおすすめです。",
    "夜になると、この通りは街灯がきれいに並んで、とても静かになります。",
    "うーん、どっちの色もかわいいけど、やっぱり青のほうが好きかな。",
    "電子レンジで二分温めてから、よくかき混ぜて召し上がってください。",
    "あの映画、思っていたよりずっと泣ける話だったよ。",
    "テストの結果が返ってきたんだけど、思っていたより良くてびっくりした。",
    "ここから先は工事中なので、右側の歩道を通ってください。",
    "せっかくだから、記念に写真を一枚撮っていこうよ。",
    "最近、寝る前にスマホを見るのをやめたら、少し眠りが深くなった気がします。",
    "その話、もう少し詳しく聞かせてもらってもいいですか？",
    "今年の冬は寒くなるらしいから、新しいコートを買おうか迷ってるんだ。",
    "お待たせしました。ご注文のホットコーヒーとチーズケーキです。",
    "山の上から見た景色は、言葉にできないくらいきれいでした。",
    "大丈夫、心配しないで。ちゃんと準備してきたから。",
    "次の角を左に曲がって、二つ目の信号を渡ったらすぐです。",
    "あれ、鍵どこに置いたっけ。さっきまでここにあったのに。",
    "この曲を聴くと、なぜか高校生のころを思い出すんだよね。",
    "申し訳ありませんが、本日の受付は終了いたしました。",
    "花火大会、今年は晴れるといいね。浴衣も用意したんだ。",
    "パソコンが急に動かなくなって、保存していないデータが消えちゃった。",
    "せーので一緒に開けよう。せーの！",
    "まだ少し時間があるから、本屋さんに寄っていってもいい？",
    "その日は予定があるので、別の日にしてもらえると助かります。",
    "海の近くに住んでいると、風の匂いで季節が変わるのがわかるんです。",
    "最初はうまくいかなかったけど、何度も練習してやっとできるようになったよ。",
    "今日はありがとう。すごく楽しかった。また遊ぼうね。",
]
NONVERBAL = [
    "[laughs] あはは、もう、笑わせないでよ。",
    "[laughing] ふふっ、それ本当に言ったの？",
    "[sighs] はぁ……やっと終わった。",
    "[sighs deeply] ふぅ……ちょっと休憩しよう。",
    "[breathing heavily, out of breath] はぁ、はぁ……ちょっと待って、走りすぎた……。",
    "[gasps] わっ、びっくりした！",
    "<whisper>ねえ、ここだけの話なんだけどね。</whisper>",
    "[yawns] ふぁあ……もう眠くなってきちゃった。",
    "[giggles] えへへ、ありがとう。",
    "[exhales slowly] ふーっ……よし、落ち着いた。",
]


def main() -> None:
    dst = OUT / "rvc_dataset" / "tgt_f"
    dst.mkdir(parents=True, exist_ok=True)
    ref_text = TARGETS["tgt_f"]["text"]
    lines = [("sp", t) for t in SPEECH] + [("nv", t) for t in NONVERBAL]
    log = (OUT / "rvc_dataset" / "tgt_f.jsonl").open("w")
    total = 0.0
    for i, (kind, text) in enumerate(lines):
        for seed in (0, 1):
            prompt = {
                "1": {"class_type": "Drama3SmallLoader", "inputs": {"gguf": "drama3-small-IQ3_XXS.gguf", "device": "auto",
                                                                    "precision": "auto", "expert_offload": "none"}},
                "2": {"class_type": "LoadAudio", "inputs": {"audio": "vc_eval_tgt_f.flac"}},
                "3": {"class_type": "Drama3SmallTTS", "inputs": {
                    "model": ["1", 0], "text": text, "voice_description": "", "seed": seed, "temperature": 0.7,
                    "top_p": 0.7, "top_k": 30, "max_seconds": 30.0, "max_pause_seconds": 0.45,
                    "reference_audio": ["2", 0], "reference_text": ref_text, "keep_model_loaded": True}},
                "4": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0], "filename_prefix": f"vc_eval/rvc_dataset/{kind}{i:02d}_s{seed}"}},
            }
            wall, files = post(prompt)
            name = f"{kind}{i:02d}_s{seed}.flac"
            shutil.copy(output_path(files[0]), dst / name)
            sec = sf.info(dst / name).duration
            total += sec
            log.write(json.dumps({"file": name, "kind": kind, "seed": seed, "text": text, "sec": round(sec, 2)}, ensure_ascii=False) + "\n")
            print(name, round(sec, 1), f"total {total / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
