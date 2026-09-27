"""whisper.cpp output becomes the transcript shape the helpers already read."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "helpers"))

from whisper_local import to_scribe  # noqa: E402


def seg(a, b, text):
    return {"offsets": {"from": a, "to": b}, "text": text}


class ToScribeTest(unittest.TestCase):
    def test_words_with_spacing_between_them(self):
        out = to_scribe({
            "result": {"language": "fr"},
            "transcription": [seg(0, 70, ""), seg(70, 1140, " Bonjour,"),
                              seg(1140, 1570, " je"), seg(1600, 2210, " suis")],
        })
        words = [w for w in out["words"] if w["type"] == "word"]
        self.assertEqual([w["text"] for w in words], ["Bonjour,", "je", "suis"])
        self.assertEqual((words[0]["start"], words[0]["end"]), (0.07, 1.14))
        # Scribe puts a spacing entry between words; pack_transcripts reads the gaps.
        gaps = [w for w in out["words"] if w["type"] == "spacing"]
        self.assertEqual(len(gaps), 2)
        self.assertEqual((gaps[1]["start"], gaps[1]["end"]), (1.57, 1.6))
        self.assertEqual(out["language_code"], "fr")
        self.assertEqual(out["text"], "Bonjour, je suis")

    def test_leading_silence_is_dropped_and_one_speaker(self):
        out = to_scribe({"transcription": [seg(0, 500, "  "), seg(500, 900, " Hi")]})
        self.assertEqual(len(out["words"]), 1)
        self.assertEqual(out["words"][0]["speaker_id"], "speaker_0")


if __name__ == "__main__":
    unittest.main()
