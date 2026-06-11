from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from anime_descale_assist.emit_vpy import emit_vpy


class EmitVpyTests(unittest.TestCase):
    def test_emit_vpy_writes_candidate_zone(self) -> None:
        payload = {
            "zones": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 10.0,
                    "label": "h810_catrom",
                    "action": "descale",
                    "candidate_height": 810,
                    "kernel": "catrom",
                    "confidence": 0.8,
                }
            ]
        }
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "filter.vpy"
            emit_vpy(r"C:\video\input.mkv", payload, out, fps="24000/1001")
            text = out.read_text(encoding="utf-8")
        self.assertIn("LWLibavSource('C:\\\\video\\\\input.mkv')", text)
        self.assertIn("'height': 810", text)
        self.assertIn("'kernel': 'catrom'", text)
        self.assertIn("out.set_output()", text)


if __name__ == "__main__":
    unittest.main()
