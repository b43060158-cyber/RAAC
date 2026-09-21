from pathlib import Path
import tempfile
import unittest

from autogen_mas.dataset_adapters import CIARAdapter


class CIARAdapterTest(unittest.TestCase):
    def test_rejects_malformed_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ciar.json"
            path.write_text('[{"question": "Q?", "answer": ["A"]}]', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "incorrect answer"):
                CIARAdapter(path).load()


if __name__ == "__main__":
    unittest.main()
