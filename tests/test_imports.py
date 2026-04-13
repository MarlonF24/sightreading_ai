import unittest
import sys
from pathlib import Path

# Add src to path just in case, though uv run usually handles this
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.resolve()))

class TestLibrary(unittest.TestCase):
    def test_pipeline_imports(self):
        """Test that core data pipeline modules can be imported."""
        from sightreading_ai.data_pipeline_scripts import converter, pipeline
        self.assertIsNotNone(converter)
        self.assertIsNotNone(pipeline)

    def test_model_imports(self):
        """Test that model and dataloader modules can be imported."""
        from sightreading_ai.model import model, dataloader
        self.assertIsNotNone(model)
        self.assertIsNotNone(dataloader)

    def test_tokeniser_imports(self):
        """Test that tokeniser modules can be imported."""
        from sightreading_ai.tokeniser import tokeniser
        self.assertIsNotNone(tokeniser)

if __name__ == '__main__':
    unittest.main()
