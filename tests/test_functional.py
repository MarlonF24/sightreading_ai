import unittest
import sys
import shutil
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.resolve()))

from tokeniser.tokeniser import MyTokeniser, Metadata
from model.model import MyModel
from data_pipeline_scripts.pipeline import construct_music_pipeline
from data_pipeline_scripts.converter import Converter

class TestFunctionalSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a temporary directory for testing."""
        cls.test_dir = Path(__file__).parent / "test_output"
        cls.test_dir.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory."""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_converter_init(self):
        """Test that the Converter initializes and creates the correct directory structure."""
        tokenizer = MyTokeniser()
        pipeline = construct_music_pipeline(tokeniser=tokenizer)
        
        # Test initialization with custom path
        converter_root = self.test_dir / "converter_test"
        converter = Converter(pipeline=pipeline, pipeline_dir_path=converter_root)
        
        # Verify root directories exist (based on data_pipeline_constants)
        # Note: Converter appends 'data_pipeline' to the root path
        full_root = converter_root / "data_pipeline"
        self.assertTrue(full_root.exists())
        self.assertTrue((full_root / "data").exists())
        self.assertTrue((full_root / "logs").exists())
        self.assertTrue((full_root / "temp").exists())

    def test_model_init_and_paths(self):
        """Test that MyModel initializes with correct instance paths."""
        from transformers import GPT2Config
        
        # Create a dummy config
        config = GPT2Config(
            architectures=["MyModel"],
            n_embd=128,
            n_layer=2,
            n_head=2,
            vocab_size=100
        )
        
        model_root = self.test_dir / "model_test"
        model = MyModel(config=config, model_dir_path=model_root)
        
        # Verify instance paths are correctly derived
        # MyModel appends 'model' to the root path
        expected_root = model_root / "model"
        self.assertEqual(model.model_dir_path, expected_root)
        self.assertEqual(model.training_dir, expected_root / "training")
        self.assertEqual(model.output_dir, expected_root / "output")

    def test_metadata_object(self):
        """Test that the Metadata conditioning object can be created as described in README."""
        metadata = Metadata.TokenisedMetadata(
            time_signature="4/4",
            num_measures=16,
            density_complexity=5,
            duration_complexity=3,
            interval_complexity=4
        )
        self.assertEqual(metadata.time_signature, "4/4")
        self.assertEqual(metadata.density_complexity, 5)

if __name__ == '__main__':
    unittest.main()
