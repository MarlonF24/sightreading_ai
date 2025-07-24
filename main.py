from data_pipeline_scripts.pipeline import construct_music_pipeline
from data_pipeline_scripts.converter import Converter
from tokeniser.tokeniser import MyTokeniser, MyTokeniserConfig
from model.model import MyModel
from model.dataloader import MyTokenDataset
from pathlib import Path
import miditok

if __name__ == "__main__":
    tokeniser = MyTokeniser.from_pretrained(pretrained_model_name_or_path=Path("C:/Users/marlo/sightreading_ai/tokeniser"))

    pipeline = construct_music_pipeline(tokeniser)

    converter = Converter(pipeline=pipeline)

    converter.multi_stage_conversion(converter.pipeline["pdf_in"], converter.pipeline["tokens_in"])

    tokeniser.save_pretrained(Path("C:/Users/marlo/sightreading_ai/model/training"))
    MyModel.train_from_tokens_folder(
        tokens_folder=Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens_in"),
        tokeniser=tokeniser)


