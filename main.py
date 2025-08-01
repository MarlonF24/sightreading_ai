from data_pipeline_scripts.pipeline import construct_music_pipeline
from data_pipeline_scripts.converter import Converter
from tokeniser.tokeniser import MyTokeniser, MyTokeniserConfig, Metadata
from model.model import MyModel
from model.dataloader import MyTokenDataset
from pathlib import Path
import miditok

if __name__ == "__main__":
    # metadata_tokens = Metadata.TokenisedMetadata(
    #    key_signature=0,
    #    time_signature="4/4",
    #    rh_clef="G",
    #    lh_clef="F",
    #    lowest_pitch=30,
    #    highest_pitch=100,
    #    num_measures=16,
    #    density_complexity=5,
    #    duration_complexity=3,
    #    interval_complexity=4
    # )

    # MyModel.generate_tokens(
    #     metadata_tokens=metadata_tokens
    # )


    # tokeniser = MyTokeniser()
    # tokeniser.train_BPE(data_dir=Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/midi_in"))
    # tokeniser.save_pretrained("C:/Users/marlo/sightreading_ai/tokeniser")

    tokeniser = MyTokeniser.from_pretrained("C:/Users/marlo/sightreading_ai/tokeniser")
    pipeline = construct_music_pipeline(tokeniser=tokeniser, pdf_preprocess=False)
    converter = Converter(pipeline=pipeline)
    converter.multi_stage_conversion(converter.pipeline["midi_in"], converter.pipeline["tokens_in"], batch_if_possible=False, overwrite=True)

    MyModel.train_from_tokens_dir(tokens_dir=Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens_in"), tokeniser=tokeniser)


    # import json
    # l = []
    # for file in Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens_in").glob("*.json"):
    
    #     with open(file, "r") as f:
    #         data = json.load(f)
    #         l.append(len(data["labels"]))
    
    # print(f"Average length of tokens: {sum(l) / len(l)}")
    # print(f"Total number of tokens: {sum(l)}")
    pass

