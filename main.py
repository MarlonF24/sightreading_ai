from data_pipeline_scripts.pipeline import construct_music_pipeline
from data_pipeline_scripts.converter import Converter
from tokeniser.tokeniser import MyTokeniser, MyTokeniserConfig
from model.model import MyModel
from model.dataloader import MyTokenDataset
from pathlib import Path
import miditok

if __name__ == "__main__":
    tokeniser1 = MyTokeniser(MyTokeniserConfig(clefs=['G', 'F'],))
    tokeniser2 = MyTokeniser(MyTokeniserConfig(clefs=['F', 'G'],))
    print(tokeniser1.hexa_hash)
    print(tokeniser2.hexa_hash)

    
    # tokeniser = MyTokeniser.from_pretrained(pretrained_model_name_or_path=Path("C:/Users/marlo/sightreading_ai/tokeniser"))

    # pipeline = construct_music_pipeline(tokeniser)

    # converter = Converter(pipeline=pipeline)

    # converter.multi_stage_conversion(converter.pipeline["pdf_in"], converter.pipeline["tokens_in"], batch_if_possible=False)
    # tokeniser.hexa_hash
    # tokeniser.save_pretrained(Path("C:/Users/marlo/sightreading_ai/tokeniser"))
    # MyModel.train_from_tokens_dir(
    #     tokens_dir=Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens_in"),
    #     tokeniser=tokeniser)

    # import json
    # l = []
    # for file in Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens_in").glob("*.json"):
    
    #     with open(file, "r") as f:
    #         data = json.load(f)
    #         l.append(len(data["labels"]))
    
    # print(f"Average length of tokens: {sum(l) / len(l)}")



