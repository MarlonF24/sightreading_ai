from data_pipeline_scripts.pipeline import construct_music_pipeline
from data_pipeline_scripts.converter import Converter
from tokeniser.tokeniser import MyTokeniser, MyTokeniserConfig, Metadata
from model.model import MyModel
from model.dataloader import MyTokenDataset
from pathlib import Path

if __name__ == "__main__":
    # metadata_tokens = Metadata.TokenisedMetadata(
    #    time_signature="4/4",
    #    num_measures=10,
    #    density_complexity=3,
    #    duration_complexity=3,
    #    interval_complexity=3
    # )

    # MyModel.generate_tokens(
    #     metadata_tokens=metadata_tokens,
    #     key_signature=-3,
    #     output_dir=Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens_out")
    # )


    # tokeniser = MyTokeniser()
    # tokeniser.train_BPE(data_dir=Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/midi_in"))

    
    # tokeniser.save_pretrained("C:/Users/marlo/sightreading_ai/tokeniser")
    

    tokeniser = MyTokeniser.from_pretrained("C:/Users/marlo/sightreading_ai/tokeniser")

    pipeline = construct_music_pipeline(tokeniser=tokeniser, pdf_preprocess=False)
    converter = Converter(pipeline=pipeline)
   
    # converter.load_stage_data_from_temp("mxl_in")

    converter.multi_stage_conversion("mxl_in", "midi_in", batch_if_possible=False, overwrite=True, move_successful_inputs_to_temp=False)

    # converter.move_stage_data_to_temp("mxl_in")
    
    
    # MyModel.train_from_tokens_dir(tokens_dir=Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens_in"), tokeniser=tokeniser)
 

    # import json
    # l = []
    # for file in Path("C:/Users/marlo/sightreading_ai/data_pipeline/data/tokens_in").glob("*.json"):
    
    #     with open(file, "r") as f:
    #         data = json.load(f)
    #         l.append(len(data["labels"]))
    
    # print(f"Average length of tokens: {sum(l) / len(l)}")
    # print(f"Total number of tokens: {sum(l)}")
     


    # from data_pipeline_scripts.conversion_functions import Generics
    # import shutil
    
    # string = "Hello, World!"
    # print(string)
    # try:
    #         terminal_width = shutil.get_terminal_size().columns
    
    # except (AttributeError, OSError):
    #     terminal_width = 80
    
    #     # Calculate how many lines the filename will take
    #     lines_needed = (len(string) + len("") + terminal_width - 1) // terminal_width  # Ceiling division

    #     # Clear the calculated number of lines
    #     Generics.clear_n_terminal_lines(lines_needed)


    pass

