from data_pipeline_scripts.pipeline import construct_music_pipeline
from data_pipeline_scripts.converter import Converter
from tokeniser.tokeniser import MyTokeniser

if __name__ == "__main__":
    # Initialize the tokeniser
    tokeniser = MyTokeniser()
    
    pipeline = construct_music_pipeline(tokeniser=tokeniser)
    
    # Create a converter instance with the pipeline
    converter = Converter(pipeline=pipeline)

    converter.multi_stage_conversion(converter.pipeline["midi_in"], converter.pipeline["tokens_in"])