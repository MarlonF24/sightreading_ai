import os, subprocess, datetime, warnings, music21, logging
from typing import *
from data_pipeline.pipeline import *
from file import *
from log import Log


class Converter():
    """
    A class to handle the conversion of files between .pdf, .mxl, .musicxml, .midi, tokens.
    It uses MuseScore for the conversion process.
    """
    own_path = os.path.abspath(__file__)
    own_directory = os.path.dirname(own_path)
    #pipeline_stages = construct_pipeline()
    
    
    def __init__(self, pipeline: Pipeline, data_folder_path: str=f"{own_directory}\\data", logs_folder_path: str= f"{own_directory}\\logs"):
        # environment.set('musescoreDirectPNGPath', musescore_path)
        self.logs_folder_path = logs_folder_path
        self.data_folder_path = data_folder_path
        os.makedirs(self.logs_folder_path, exist_ok=True)
        os.makedirs(self.data_folder_path, exist_ok=True)
        
        self.pipeline = pipeline
        self.assign_data_folders()  
        self.assign_log_folders()


    def assign_log_folders(self) -> None:
        self.log_folder_map: dict[(Pipeline_stage, Pipeline_stage), str] = {}
        for stage, children in self.pipeline.graph.items():
            for child in children:
                path = os.path.join(self.logs_folder_path)
                self.log_folder_map[(stage, child)] = path
                os.makedirs(path, exist_ok=True)


    def assign_data_folders(self) -> None:
        self.data_folder_map: dict[Pipeline_stage, str] = {}
        for stage in self.pipeline.stages:
            path = os.path.join(self.data_folder_path, stage.name)
            stage.folder_path = path
            self.data_folder_map[stage] = path
            os.makedirs(path, exist_ok=True)


    def multi_stage_conversion(self, start_stage: Pipeline_stage, target_stage: Pipeline_stage, overwrite: bool = True) -> None:
                     
        route = self.find_conversion_route(start_stage, target_stage)

        current_start_stage = start_stage

        for current_target_stage in route[1:]:
            self.single_stage_conversion(current_start_stage, current_target_stage, overwrite)
            current_start_stage = current_target_stage
            


    def single_stage_conversion(self, start_stage: Pipeline_stage, target_stage: Pipeline_stage, overwrite: bool=True) -> None: 
        if target_stage not in start_stage.children:
                raise ValueError(f"Cannot convert from stage: {start_stage.name} to stage: {target_stage.name}")


        conversion_function = start_stage.children[target_stage]
        
        log = Log(start_stage, target_stage, self.log_folder_map[(start_stage, target_stage)])
        
        for file_name in os.listdir(start_stage.folder_path):
            if file_name.endswith(start_stage.extension):            
                input_file = File(os.path.join(start_stage.folder_path, file_name))
                
                output_file = File(os.path.join(target_stage.folder_path, os.path.splitext(file_name)[0] + target_stage.extension))

                self.logged_single_file_conversion(conversion_function, input_file, output_file, log, overwrite)
        
        log.commit()

                
    def logged_single_file_conversion(self, func, input_file: File, output_file: File, log: Log, overwrite: bool):
        log.num_attempted += 1
        
        if os.path.exists(output_file.path) and not overwrite:
            log.skip(input_file, output_file)
        else:
            try: 
                outcome = func(input_file, output_file)
                log.log(outcome)

            except Exception as e:
                log.halt(input_file, e)
                log.commit()
                raise RuntimeError(f"Critical failure since conversion of {input_file.name}. All upcoming conversion aborted\n" + str(e))


if __name__ == "__main__":
    # Example usage
    folders = {
        "pdf_in": "pdf_in",
        "mxl_in": "mxl_in",
        "musicxml_in": "musicxml_in",
        "midi_in": "midi_in",
        "tokens": "tokens",
    }
    print(os.getcwd())
    pipeline = Converter.pipeline_stages
    converter = Converter(folders)
    converter.multi_stage_conversion(pipeline["mxl_in"], pipeline["musicxml_in"], overwrite=True)
