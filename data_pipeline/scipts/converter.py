import os
from typing import *
from pipeline import *
from file import *
from log import Log
from conversion_functions import ConversionFunction


class Converter():
    """
    A class to handle the conversion of files between .pdf, .mxl, .musicxml, .midi, tokens.
    It uses MuseScore for the conversion process.
    """
    own_path: Path = os.path.abspath(__file__)
    own_directory: Path = os.path.dirname(own_path)
    own_directory_directory = os.path.dirname(own_directory)
    
    
    def __init__(self, pipeline: Pipeline, data_folder_path: Path=f"{own_directory_directory}\\data", logs_folder_path: Path= f"{own_directory_directory}\\logs") -> None:
        # environment.set('musescoreDirectPNGPath', musescore_path)
        self.logs_folder_path = logs_folder_path
        self.data_folder_path = data_folder_path
        os.makedirs(self.logs_folder_path, exist_ok=True)
        os.makedirs(self.data_folder_path, exist_ok=True)
        
        self.pipeline = pipeline
        self.assign_data_folders()  
        self.assign_log_folders()


    def assign_log_folders(self) -> None:
        self.log_folder_map: dict[Tuple[PipelineStage, PipelineStage], Path] = {}
        for stage in self.pipeline:
            for child in stage.children:
                path = os.path.join(self.logs_folder_path, f"{stage.name}_to_{child.name}")
                self.log_folder_map[(stage, child)] = path
                os.makedirs(path, exist_ok=True)


    def assign_data_folders(self) -> None:
        self.data_folder_map: dict[PipelineStage, Path] = {}
        for stage in self.pipeline:
            path = os.path.join(self.data_folder_path, stage.name)
            self.data_folder_map[stage] = path
            os.makedirs(path, exist_ok=True)


    def multi_stage_conversion(self, start_stage: PipelineStage, target_stage: PipelineStage, overwrite: bool = True) -> None:
                     
        route = self.pipeline.shortest_conversion_route(start_stage, target_stage)

        print(f"Converting from {start_stage.name} to {target_stage.name} via {[stage.name for stage in route]}...\n")

        current_start_stage = start_stage

        for current_target_stage in route[1:]:
            self.single_stage_conversion(current_start_stage, current_target_stage, overwrite)
            current_start_stage = current_target_stage
            


    def single_stage_conversion(self, start_stage: PipelineStage, target_stage: PipelineStage, overwrite: bool=True) -> None: 
        if target_stage not in start_stage.children:
                raise ValueError(f"Cannot convert from stage: {start_stage.name} to stage: {target_stage.name}")

        print(f"Converting from {start_stage.name} to {target_stage.name}...\n")

        conversion_function = start_stage.children[target_stage]
        
        log = Log(start_stage, target_stage, self.log_folder_map[(start_stage, target_stage)], self.data_folder_map[start_stage], self.data_folder_map[target_stage])
        
        for file_name in os.listdir(self.data_folder_map[start_stage]):
            if file_name.endswith(start_stage.extension):            
                input_file = File(os.path.join(self.data_folder_map[start_stage], file_name))
                
                output_file = File(os.path.join(self.data_folder_map[target_stage], os.path.splitext(file_name)[0] + target_stage.extension))
                try:
                    self.logged_single_file_conversion(conversion_function, input_file, output_file, log, overwrite)
                except Log.HaltError as e:
                    raise RuntimeError(f"{start_stage.name} to {target_stage.name} conversion halted. Following conversions aborted.\nHalt Error: {e}")

        log.commit()
        
        print(f"Conversion from {start_stage.name} to {target_stage.name} completed. Log saved as {log.name} in {log.folder}.\n")

                
    def logged_single_file_conversion(self, conversion_function: ConversionFunction, input_file: File, output_file: File, log: Log, overwrite: bool) -> None:
        log.num_attempted += 1
        
        if os.path.exists(output_file.path) and not overwrite:
            log.skip(input_file, output_file)
        else:
            outcome = conversion_function(input_file, output_file)
            log.log(outcome)


if __name__ == "__main__":
    # Example usage
    pipeline = construct_music_pipeline()
    converter = Converter(pipeline)
    converter.multi_stage_conversion(converter.pipeline["mxl_in"], converter.pipeline["musicxml_in"], overwrite=True)
