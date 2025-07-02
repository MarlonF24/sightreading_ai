import os, datetime
from __future__ import annotations
from typing import *
from pipeline import *
from pathlib import Path
from conversion_func_infrastructure import *


class Converter():
    """
    A class to handle the conversion of files between .pdf, .mxl, .musicxml, .midi, tokens.
    It uses MuseScore for the conversion process.
    """
    own_path: FilePath = Path(os.path.abspath(__file__))
    own_directory: FolderPath = own_path.parent
    own_directory_directory = own_directory.parent
    
    
    def __init__(self, pipeline: Pipeline, data_folder_path: FolderPath = Path(fr"{own_directory_directory}\data_pipeline\data"), logs_folder_path: FolderPath = Path(fr"{own_directory_directory}\data_pipeline\logs")) -> None:
        # environment.set('musescoreDirectPNGPath', musescore_path)
        self.logs_folder_path = logs_folder_path
        self.data_folder_path = data_folder_path
        os.makedirs(self.logs_folder_path, exist_ok=True)
        os.makedirs(self.data_folder_path, exist_ok=True)
        
        self.pipeline = pipeline
        self.assign_data_folders()  
        self.assign_log_folders()


    def assign_log_folders(self) -> None:
        self.log_folder_map: dict[Tuple[PipelineStage, PipelineStage], FolderPath] = {}
        for stage in self.pipeline:
            for child in stage.children:
                path = self.logs_folder_path.joinpath(f"{stage.name}_to_{child.name}")
                self.log_folder_map[(stage, child)] = path
                path.mkdir(exist_ok=True)


    def assign_data_folders(self) -> None:
        self.data_folder_map: dict[PipelineStage, FolderPath] = {}
        for stage in self.pipeline:
            path = self.data_folder_path.joinpath(stage.name)
            self.data_folder_map[stage] = path
            path.mkdir(exist_ok=True)


    def multi_stage_conversion(self, start_stage: PipelineStage, target_stage: PipelineStage, overwrite: bool = True, batch_if_possible: bool = True) -> None:
                     
        route = self.pipeline.shortest_conversion_route(start_stage, target_stage)

        print(f"\nConverting from {start_stage.name} to {target_stage.name} via {[stage.name for stage in route]}...\n")

        current_start_stage = start_stage

        for current_target_stage in route[1:]:
            self.single_stage_conversion(current_start_stage, current_target_stage, overwrite, batch_if_possible)
            current_start_stage = current_target_stage
            

    def single_stage_conversion(self, start_stage: PipelineStage, target_stage: PipelineStage, overwrite: bool=True, batch_if_possible: bool = True) -> None: 
        if target_stage not in start_stage.children:
                raise ValueError(f"Cannot convert from stage: {start_stage.name} to stage: {target_stage.name}")

        conversion_function = start_stage.children[target_stage]
        
        log = Log(self, start_stage, target_stage)
        
        if batch_if_possible and isinstance(conversion_function, BatchConversionFunction):
            print(f"Batch converting from {start_stage.name} to {target_stage.name}...\n")
            self.logged_batch_file_conversion(conversion_function, start_stage, target_stage, log, overwrite)
        else:
            print(f"Single-file converting from {start_stage.name} to {target_stage.name}...\n")
            self.logged_single_file_conversion(conversion_function, start_stage, target_stage, log, overwrite)
        
        print(f"\n\nConversion from {start_stage.name} to {target_stage.name} completed. Log saved as {log.path.name} in {log.path.parent}.\n")


    def logged_single_file_conversion(self, conversion_function: ConversionFunction, start_stage: PipelineStage, target_stage: PipelineStage, log: Log, overwrite: bool) -> None:
        start_folder = self.data_folder_map[start_stage]
        output_folder = self.data_folder_map[target_stage]

        for _file in start_folder.glob(f"*{start_stage.extension}"):          
            input_file: FilePath = FilePath(_file)
            
            if isinstance(conversion_function, BatchConversionFunction):
                outcome = conversion_function(input_file, output_folder, do_batch=False, overwrite=overwrite)
            else:
                outcome = conversion_function(input_file, output_folder, overwrite=overwrite)
            log.log(outcome)

        log.commit()

    def logged_batch_file_conversion(self, conversion_function: BatchConversionFunction, start_stage: PipelineStage, target_stage: PipelineStage, log: Log, overwrite: bool) -> None:
        start_folder = self.data_folder_map[start_stage]
        output_folder = self.data_folder_map[target_stage]
    
        outcome = conversion_function(start_folder, output_folder, do_batch=True, overwrite=overwrite)
        log.log(outcome)
        log.commit()    


class Log():
    class HaltError(Exception):
        pass

    def __init__(self, converter: Converter, start_stage: PipelineStage, target_stage: PipelineStage) -> None:
        self.start_stage: PipelineStage = start_stage
        self.target_stage: PipelineStage = target_stage
        self.converter: Converter = converter

        self.path: FilePath = Path(self.converter.log_folder_map[(start_stage, target_stage)].joinpath(f"{start_stage.name}_to_{target_stage.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"))

        self.start_stage_folder_path: FolderPath = self.converter.data_folder_map[start_stage]
        self.target_stage_folder_path: FolderPath = self.converter.data_folder_map[target_stage]
        self.text: List[str] = [self.path.name + f"\n {self.start_stage_folder_path} -> {self.target_stage_folder_path}" + 2 * "\n"]
        
        self.num_total_files: int = len(os.listdir(self.converter.data_folder_map[start_stage]))
        self.num_attempted: int = 0
        self.num_skipped: int = 0 
        self.num_successful: int = 0 
        self.num_errors: int = 0
        self.num_warned_successful: int = 0
        self.has_halted: bool = False

    @property
    def index(self) -> str:
        return f"[{self.num_attempted}/{self.num_total_files}]"

    def log_skip(self, outcome: ConversionOutcome) -> None:
        self.num_skipped += 1
        
        if not outcome.error_message:
            reason = f"Output: {f"{outcome.output_files[0].name} already exists" if len(outcome.output_files) == 1 else f"Files {outcome.output_files[0].name}... already exist"} and shall not be overwritten\n"

        self.text.append(f"{self.index} [SKIPPED] {datetime.datetime.now()} - Input: {outcome.input_file.name}; \n" + reason)
        
        print(f"{self.index} [SKIPPED]", end="\r")

    def log_success(self, outcome: ConversionOutcome) -> None:
        self.num_successful += 1

        self.text.append(f"{self.index} [SUCCESS] {datetime.datetime.now()} - Input: {outcome.input_file.name}; Output: {outcome.output_files[0].name if len(outcome.output_files) == 1 else f"{len(outcome.output_files)} Files ({outcome.output_files[0].name})..."}\n") 
    
        if outcome.warning_messages:
            self.num_warned_successful += 1
            for warning_message in outcome.warning_messages:
                self.text.append(f"\t[WARNING] {warning_message}\n")
        
        print(f"{self.index} [SUCCESS]", end="\r")

    def log_error(self, outcome: ConversionOutcome) -> None:
        self.num_errors += 1
             
        self.text.append(f"{self.index} [ERROR] {datetime.datetime.now()} - Input: {outcome.input_file.name}\n" + 
                        f"\tError: {outcome.error_message}\n")
        
        print(f"{self.index} [ERROR]", end="\r")

    def log_halt(self, outcome: ConversionOutcome) -> None:
        self.num_errors += 1
        
        self.has_halted = True
        
        self.text.append(f"{self.index} [HALT] {datetime.datetime.now()} - ON {outcome.input_file.name}\n"+ f"\tError: {outcome.error_message}\n")
        
        self.commit()
        
        raise Log.HaltError(f"\nSingle-file conversion from {self.start_stage.name} to {self.target_stage.name} halted. Potential following conversions aborted.\nCritical failure since conversion of {self.index, outcome.input_file.name}.\n\tError:" + outcome.error_message)
    

    def log(self, outcomes: List[ConversionOutcome]) -> None:
        for outcome in outcomes:
            self.num_attempted += 1

            if outcome.skipped:
                self.log_skip(outcome)
            elif outcome.successful:
                self.log_success(outcome)
            else:
                if outcome.halt:
                    self.log_halt(outcome)
                else:
                    self.log_error(outcome)
                
    
    @property
    def stats(self):
        return {
        "num_total": self.num_total_files,
        "num_attempted": self.num_attempted,
        "num_successful": (self.num_successful, self.num_successful / self.num_attempted if self.num_attempted else 0.0),
        "num_skipped": (self.num_skipped, self.num_skipped / self.num_attempted if self.num_attempted else 0.0),
        "num_errors": (self.num_errors, self.num_errors / self.num_attempted if self.num_attempted else 0.0),
        "num_warned_from_successful": (self.num_warned_successful, self.num_warned_successful / self.num_successful if self.num_successful else 0.0),
        "has_halted": self.has_halted
        }
    
    def insert_evaluation(self):
        self.text[1:1] = "\n"
        self.text[1:1] = [
            f"{key}: {value[0]} ({value[1] * 100:.2f}%)\n" if isinstance(value, tuple) else f"{key}: {value}\n"
            for key, value in self.stats.items()
        ]
        
    
    def commit(self):
        self.insert_evaluation()
        file = self.path.open(mode="w", encoding="utf-8")
        file.write("".join(self.text))


if __name__ == "__main__":
    pipeline = construct_music_pipeline()
    converter = Converter(pipeline)
    converter.multi_stage_conversion(converter.pipeline["pdf_in"], converter.pipeline["mxl_in"], overwrite=True, batch_if_possible=True)
