import os, datetime
from typing import *
from pipeline import *
from file import *
from conversion_functions import ConversionFunction
from __future__ import annotations


class Converter():
    """
    A class to handle the conversion of files between .pdf, .mxl, .musicxml, .midi, tokens.
    It uses MuseScore for the conversion process.
    """
    own_path: Path = os.path.abspath(__file__)
    own_directory: Path = os.path.dirname(own_path)
    own_directory_directory = os.path.dirname(own_directory)
    
    
    def __init__(self, pipeline: Pipeline, data_folder_path: Path=fr"{own_directory_directory}\data", logs_folder_path: Path= fr"{own_directory_directory}\logs") -> None:
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

        print(f"\nConverting from {start_stage.name} to {target_stage.name} via {[stage.name for stage in route]}...\n")

        current_start_stage = start_stage

        for current_target_stage in route[1:]:
            self.single_stage_conversion(current_start_stage, current_target_stage, overwrite)
            current_start_stage = current_target_stage
            


    def single_stage_conversion(self, start_stage: PipelineStage, target_stage: PipelineStage, overwrite: bool=True) -> None: 
        if target_stage not in start_stage.children:
                raise ValueError(f"Cannot convert from stage: {start_stage.name} to stage: {target_stage.name}")

        print(f"\nConverting from {start_stage.name} to {target_stage.name}...\n")

        conversion_function = start_stage.children[target_stage]

        start_data_folder_path = self.data_folder_map[start_stage]
        target_data_folder_path = self.data_folder_map[target_stage]
        
        log = Log(self, start_stage, target_stage)
        

        for file_name in os.listdir(start_data_folder_path):
            if file_name.endswith(start_stage.extension):            
                input_file = File(os.path.join(start_data_folder_path, file_name))
                output_file = File(os.path.join(target_data_folder_path, os.path.splitext(file_name)[0] + target_stage.extension))

                try:
                    self.logged_single_file_conversion(conversion_function, input_file, output_file, log, overwrite)
                except Log.HaltError as e:
                    raise RuntimeError(f"{start_stage.name} to {target_stage.name} conversion halted. Potential following conversions aborted.\n{e}")
                
        log.commit()
        
        print(f"\nConversion from {start_stage.name} to {target_stage.name} completed. Log saved as {log.name} in {log.folder_path}.\n")

                
    def logged_single_file_conversion(self, conversion_function: ConversionFunction, input_file: File, output_file: File, log: Log, overwrite: bool) -> None:
        log.num_attempted += 1
    
        if os.path.exists(output_file.path) and not overwrite:
            log.skip(input_file, output_file)
        else:
            outcome = conversion_function(input_file, output_file)
            log.log(outcome)
        print(f"{log.index} {"[SUCCESS]" if outcome.successful else "[ERROR]"}", end="\r")


class Log(File) :
    class HaltError(Exception):
        pass

    def __init__(self, converter: Converter, start_stage: PipelineStage, target_stage: PipelineStage) -> None:
        
        self.start_stage: PipelineStage = start_stage
        self.target_stage: PipelineStage = target_stage
        self.converter: Converter = converter
        
        super().__init__(os.path.join(self.converter.log_folder_map[(start_stage, target_stage)], f"{start_stage.name}_to_{target_stage.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"))

        self.start_stage_folder_path: Path = self.converter.data_folder_map[start_stage]
        self.target_stage_folder_path: Path = self.converter.data_folder_map[target_stage]
        self.text: List[str] = [self.name + f"\n {self.start_stage_folder_path} -> {self.target_stage_folder_path}" + 2 * "\n"]
        
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

    def skip(self, input_file: File, output_file: File, reason: str = "") -> None:
        self.num_skipped += 1
        if reason is None:
            reason = f"Output: {output_file.name} already exists and shall not be overwritten\n"
        self.text.append(f"{self.index} [SKIPPED] {datetime.datetime.now()} - Input: {input_file.name}; \n"
                         + reason)
    
    
    def log(self, outcome: ConversionOutcome) -> None:
        if outcome.successful:
            self.num_successful += 1
            self.text.append(f"{self.index} [SUCCESS] {datetime.datetime.now()} - Input: {outcome.input_file.name}; Output: {outcome.output_file.name}:\n") 
            

            if outcome.warning_messages:
                self.num_warned_successful += 1
                for warning_message in outcome.warning_messages:
                    self.text.append(f"\t[WARNING] {warning_message}\n")

        else:
            self.num_errors += 1
            if outcome.go_on:
                self.text.append(f"{self.index} [ERROR] {datetime.datetime.now()} - Input: {outcome.input_file.name}\n:" + 
                                 f"\tError: {outcome.error_message}\n")
            else:
                self.has_halted = True
                self.text.append(f"{self.index} [HALT] {datetime.datetime.now()} - ON {outcome.input_file.name}\n"+ 
                                        f"\tError: {outcome.error_message}\n")
                self.commit()
                raise Log.HaltError(f"Critical failure since conversion of {outcome.input_file.name}. All upcoming conversions aborted.\n\tError:" + outcome.error_message)
    
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
            f"{key}: {value[0]} ({value * 100:.2f}%\n)" if isinstance(value, Tuple[int, float]) else f"{key}: {value}\n"
            for key, value in self.stats.items()
        ]
        
    
    def commit(self):
        self.insert_evaluation()
        with open(self.path, "w", encoding="utf-8") as file:
            file.write("".join(self.text))


if __name__ == "__main__":
    # Example usage
    pipeline = construct_music_pipeline()
    converter = Converter(pipeline)
    converter.multi_stage_conversion(converter.pipeline["mxl_in"], converter.pipeline["musicxml_in"], overwrite=True)
