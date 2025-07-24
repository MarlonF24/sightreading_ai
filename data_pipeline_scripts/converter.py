from __future__ import annotations
import os, datetime
from typing import *
from data_pipeline_scripts.pipeline import *
from pathlib import Path
from data_pipeline_scripts.conversion_func_infrastructure import *
from data_pipeline_scripts.conversion_func_infrastructure import _ConversionFunction


class Converter():
    """
    A class responsible for managing the conversion process between different file formats.

    Class Attributes:
        OWN_PATH (FilePath): The absolute file path of the current Converter class.
        OWN_DIRECTORY (dirPath): The directory containing the current Converter class.
        OWN_DIRECTORY_DIRECTORY (dirPath): The directory containing the OWN_DIRECTORY directory.

    Attributes:
        pipeline (Pipeline): The pipeline object representing the conversion flow.
        data_dir_path (dirPath): The path to the dir containing the input data files.
        logs_dir_path (dirPath): The path to the dir where the log files will be stored.
    """

    OWN_PATH: FilePath = Path(os.path.abspath(__file__))
    OWN_DIRECTORY: dirPath = OWN_PATH.parent
    OWN_DIRECTORY_DIRECTORY = OWN_DIRECTORY.parent
    
    
    def __init__(self, pipeline: Pipeline, data_dir_path: dirPath = Path(fr"{OWN_DIRECTORY_DIRECTORY}\data_pipeline\data"), logs_dir_path: dirPath = Path(fr"{OWN_DIRECTORY_DIRECTORY}\data_pipeline\logs")) -> None:
        """
        Initializes a Converter object with the given pipeline, data dir path, and logs dir path.

        Parameters:
            pipeline (Pipeline): The pipeline object representing the conversion flow.
            data_dir_path (dirPath, optional): The path to the dir containing the input data files. Defaults to the 'data' dir within the 'data_pipeline' directory.
            logs_dir_path (dirPath, optional): The path to the dir where the log files will be stored. Defaults to the 'logs' dir within the 'data_pipeline' directory.
        """
        # environment.set('musescoreDirectPNGPath', musescore_path)
        self.logs_dir_path = logs_dir_path
        self.data_dir_path = data_dir_path
        os.makedirs(self.logs_dir_path, exist_ok=True)
        os.makedirs(self.data_dir_path, exist_ok=True)
        
        self.pipeline = pipeline
        self.assign_data_dirs()  
        self.assign_log_dirs()


    def assign_log_dirs(self) -> None:
        """
        Assigns and creates the log dirs for each conversion route in the pipeline.

        The log dirs are created within the logs_dir_path, and their names are based on the conversion routes.
        Each conversion route is represented by a tuple of two PipelineStage instances, and the corresponding log dir
        is stored in the log_dir_map dictionary.
        """
        self.log_dir_map: dict[Tuple[PipelineStage, PipelineStage], dirPath] = {}
        for stage in self.pipeline:
            for child in stage.children:
                path = self.logs_dir_path.joinpath(f"{stage.name}_to_{child.name}")
                self.log_dir_map[(stage, child)] = path
                path.mkdir(exist_ok=True)


    def assign_data_dirs(self) -> None:
        """
        Assigns and creates the data dirs for each stage in the pipeline.

        The data dirs are created within the data_dir_path, and their names are based on the stage names.
        Each stage is represented by a PipelineStage instance, and the corresponding data dir
        is stored in the data_dir_map dictionary.
        """
        self.data_dir_map: dict[PipelineStage, dirPath] = {}
        for stage in self.pipeline:
            path = self.data_dir_path.joinpath(stage.name)
            self.data_dir_map[stage] = path
            path.mkdir(exist_ok=True)


    def batch_get_license(self, func: BatchConversionFunction) -> bool:
        """
        Asks the user for permission to proceed with a batch conversion, even if non-overwriting was specified.

        Parameters:
            func (BatchConversionFunction): The batch conversion function that will be executed.

        Returns:
            bool: True if the user agrees to proceed with the batch conversion, False otherwise.

        The function prompts the user to confirm whether they want to proceed with the batch conversion,
        even if non-overwriting was specified. The user's input is case-insensitive and must be either 'y' or 'n'.
        If the user enters 'y', the function returns True; otherwise, it returns False.
        """
        do_conversion = input(f"Next function will do a batch conversion({func}). Non-overwriting was specified, but cannot be guaranteed for batch mode.\nDo you want to proceed regardless? (y/n): ")
        return do_conversion.lower() == "y"
            

    def multi_stage_conversion(self, start_stage: PipelineStage, target_stage: PipelineStage, overwrite: bool = True, batch_if_possible: bool = True) -> None:
        """
        Performs a multi-stage conversion from the start stage to the target stage.

        Parameters:
            start_stage (PipelineStage): The starting stage of the conversion process.
            target_stage (PipelineStage): The target stage of the conversion process.
            overwrite (bool, optional): A flag indicating whether existing files should be overwritten. Defaults to True.
            batch_if_possible (bool, optional): A flag indicating whether batch conversion should be attempted if possible. Defaults to True.

        The function first finds the shortest conversion route from the start stage to the target stage.
        It then prints a message indicating the conversion process and iterates through each stage in the route.
        For each stage, it calls the single_stage_conversion method to perform the conversion.
        """
        
        route = self.pipeline.shortest_conversion_route(start_stage, target_stage)
         
        print(f"\nConverting from {start_stage.name} to {target_stage.name} via {[stage.name for stage in route]}...\n")

        current_start_stage = start_stage

        for current_target_stage in route[1:]:
            self.single_stage_conversion(current_start_stage, current_target_stage, overwrite, batch_if_possible)
            current_start_stage = current_target_stage
        
            
    def single_stage_conversion(self, start_stage: PipelineStage, target_stage: PipelineStage, overwrite: bool=True, batch_if_possible: bool = True) -> None: 
        """
        Performs a single-stage conversion from the start stage to the target stage.

        Parameters:
            start_stage (PipelineStage): The starting stage of the conversion process.
            target_stage (PipelineStage): The target stage of the conversion process.
            overwrite (bool, optional): A flag indicating whether existing files should be overwritten. Defaults to True.
            batch_if_possible (bool, optional): A flag indicating whether batch conversion should be attempted if possible. Defaults to True.

        Raises:
            ValueError: If the conversion from the start stage to the target stage is not possible.
        """
        if target_stage not in start_stage.children:
                raise ValueError(f"Cannot convert from stage: {start_stage.name} to stage: {target_stage.name}")
        
        conversion_function = start_stage.children[target_stage]     
        start_dir = self.data_dir_map[start_stage]
        target_dir = self.data_dir_map[target_stage]
        
        log = Log(self, start_stage, target_stage)

        
        if batch_if_possible and isinstance(conversion_function, BatchConversionFunction):
            batch_licenced = self.batch_get_license(conversion_function) if not overwrite else True
            if batch_licenced:
                print(f"Batch converting from {start_stage.name} to {target_stage.name}...\n")
                self.logged_batch_file_conversion(conversion_function, start_dir, target_dir, log, overwrite)
            
            else:
                print(f"Conversion from {start_stage.name} to {target_stage.name} aborted.\n")
            
        else:
            print(f"Single-file converting from {start_stage.name} to {target_stage.name}...\n")
            self.logged_single_file_conversion(conversion_function, start_dir, target_dir, log, overwrite, start_stage.extension)

        print(log.stats["num_successful"], "successful, successful / attempted")
        print(f"\n\nConversion from {start_stage.name} to {target_stage.name} completed. Log saved as {log.path.name} in {log.path.parent}.\n")


    def logged_single_file_conversion(self, conversion_function: _ConversionFunction, input_dir: dirPath, output_dir: dirPath, log: Log, overwrite: bool, extension: str) -> None:
        """
        Performs a single-file conversion from the start stage to the target stage, logs the outcome, and commits the log.

        Parameters:
            conversion_function (ConversionFunction): The conversion function to be applied to each input file.
            start_stage (PipelineStage): The starting stage of the conversion process.
            target_stage (PipelineStage): The target stage of the conversion process.
            log (Log): The log object to store the conversion outcomes.
            overwrite (bool): A flag indicating whether existing files should be overwritten.
        """

        for _file in input_dir.glob(f"*{extension}"):          
            input_file: FilePath = FilePath(_file)
            
            if isinstance(conversion_function, BatchConversionFunction):
                outcome = conversion_function(input_file, output_dir, do_batch=False, overwrite=overwrite)
            else:
                outcome = conversion_function(input_file, output_dir, overwrite=overwrite)
            log.log(outcome)

        log.commit()


    def logged_batch_file_conversion(self, conversion_function: BatchConversionFunction, input_dir: dirPath, output_dir: dirPath, log: Log, overwrite: bool) -> None:
        """
        Performs a batch file conversion from the start stage to the target stage, logs the outcome, and commits the log.

        Parameters:
            conversion_function (BatchConversionFunction): The batch conversion function to be applied to the input files.
            input_dir (dirPath): The dir containing the input files.
            output_dir (dirPath): The dir where the output files will be saved.
            log (Log): The log object to store the conversion outcomes.
            overwrite (bool): A flag indicating whether existing files should be overwritten.

        The function retrieves the input and output dirs based on the start and target stages.
        It then calls the batch conversion function with the input and output dirs, and logs the outcome.
        Finally, it commits the log.
        """
    
        outcome = conversion_function(input_dir, output_dir, do_batch=True, overwrite=overwrite)
        log.log(outcome)
        log.commit()


class Log():
    """
    A class responsible for logging the outcome of each conversion process.

    Attributes:
        converter (Converter): The Converter object that initiated the conversion process.
        start_stage (PipelineStage): The starting stage of the conversion process.
        target_stage (PipelineStage): The target stage of the conversion process.
        path (FilePath): The file path where the log will be saved.
        
        start_stage_dir_path (dirPath): The dir path where the input files for the conversion process are located.
        target_stage_dir_path (dirPath): The dir path where the output files for the conversion process will be saved.
        
        text (List[str]): A list to store the log messages.
        
        num_total_files (int): The total number of files to be converted.
        num_attempted (int): The total number of files attempted for conversion.
        num_skipped (int): The total number of files skipped for conversion.
        num_successful (int): The total number of successful conversions.
        num_errors (int): The total number of errors encountered during the conversion process.
        num_warned_successful (int): The total number of successful conversions with warnings.
        has_halted (bool): A flag indicating whether a halt has occurred during the conversion process.
        
    """
    
    class HaltError(Exception):
        """
        An exception raised when a halt occurs during the conversion process.
        """
        pass

    def __init__(self, converter: Converter, start_stage: PipelineStage, target_stage: PipelineStage) -> None:
        """
        Initializes a new instance of the Log class.

        Parameters:
            converter (Converter): The Converter object that initiated the conversion process.
            start_stage (PipelineStage): The starting stage of the conversion process.
            target_stage (PipelineStage): The target stage of the conversion process.

        The `__init__` method initializes the attributes of the `Log` class, such as the `converter`, `start_stage`, `target_stage`, log file path, and other counters for tracking the number of attempted, skipped, successful, error, and warned successful conversions. It also initializes the list `text` to store the log messages.
        """
        self.start_stage: PipelineStage = start_stage
        self.target_stage: PipelineStage = target_stage
        self.converter: Converter = converter

        self.path: FilePath = Path(self.converter.log_dir_map[(start_stage, target_stage)].joinpath(f"{start_stage.name}_to_{target_stage.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"))

        self.start_stage_dir_path: dirPath = self.converter.data_dir_map[start_stage]
        self.target_stage_dir_path: dirPath = self.converter.data_dir_map[target_stage]
        self.text: List[str] = [self.path.name + f"\n {self.start_stage_dir_path} -> {self.target_stage_dir_path}" + 2 * "\n"]

        self.num_total_files: int = len(list(self.converter.data_dir_map[start_stage].glob(f"*{start_stage.extension}")))
        self.num_attempted: int = 0
        self.num_skipped: int = 0 
        self.num_successful: int = 0 
        self.num_errors: int = 0
        self.num_warned_successful: int = 0
        self.has_halted: bool = False

    @property
    def index(self) -> str:
        """
        Returns a string representing the current progress index in the conversion process.

        The index is formatted as "[attempted/total]", where "attempted" is the number of files attempted to be converted,
        and "total" is the total number of files in the starting stage dir.

        Returns:
            str: The progress index string.
        """
        return f"[{self.num_attempted}/{self.num_total_files}]"

    def log_skip(self, outcome: ConversionOutcome) -> None:
        """
        Logs a skipped conversion outcome.

        Parameters:
            outcome (ConversionOutcome): The outcome of the conversion process.

        The function increments the number of skipped conversions, appends a message to the log text,
        and prints a progress indicator. If the outcome does not contain an error message, it adds a reason
        for skipping the conversion.
        """
        self.num_skipped += 1
        
        if not outcome.error_message:
            outcome.error_message = f"Output: {f'{outcome.output_files[0].name} already exists' if len(outcome.output_files) == 1 else f'Files {outcome.output_files[0].name}... already exist'} and shall not be overwritten\n"

        self.text.append(f"{self.index} [SKIPPED] {datetime.datetime.now()} - Input: {outcome.input_file.name}; \n" + outcome.error_message)
        
        print(f"{self.index} [SKIPPED]", end="\r")

    def log_success(self, outcome: ConversionOutcome) -> None:
        """
        Logs a successful conversion outcome.

        Parameters:
            outcome (ConversionOutcome): The outcome of the conversion process.
            This object contains information about the input file, output files,
            warning messages, and error message (if any).

        The function increments the number of successful conversions, appends a message to the log text,
        and prints a progress indicator. If the outcome contains warning messages, it adds them to the log text.
        """
        self.num_successful += 1

        self.text.append(f"{self.index} [SUCCESS] {datetime.datetime.now()} - Input: {outcome.input_file.name}; Output: {outcome.output_files[0].name if len(outcome.output_files) == 1 else f'{len(outcome.output_files)} Files ({outcome.output_files[0].name})...'}\n") 
    
        if outcome.warning_messages:
            self.num_warned_successful += 1
            for warning_message in outcome.warning_messages:
                self.text.append(f"\t[WARNING] {warning_message}\n")
        
        print(f"{self.index} [SUCCESS]", end="\r")

    def log_error(self, outcome: ConversionOutcome) -> None:
        """
        Logs an error during the conversion process.

        Parameters:
            outcome (ConversionOutcome): The outcome of the conversion process.
            This object contains information about the input file and the error message.

        The function increments the number of errors, appends a message to the log text,
        and prints a progress indicator. The message includes the current progress index,
        the current date and time, the name of the input file, and the error message.
        """
        self.num_errors += 1
             
        self.text.append(f"{self.index} [ERROR] {datetime.datetime.now()} - Input: {outcome.input_file.name}\n" + 
                        f"\tError: {outcome.error_message}\n")
        
        print(f"{self.index} [ERROR]", end="\r")

    def log_halt(self, outcome: ConversionOutcome) -> None:
        """
        Logs a halt during the conversion process.

        Parameters:
            outcome (ConversionOutcome): The outcome of the conversion process.
            This object contains information about the input file and the error message.

        The function increments the number of errors, appends a message to the log text,
        and prints a progress indicator. The message includes the current progress index,
        the current date and time, the name of the input file, and the error message.
        It then commits the log and raises a `Log.HaltError` exception.
        """
        self.num_errors += 1
        
        self.has_halted = True
        
        self.text.append(f"{self.index} [HALT] {datetime.datetime.now()} - ON {outcome.input_file.name}\n"+ f"\tError: {outcome.error_message}\n")
        
        self.commit()
        
        raise Log.HaltError(f"\nSingle-file conversion from {self.start_stage.name} to {self.target_stage.name} halted. Potential following conversions aborted.\nCritical failure since conversion of {self.index, outcome.input_file.name}.\n\tError:" + outcome.error_message)
    

    def log(self, outcomes: List[ConversionOutcome]) -> None:
        """
        Logs the outcomes of each conversion process.

        Parameters:
            outcomes (List[ConversionOutcome]): A list of ConversionOutcome objects, each representing the outcome of a single conversion process.

        The function iterates through each outcome in the list. It increments the number of attempted conversions,
        and then calls the appropriate logging function based on the outcome's properties.
        If the outcome is skipped, it calls the log_skip method.
        If the outcome is successful, it calls the log_success method.
        If the outcome is not successful and contains a halt, it calls the log_halt method.
        If the outcome is not successful and does not contain a halt, it calls the log_error method.
        """
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
    def stats(self) -> dict[str, Any]:
        """
        Returns a dictionary containing various statistics related to the conversion process.

        Returns:
            dict: A dictionary with the following keys and their corresponding values:
            - "num_total": The total number of files in the starting stage dir.
            - "num_attempted": The number of files attempted to be converted.
            - "num_successful": A tuple containing the number of successful conversions and the percentage of successful conversions.
            - "num_skipped": A tuple containing the number of skipped conversions and the percentage of skipped conversions.
            - "num_errors": A tuple containing the number of errors during the conversion process and the percentage of errors.
            - "num_warned_from_successful": A tuple containing the number of warned successful conversions and the percentage of warned successful conversions.
            - "has_halted": A boolean indicating whether the conversion process has been halted.
        """
        return {
        "num_total": self.num_total_files,
        "num_attempted": self.num_attempted,
        "num_successful": (self.num_successful, self.num_successful / self.num_attempted if self.num_attempted else 0.0),
        "num_skipped": (self.num_skipped, self.num_skipped / self.num_attempted if self.num_attempted else 0.0),
        "num_errors": (self.num_errors, self.num_errors / self.num_attempted if self.num_attempted else 0.0),
        "num_warned_from_successful": (self.num_warned_successful, self.num_warned_successful / self.num_successful if self.num_successful else 0.0),
        "has_halted": self.has_halted
        }
    
    def insert_evaluation(self) -> None:
        """
        Inserts evaluation statistics at the beginning of the log.

        The function calculates and inserts various statistics related to the conversion process,
        such as the total number of files, attempted files, successful files, skipped files, errors,
        and warned successful conversions. The statistics are formatted as key-value pairs and added
        to the log text.

        The statistics include:
        - "num_total": The total number of files in the starting stage dir.
        - "num_attempted": The number of files attempted to be converted.
        - "num_successful": A tuple containing the number of successful conversions and the percentage of successful conversions.
        - "num_skipped": A tuple containing the number of skipped conversions and the percentage of skipped conversions.
        - "num_errors": A tuple containing the number of errors during the conversion process and the percentage of errors.
        - "num_warned_from_successful": A tuple containing the number of warned successful conversions and the percentage of warned successful conversions.
        """
        self.text[1:1] = "\n"
        self.text[1:1] = [
            f"{key}: {value[0]} ({value[1] * 100:.2f}%)\n" if isinstance(value, tuple) else f"{key}: {value}\n"
            for key, value in self.stats.items()
        ]
        
    
    def commit(self) -> None:
        """
        Writes the log to a file and prints the evaluation statistics.

        This method first inserts the evaluation statistics at the beginning of the log using the
        `insert_evaluation` method. Then, it opens the log file in write mode and writes the log text
        to the file.

        Parameters:
        None

        Returns:
        None
        """
        self.insert_evaluation()
        with self.path.open(mode="w", encoding="utf-8") as file:
            file.write("".join(self.text))


if __name__ == "__main__":
    pipeline = construct_music_pipeline()
    converter = Converter(pipeline)
    converter.multi_stage_conversion(converter.pipeline["midi_in"], converter.pipeline["tokens_in"], overwrite=True, batch_if_possible=True)
