from __future__ import annotations
import os, datetime, constants as constants
from typing import * # type: ignore
from data_pipeline_scripts.pipeline import *
from pathlib import Path
from data_pipeline_scripts.conversion_func_infrastructure import *
from data_pipeline_scripts.conversion_func_infrastructure import _ConversionFunction


class Converter():
    """
    Manages the conversion process between different music file formats using a pipeline architecture.

    The Converter class orchestrates file conversions through a defined pipeline, handling data organization,
    logging, temporary file management, and batch/single-file processing. It maintains separate directories
    for input data, temporary files, error files, and conversion logs.

    Class Attributes:
        OWN_PATH (FilePath): Absolute path to this Converter class file.
        OWN_DIRECTORY (DirPath): Directory containing this Converter class file.
        OWN_DIRECTORY_DIRECTORY (DirPath): Parent directory of OWN_DIRECTORY.

    Attributes:
        pipeline (Pipeline): The conversion pipeline defining available format transformations.
        pipeline_dir_path (DirPath): Root directory for all pipeline-related files and folders.
        data_dir_path (DirPath): Directory containing input data files organized by stage.
        logs_dir_path (DirPath): Directory where conversion log files are stored.
        temp_dir_path (DirPath): Directory for successfully processed files.
        error_temp_dir_path (DirPath): Directory for files that failed processing.
        data_dir_map (Dict[PipelineStage, DirPath]): Maps each pipeline stage to its data directory.
        temp_dir_map (Dict[PipelineStage, DirPath]): Maps each pipeline stage to its temp directory.
        error_temp_dir_map (Dict[PipelineStage, DirPath]): Maps each pipeline stage to its error temp directory.
        log_dir_map (Dict[Tuple[PipelineStage, PipelineStage], DirPath]): Maps conversion routes to log directories.

    Example:
        >>> pipeline = construct_music_pipeline(tokenizer, pdf_preprocess=True)
        >>> converter = Converter(pipeline)
        >>> converter.single_stage_conversion("pdf_in", "mxl_in", overwrite=True)
        >>> converter.multi_stage_conversion("pdf_in", "tokens_in")
    """

    OWN_PATH: FilePath = Path(os.path.abspath(__file__))
    OWN_DIRECTORY: DirPath = OWN_PATH.parent
    OWN_DIRECTORY_DIRECTORY = OWN_DIRECTORY.parent
    
    
    def __init__(self, pipeline: Pipeline, pipeline_dir_path: DirPath = Path(fr"{OWN_DIRECTORY_DIRECTORY}")) -> None:
        """
        Initializes a Converter with the specified pipeline and directory structure.

        Creates the complete directory structure for pipeline operations, including data directories
        for each stage, temporary directories for file management, and log directories for tracking
        conversion outcomes.

        Args:
            pipeline (Pipeline): The conversion pipeline defining available format transformations.
            pipeline_dir_path (DirPath, optional): Root directory for pipeline operations. 
                Defaults to the parent directory of this file's location.

        Side Effects:
            Creates directory structure:
            - pipeline_dir_path/data/{stage_name}/ for each pipeline stage
            - pipeline_dir_path/temp/{stage_name}/ for each pipeline stage  
            - pipeline_dir_path/error_temp/{stage_name}/ for each pipeline stage
            - pipeline_dir_path/logs/{stage1}_to_{stage2}/ for each conversion route
        """
        # environment.set('musescoreDirectPNGPath', musescore_path)
        self.pipeline_dir_path: DirPath = pipeline_dir_path / constants.data_pipeline_constants.CONVERTER_PIPELINE_DIR_NAME
        self.logs_dir_path: DirPath = self.pipeline_dir_path / constants.data_pipeline_constants.CONVERTER_LOGS_DIR_NAME
        self.data_dir_path: DirPath = self.pipeline_dir_path / constants.data_pipeline_constants.CONVERTER_DATA_DIR_DEFAULT_NAME
        self.temp_dir_path: DirPath = self.pipeline_dir_path / constants.data_pipeline_constants.CONVERTER_TEMP_DIR_NAME
        self.error_temp_dir_path: DirPath = self.pipeline_dir_path / constants.data_pipeline_constants.CONVERTER_ERROR_TEMP_DIR_NAME

        self.pipeline = pipeline
        self.assign_log_dirs()

        self.data_dir_map: dict[PipelineStage, DirPath] = self.assign_dirs(self.data_dir_path)
        self.temp_dir_map: dict[PipelineStage, DirPath] = self.assign_dirs(self.temp_dir_path)
        self.error_temp_dir_map: dict[PipelineStage, DirPath] = self.assign_dirs(self.error_temp_dir_path)
        

    def assign_log_dirs(self) -> None:
        """
        Creates log directories for each possible conversion route in the pipeline.

        Generates a separate log directory for every direct conversion path between pipeline stages.
        Directory names follow the pattern: "{source_stage}_to_{target_stage}".

        Side Effects:
            - Creates logs_dir_path if it doesn't exist
            - Creates subdirectory for each conversion route
            - Populates log_dir_map with stage tuple to directory mappings

        Example:
            For a pipeline with pdf_in → mxl_in → midi_in stages, creates:
            - logs/pdf_in_to_mxl_in/
            - logs/mxl_in_to_midi_in/
        """
        self.logs_dir_path.mkdir(parents=True, exist_ok=True)
        self.log_dir_map: dict[Tuple[PipelineStage, PipelineStage], DirPath] = {}
        for stage in self.pipeline:
            for child in stage.children:
                path = self.logs_dir_path / f"{stage.name}_to_{child.name}"
                self.log_dir_map[(stage, child)] = path
                path.mkdir(exist_ok=True)

    def assign_dirs(self, super_dir_path: DirPath) -> dict[PipelineStage, DirPath]:
        """
        Creates and maps directories for each pipeline stage under the specified parent directory.

        Args:
            super_dir_path (DirPath): Parent directory under which stage subdirectories will be created.

        Returns:
            dict[PipelineStage, DirPath]: Mapping from each pipeline stage to its created directory.

        Side Effects:
            - Creates super_dir_path if it doesn't exist
            - Creates subdirectory named after each stage (stage.name)

        Example:
            >>> stage_dirs = converter.assign_dirs(Path("./data"))
            # Creates: ./data/pdf_in/, ./data/mxl_in/, ./data/midi_in/, etc.
        """
        super_dir_path.mkdir(parents=True, exist_ok=True)
        dir_map = {}
        for stage in self.pipeline:
            path = super_dir_path / stage.name
            dir_map[stage] = path
            path.mkdir(exist_ok=True)
        return dir_map

    @staticmethod
    def move_file(src: FilePath, dest: DirPath) -> None:
        """
        Moves a file or directory from source to destination, handling conflicts and directory trees.

        Supports moving both individual files and entire directory trees. For directories,
        recursively moves all contents. Overwrites existing files at destination.

        Args:
            src (FilePath): Source file or directory to move.
            dest (DirPath): Destination directory where the file/directory will be moved.

        Side Effects:
            - Creates destination directory if it doesn't exist
            - Removes existing files with same name at destination
            - For directories: recursively moves all contents, then removes empty source directory

        Example:
            >>> Converter.move_file(Path("data/song.pdf"), Path("temp/pdf_in/"))
            # Moves to: temp/pdf_in/song.pdf
        """

        dest.mkdir(parents=True, exist_ok=True)
        
        if src.is_dir():
            if not (temp := dest / src.name).exists():
                temp.mkdir(parents=True, exist_ok=False)
            
            for item in src.glob("*"):
                Converter.move_file(item, temp)
            
            src.rmdir()
        else:
            if (temp := dest / src.name).exists():
                temp.unlink()
            
            src.rename(temp)


    def outcome_move_inputs_to_temp(self, outcome: ConversionOutcome, stage: PipelineStage, 
                                   move_succ: bool, move_err: bool) -> None:
        """
        Conditionally moves conversion input files to appropriate temporary directories.

        Moves the input file and any associated extra files (like metadata) to either the regular
        temp directory (for successful conversions) or error temp directory (for failed conversions)
        based on the outcome and specified flags.

        Args:
            outcome (ConversionOutcome): Result of a conversion operation containing input/output info.
            stage (PipelineStage): Pipeline stage from which to move the input files.
            move_succ (bool): Whether to move files when conversion was successful.
            move_err (bool): Whether to move files when conversion failed.

        Side Effects:
            - Moves input file to temp_dir_map[stage] if successful and move_succ is True
            - Moves input file to error_temp_dir_map[stage] if failed and move_err is True
            - Also moves any extra files defined in stage.extra_dirs
            - Skipped conversions are never moved

        Example:
            >>> # Move successful conversions to temp, errors to error_temp
            >>> converter.outcome_move_inputs_to_temp(outcome, pdf_stage, True, True)
        """
        if not outcome.skipped:
            if outcome.successful and move_succ:
                self.move_file(outcome.input_file, self.temp_dir_map[stage])
                for folder, extension in stage.extra_dirs:
                    if (temp := self.data_dir_map[stage] / folder / (outcome.input_file.stem + extension)).exists():
                        self.move_file(temp, self.temp_dir_map[stage] / folder)
            
            elif not outcome.successful and move_err:
                self.move_file(outcome.input_file, self.error_temp_dir_map[stage])
                for folder, extension in stage.extra_dirs:
                    if (temp := self.data_dir_map[stage] / folder / (outcome.input_file.stem + extension)).exists():
                        self.move_file(temp, self.error_temp_dir_map[stage] / folder)

    def move_stage_data_to_temp(self, *stages: str | PipelineStage, to_error_temp: bool = False) -> None:
        """
        Moves all data files from specified stages to their temporary directories.

        Bulk operation to move all files from stage data directories to either regular temp
        or error temp directories. Useful for clearing processed data or isolating problematic files.

        Args:
            to_error_temp (bool): If True, moves to error temp directories; if False, to regular temp.
            *stages (str | PipelineStage): Variable number of stages to process. Can be stage names or objects.

        Side Effects:
            - Moves all files from each stage's data directory to its temp directory
            - Prints confirmation message for each stage processed

        Raises:
            ValueError: If any stage name/object is invalid.

        Example:
            >>> # Move all PDF and MXL files to error temp
            >>> converter.move_stage_data_to_temp(True, "pdf_in", "mxl_in")
        """
        for s in self.pipeline.to_stage(*stages):
            if not isinstance(s, PipelineStage):
                raise ValueError(f"Stage {s} is not a valid PipelineStage.")
            
            for file in self.data_dir_map[s].glob("*"):
                if to_error_temp:
                    Converter.move_file(file, self.error_temp_dir_map[s])
                else:
                    Converter.move_file(file, self.temp_dir_map[s])
            print(f"Moved data from {s.name} to {'error' if to_error_temp else ''} temp dir {self.temp_dir_map[s].name}.\n")

    def load_stage_data_from_temp(self, from_error_temp: bool, *stages: str | PipelineStage) -> None:
        """
        Restores data files from temporary directories back to stage data directories.

        Bulk operation to move files from temp directories back to their original data directories.
        Useful for retrying conversions or restoring accidentally moved files.

        Args:
            from_error_temp (bool): If True, loads from error temp; if False, from regular temp.
            *stages (str | PipelineStage): Variable number of stages to process. Can be stage names or objects.

        Side Effects:
            - Moves all files from each stage's temp directory back to its data directory
            - Prints confirmation message for each stage processed

        Raises:
            ValueError: If any stage name/object is invalid.

        Example:
            >>> # Restore PDF files from error temp for retry
            >>> converter.load_stage_data_from_temp(True, "pdf_in")
        """
        for s in self.pipeline.to_stage(*stages):
            if not isinstance(s, PipelineStage):
                raise ValueError(f"Stage {s} is not a valid PipelineStage.")
            
            for file in (self.error_temp_dir_map[s] if from_error_temp else self.temp_dir_map[s]).glob("*"):
                Converter.move_file(file, self.data_dir_map[s])
            print(f"Loaded data for {s.name} from {'error' if from_error_temp else ''} temp dir {self.temp_dir_map[s].name}.\n")
            

    def multi_stage_conversion(self, start_stage: str | PipelineStage, target_stage: str | PipelineStage, 
                              overwrite: bool = True, batch_if_possible: bool = False, 
                              move_successful_inputs_to_temp: bool = False, move_error_inputs_to_temp: bool = False) -> None:
        """
        Performs a complete multi-step conversion between two potentially distant pipeline stages.

        Automatically finds the shortest conversion route between start and target stages, then
        executes each conversion step in sequence. Useful for complex transformations like
        PDF → MXL → MIDI → Tokens that require multiple intermediate steps.

        Args:
            start_stage (str | PipelineStage): Starting stage for the conversion (source format).
            target_stage (str | PipelineStage): Target stage for the conversion (destination format).
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to True.
            batch_if_possible (bool, optional): Use batch conversion when available. Defaults to False.
            move_successful_inputs_to_temp (bool, optional): Move successful inputs to temp. Defaults to False.
            move_error_inputs_to_temp (bool, optional): Move failed inputs to error temp. Defaults to False.

        Side Effects:
            - Executes each step in the conversion route via single_stage_conversion()
            - Creates log files for each conversion step
            - May move files to temp directories based on flags

        Example:
            >>> # Convert PDF sheet music to training tokens
            >>> converter.multi_stage_conversion("pdf_in", "tokens_in")
            # Executes: PDF → MXL → MIDI → Tokens (3 conversion steps)
        """

        start_stage = self.pipeline.to_stage(start_stage)[0] # this is also a typechecker
        target_stage = self.pipeline.to_stage(target_stage)[0]

        route = self.pipeline.shortest_conversion_route(start_stage, target_stage)
         
        print(f"\nConverting from {start_stage.name} to {target_stage.name} via {[stage.name for stage in route]}...\n")

        current_start_stage = start_stage

        for current_target_stage in route[1:]:
            self.single_stage_conversion(current_start_stage, current_target_stage, overwrite, batch_if_possible, move_successful_inputs_to_temp, move_error_inputs_to_temp)
            current_start_stage = current_target_stage


    def single_stage_conversion(self, start_stage: str | PipelineStage, target_stage: str | PipelineStage, 
                               overwrite: bool = True, batch_if_possible: bool = False, 
                               move_successful_inputs_to_temp: bool = False, move_error_inputs_to_temp: bool = False) -> None:
        """
        Performs a direct single-step conversion between two adjacent pipeline stages.

        Executes conversion for all files in the start stage's data directory to the target stage's
        data directory. Automatically chooses between batch and single-file processing based on
        the conversion function type and user preferences.

        Args:
            start_stage (str | PipelineStage): Source stage containing input files.
            target_stage (str | PipelineStage): Target stage for output files.
            overwrite (bool, optional): Whether to overwrite existing output files. Defaults to True.
            batch_if_possible (bool, optional): Use batch conversion when available. Defaults to True.
            move_successful_inputs_to_temp (bool, optional): Move successful inputs to temp. Defaults to False.
            move_error_inputs_to_temp (bool, optional): Move failed inputs to error temp. Defaults to False.

        Raises:
            ValueError: If target_stage is not a direct child of start_stage.
            RuntimeError: If batch conversion is aborted by user.

        Side Effects:
            - Processes all files matching start_stage.extension in start stage data directory
            - Creates comprehensive log file with conversion outcomes
            - May move processed files to temp directories based on flags

        Example:
            >>> # Convert all PDF files to MXL format
            >>> converter.single_stage_conversion("pdf_in", "mxl_in", overwrite=True)
        """

        start_stage = self.pipeline.to_stage(start_stage)[0] # this is also a typechecker
        target_stage = self.pipeline.to_stage(target_stage)[0]

        if target_stage not in start_stage.children:
                raise ValueError(f"Cannot convert from stage: {start_stage.name} to stage: {target_stage.name}")
        
        conversion_function = start_stage.children[target_stage]     
        
        log = Log(self, start_stage, target_stage)

        
        if batch_if_possible and isinstance(conversion_function, BatchConversionFunction):
            batch_licenced = input(f"The next function will do a batch conversion({conversion_function}). Non-overwriting was specified, but cannot be guaranteed for batch mode.\nDo you want to proceed regardless? (y/n): ") == "y"
            
            if not batch_licenced:
                raise RuntimeError(f"Conversion from {start_stage.name} to {target_stage.name} aborted.\n")
        
        self.logged_single_file_conversion(conversion_function, start_stage, target_stage, log, overwrite, start_stage.extension, move_successful_inputs_to_temp, move_error_inputs_to_temp)

        print(log.stats["num_successful"], "successful, successful / attempted")
        print(f"\n\nConversion from {start_stage.name} to {target_stage.name} completed. Log saved as {log.path.name} in {log.path.parent}.\n")


    def logged_single_file_conversion(self, conversion_function: _ConversionFunction, start_stage: PipelineStage, 
                                     target_stage: PipelineStage, log: Log, overwrite: bool, extension: str, 
                                     move_successful_inputs_to_temp: bool, move_error_inputs_to_temp: bool) -> None:
        """
        Executes single-file conversion for each file in the input directory with comprehensive logging.

        Low-level conversion method that processes each file individually, logs outcomes, and handles
        temporary file movement. Supports both regular and batch conversion functions operating in
        single-file mode.

        Args:
            conversion_function (_ConversionFunction): Function to perform the actual conversion.
            start_stage (PipelineStage): Source stage containing input files.
            target_stage (PipelineStage): Target stage for output files.
            log (Log): Log object to record conversion outcomes.
            overwrite (bool): Whether to overwrite existing output files.
            extension (str): File extension filter for input files.
            move_successful_inputs_to_temp (bool): Move successful inputs to temp directory.
            move_error_inputs_to_temp (bool): Move failed inputs to error temp directory.

        Side Effects:
            - Processes each file matching the extension in start stage directory
            - Logs each conversion outcome (success, error, skip)
            - Commits log entries to file
            - Conditionally moves input files to temp directories
        """

        input_dir: DirPath = self.data_dir_map[start_stage]
        output_dir: DirPath = self.data_dir_map[target_stage]
        print(f"Single-file converting from {start_stage.name} to {target_stage.name}...\n")
        
        
        for _file in input_dir.glob(f"*{extension}"):
            input_file: FilePath = FilePath(_file)
            
            if isinstance(conversion_function, BatchConversionFunction):
                outcome = conversion_function(input_file, output_dir, do_batch=False, overwrite=overwrite)
            else:
                outcome = conversion_function(input_file, output_dir, overwrite=overwrite)
            log.log(outcome)

            if move_successful_inputs_to_temp:
                self.outcome_move_inputs_to_temp(outcome, start_stage, move_successful_inputs_to_temp, move_error_inputs_to_temp)

        log.commit()


    def logged_batch_file_conversion(self, conversion_function: BatchConversionFunction, start_stage: PipelineStage, 
                                    target_stage: PipelineStage, log: Log, overwrite: bool, 
                                    move_successful_inputs_to_temp: bool, move_error_inputs_to_temp: bool) -> None:
        """
        Executes batch conversion for all files in the input directory with comprehensive logging.

        High-performance conversion method that processes all files in bulk, logs outcomes, and commits
        the log. Supports moving of successful and error files to respective temp directories.

        Args:
            conversion_function (BatchConversionFunction): The batch conversion function to be applied to the input files.
            start_stage (PipelineStage): The starting stage of the conversion process.
            target_stage (PipelineStage): The target stage of the conversion process.
            log (Log): The log object to store the conversion outcomes.
            overwrite (bool): A flag indicating whether existing files should be overwritten.
            move_successful_inputs_to_temp (bool): Move successful inputs to temp directory.
            move_error_inputs_to_temp (bool): Move failed inputs to error temp directory.

        Side Effects:
            - Processes all files in input_dir in one batch
            - Logs the outcome of the batch conversion
            - Commits the log
            - Conditionally moves input files to temp directories
        """
        
        input_dir: DirPath = self.data_dir_map[start_stage]
        output_dir: DirPath = self.data_dir_map[target_stage]

        print(f"Batch converting from {start_stage.name} to {target_stage.name}...\n")

        outcomes = conversion_function(input_dir, output_dir, do_batch=True, overwrite=overwrite)
        log.log(outcomes)
        log.commit()

         
        if move_successful_inputs_to_temp:
            for outcome in outcomes:
                self.outcome_move_inputs_to_temp(outcome, start_stage, move_successful_inputs_to_temp, move_error_inputs_to_temp)

class Log():
    """
    Manages comprehensive logging for music file conversion processes with real-time statistics.

    The Log class tracks conversion outcomes in real-time, provides detailed statistics, and generates
    structured log files for each conversion operation. It supports various outcome types including
    successful conversions, errors, warnings, skipped files, and critical halts.

    Attributes:
        converter (Converter): The Converter instance that initiated this conversion process.
        start_stage (PipelineStage): Source stage of the conversion.
        target_stage (PipelineStage): Target stage of the conversion.
        path (FilePath): Full path to the log file for this conversion session.
        start_stage_dir_path (DirPath): Directory containing input files for conversion.
        target_stage_dir_path (DirPath): Directory where output files are saved.
        header (str): Log file header containing session information.
        text (List[str]): Accumulated log messages for this session.
        
        num_total_files (int): Total number of files available for conversion.
        num_attempted (int): Number of files that conversion was attempted on.
        num_skipped (int): Number of files skipped (usually due to existing outputs).
        num_successful (int): Number of files successfully converted.
        num_errors (int): Number of files that failed conversion.
        num_warned_successful (int): Number of successful conversions that generated warnings.
        has_halted (bool): Whether a critical error caused the conversion process to halt.

    Example:
        >>> log = Log(converter, pdf_stage, mxl_stage)
        >>> log.log(conversion_outcome)
        >>> log.commit()  # Finalizes log with statistics
        >>> print(log.stats)  # View conversion statistics
    """
    
    class HaltError(Exception):
        """
        Exception raised when a critical error halts the conversion process.
        
        This exception indicates that a conversion failure was severe enough to stop
        the entire conversion operation, preventing further file processing.
        """

    def __init__(self, converter: Converter, start_stage: PipelineStage, target_stage: PipelineStage) -> None:
        """
        Initializes a new conversion log session.

        Creates a timestamped log file, calculates the total number of files to process,
        and sets up tracking counters for various conversion outcomes.

        Args:
            converter (Converter): The Converter managing this conversion process.
            start_stage (PipelineStage): Source stage containing input files.
            target_stage (PipelineStage): Target stage for output files.

        Side Effects:
            - Creates a timestamped log file in the appropriate log directory
            - Writes initial header information to the log file
            - Counts total files available for conversion in the start stage
        """

        self.start_stage: PipelineStage = start_stage
        self.target_stage: PipelineStage = target_stage
        self.converter: Converter = converter

        self.path: FilePath = Path(self.converter.log_dir_map[(start_stage, target_stage)].joinpath(f"{start_stage.name}_to_{target_stage.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{constants.LOG_EXTENSION}"))


        self.start_stage_dir_path: DirPath = self.converter.data_dir_map[start_stage]
        self.target_stage_dir_path: DirPath = self.converter.data_dir_map[target_stage]
        
        self.header: str = self.path.name + f"\n {self.start_stage_dir_path} -> {self.target_stage_dir_path}" + 2 * "\n"

        self.text: List[str] = [self.header]

        with self.path.open(mode="w", encoding="utf-8") as file:
            file.write(self.text[0])

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
        Returns the current progress indicator for log entries.

        Provides a standardized progress format showing current position in the conversion process.

        Returns:
            str: Progress indicator in format "[attempted/total]".

        Example:
            >>> log.index
            "[15/127]"  # 15 files attempted out of 127 total
        """
        return f"[{self.num_attempted}/{self.num_total_files}]"
    
    
    def log_skip(self, outcome: ConversionOutcome) -> None:
        """
        Records a skipped conversion outcome.

        Logs when a file conversion is skipped, typically because the output file already exists
        and overwrite is disabled. Updates skip counters and provides default reason if none given.

        Args:
            outcome (ConversionOutcome): The conversion outcome containing skip information.

        Side Effects:
            - Increments num_skipped counter
            - Adds timestamped skip entry to log
            - Prints progress indicator to console
            - Adds default skip reason if outcome.error_message is empty
            - Immediately writes log entry to file
        """
        self.num_skipped += 1
        
        if not outcome.error_message:
            outcome.error_message = f"Output: {f'{outcome.output_files[0].name} already exists' if len(outcome.output_files) == 1 else f'Files {outcome.output_files[0].name}... already exist'} and shall not be overwritten\n"


        self.text.append(f"{self.index} [SKIPPED] {datetime.datetime.now()} - Input: {outcome.input_file.name}; \n" + outcome.error_message)
        
        print(f"{self.index} [SKIPPED]", end="\r")

    def log_success(self, outcome: ConversionOutcome) -> None:
        """
        Records a successful conversion outcome.

        Logs successful file conversions, including any warning messages that occurred during
        the process. Tracks both clean successes and warned successes separately.

        Args:
            outcome (ConversionOutcome): The conversion outcome containing success information.

        Side Effects:
            - Increments num_successful counter
            - Increments num_warned_successful if warnings present
            - Adds timestamped success entry to log with output file information
            - Appends any warning messages to the log entry
            - Prints progress indicator to console
            - Immediately writes log entry to file
        """
        self.num_successful += 1

        self.text.append(f"{self.index} [SUCCESS] {datetime.datetime.now()} - Input: {outcome.input_file.name}; Output: {outcome.output_files[0].name if len(outcome.output_files) == 1 else f'{len(outcome.output_files)} Files ({outcome.output_files[0].name})...'}\n") 
    
        if outcome.warning_messages:
            self.num_warned_successful += 1
            for warning_message in outcome.warning_messages:
                self.text[-1] += f"\t[WARNING] {warning_message}\n"


        print(f"{self.index} [SUCCESS]", end="\r")

    def log_error(self, outcome: ConversionOutcome) -> None:
        """
        Records a conversion error outcome.

        Logs conversion failures with detailed error information for debugging purposes.

        Args:
            outcome (ConversionOutcome): The conversion outcome containing error information.

        Side Effects:
            - Increments num_errors counter
            - Adds timestamped error entry to log with error details
            - Prints progress indicator to console
            - Immediately writes log entry to file
        """

        self.num_errors += 1
             
        self.text.append(f"{self.index} [ERROR] {datetime.datetime.now()} - Input: {outcome.input_file.name}\n" + 
                        f"\tError: {outcome.error_message}\n")
        
        print(f"{self.index} [ERROR]", end="\r")

    def log_halt(self, outcome: ConversionOutcome) -> None:
        """
        Records a critical halt and immediately stops the conversion process.

        Logs critical errors that require immediate termination of the conversion process.
        Automatically commits the log and raises a HaltError exception.

        Args:
            outcome (ConversionOutcome): The conversion outcome that caused the halt.

        Raises:
            Log.HaltError: Always raised after logging the halt condition.

        Side Effects:
            - Increments num_errors counter
            - Sets has_halted flag to True
            - Adds timestamped halt entry to log
            - Immediately commits the log to file
            - Terminates the conversion process via exception
        """
        self.num_errors += 1
        
        self.has_halted = True
        
        self.text.append(f"{self.index} [HALT] {datetime.datetime.now()} - ON {outcome.input_file.name}\n"+ f"\tError: {outcome.error_message}\n")
        
        self.commit()
        
        raise Log.HaltError(f"\nSingle-file conversion from {self.start_stage.name} to {self.target_stage.name} halted. Potential following conversions aborted.\nCritical failure since conversion of {self.index, outcome.input_file.name}.\n\tError:" + outcome.error_message)


    def log(self, outcomes: List[ConversionOutcome] | ConversionOutcome) -> None:
        """
        Processes and logs one or more conversion outcomes.

        Central logging method that determines the appropriate logging action based on each
        outcome's properties. Handles both single outcomes and lists of outcomes from batch operations.

        Args:
            outcomes (List[ConversionOutcome] | ConversionOutcome): Single outcome or list of outcomes to log.

        Side Effects:
            - Increments num_attempted for each outcome processed
            - Calls appropriate logging method based on outcome properties:
              * log_skip() for skipped conversions
              * log_success() for successful conversions  
              * log_halt() for critical errors (may raise HaltError)
              * log_error() for regular errors
            - Each log entry is immediately written to file

        Example:
            >>> # Log single outcome
            >>> log.log(single_outcome)
            >>> 
            >>> # Log batch outcomes
            >>> log.log([outcome1, outcome2, outcome3])
        """
        if isinstance(outcomes, ConversionOutcome):
            outcomes = [outcomes]
        
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

            with self.path.open(mode="a", encoding="utf-8") as file:
                file.write(self.text[-1])

    @property
    def stats(self) -> dict[str, Any]:
        """
        Returns comprehensive conversion statistics.

        Provides detailed metrics about the conversion process including counts and percentages
        for all outcome types.

        Returns:
            dict[str, Any]: Statistics dictionary with keys:
                - "num_total": Total files available for conversion
                - "num_attempted": Files that conversion was attempted on
                - "num_successful": (count, percentage) of successful conversions
                - "num_skipped": (count, percentage) of skipped files
                - "num_errors": (count, percentage) of failed conversions
                - "num_warned_from_successful": (count, percentage) of warned successes
                - "has_halted": Boolean indicating if process was halted

        Example:
            >>> stats = log.stats
            >>> print(f"Success rate: {stats['num_successful'][1]*100:.1f}%")
            >>> print(f"Total errors: {stats['num_errors'][0]}")
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


    def commit(self) -> None:
        """
        Finalizes the log file with comprehensive statistics and summary information.

        Inserts detailed statistics at the beginning of the log file and writes the complete
        log to disk. Should be called at the end of each conversion session.

        Side Effects:
            - Calculates and formats statistics for all outcome types
            - Inserts statistics summary at the beginning of the log file
            - Writes the complete log content to the log file
            - Overwrites any existing log content with the finalized version

        Example:
            >>> log.commit()
            # Log file now contains:
            # [Header]
            # num_successful: 95 (87.16%)
            # num_errors: 12 (11.01%)
            # num_skipped: 2 (1.83%)
            # ...
            # [Individual conversion entries]
        """
        stats_lines = [
            f"{key}: {value[0]} ({value[1] * 100:.2f}%)\n" if isinstance(value, tuple) else f"{key}: {value}\n"
            for key, value in self.stats.items()
        ] + ["\n"]

        for i, line in enumerate(stats_lines):
            self.text.insert(1 + i, line)

        with self.path.open(mode="w", encoding="utf-8") as file:
            file.write("".join(self.text))

if __name__ == "__main__":
    pass
