from dataclasses import dataclass
from enum import Enum, auto
from typing import *
from pipeline_stage import Pipeline_stage
import datetime, os

@dataclass
class File():
    path: str

    def __post_init__(self):
        self.name = os.path.basename(self.path)
        self.extention = os.path.splitext(self.name)

@dataclass
class ConversionOutcome:
    input_file: File
    
    output_file: File
    
    successful: bool = True
    warning_messages: Tuple[str] = ()
    error_message: Optional[str] = None
    go_on: bool = True
    


class Log(File) :
    

    def __init__(self, start_stage: Pipeline_stage, target_stage: Pipeline_stage, log_folder):
        super().__init__(os.path.join(log_folder, f"{start_stage.name}_to_{target_stage.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"))
        
        self.start_stage = start_stage
        self.target_stage = target_stage
        self._is_open = False
        self._file = None
        
    class LogClosedError(Exception):
        """Raised when attempting to write to a closed log."""
        pass

    def __enter__(self):
        self._file = open(self.path, "w", encoding="utf-8")
        self._is_open = True
        self._file.write(self.name + f"\n {self.start_stage.folder_path} -> {self.target_stage.folder_path}" + 4 * "\n")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()
        self._is_open = False


    def _check_open(self) -> None:
        if not self._is_open:
            raise Log.LogClosedError(f"Attempted to write to log {self.name} outside of context manager (log is closed).")

    
    def skip(self, input_file: File, output_file: File, reason: str = None) -> None:
        if reason is None:
            reason = f"Output: {output_file.name} already exists and shall not be overwritten\n"
        self._check_open()
        self._file.write(f"[SKIPPED] {datetime.datetime.now()} - Input: {input_file.name}; \n"
                         + reason)
    
    
    def log(self, outcome: ConversionOutcome) -> None:
        self._check_open()
        if outcome.successful:
           self._file.write(f"[SUCCESS] {datetime.datetime.now()} - Input: {outcome.input_file.name}; Output: {outcome.output_file.name}:\n") 

           for warning_message in outcome.warning_messages:
                self._file.write(f"       [WARNING] {warning_message}\n")

        else:
            if outcome.go_on:
                self._file.write(f"[ERROR] {datetime.datetime.now()} - Input: {outcome.input_file.name}\n:" + 
                                 f"         Error: {outcome.error_message}\n")
            else:
                raise Exception(outcome.error_message)
    
    def halt(self, input_file: File, error_message) -> None:
        self._check_open()
        self._file.write(f"[HALT] {datetime.datetime.now()} - ON {input_file.name}\n"+ 
                                 f"         Error: {error_message}\n")
        return input_file.name

    def evaluation(self):
        self._check_open()


if __name__ == "__main__":
    pass

