from typing import *
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

FilePath = Path
FolderPath = Path

@dataclass
class ConversionOutcome():
    input_file: FilePath
    output_files: List[FilePath]
    warning_messages: List[str]
    
    skipped: bool = False
    successful: bool = True
    error_message: str = ""
    halt: bool = False


class ConversionFunction(ABC):
    is_batchable: ClassVar[bool]

    @abstractmethod
    def __call__(self, *args, **kwargs) -> List[ConversionOutcome]:...


class SingleFileConversionFunction(ConversionFunction):
    is_batchable: ClassVar[bool] = False

    @abstractmethod
    def __call__(self, input_file: FilePath, output_folder: FolderPath, overwrite: bool = True) -> List[ConversionOutcome]: ...
        

class BatchConversionFunction(ConversionFunction):
    is_batchable: ClassVar[bool] = True
    
    @overload
    def __call__(self, input_path: FilePath, output_folder: FolderPath, do_batch: Literal[False], overwrite: bool = True) -> List[ConversionOutcome]: ...
    
    @overload
    def __call__(self, input_path: FolderPath, output_folder: FolderPath, do_batch: Literal[True], overwrite: bool = True) -> List[ConversionOutcome]: ...
    
    @abstractmethod
    def __call__(self, input_path: FilePath | FolderPath, output_folder: FolderPath, do_batch: bool = True, overwrite: bool = True) -> List[ConversionOutcome]: ...