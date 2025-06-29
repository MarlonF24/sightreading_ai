from dataclasses import dataclass
from typing import *
import  os

Path = str
Extension = str

@dataclass
class Storable():
    path: Path

    def __post_init__(self) -> None:
        self.name: str = os.path.basename(self.path)
        

@dataclass
class File(Storable):

    def __post_init__(self) -> None:
        self.folder_path: Path = os.path.dirname(self.path)
        self.extension: Extension = os.path.splitext(self.name)[1]



@dataclass
class ConversionOutcome():
    input_file: File
    output_file: File
    
    successful: bool = True
    warning_messages: Tuple[str, ...] = ()
    error_message: str = ""
    go_on: bool = True
    

if __name__ == "__main__":
    print(float("inf") - 1)

