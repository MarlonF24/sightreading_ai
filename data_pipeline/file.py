from dataclasses import dataclass
from typing import *
import  os

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
    

if __name__ == "__main__":
    print(float("inf") - 1)

