from dataclasses import dataclass
from typing import *
import  os

@dataclass
class File():
    path: str

    def __post_init__(self):
        self.name = os.path.basename(self.path)
        self.folder = os.path.dirname(self.path)
        self.extension = os.path.splitext(self.name)[1]

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

