import warnings, music21, subprocess, os
from typing import *
from dataclasses import dataclass
from pathlib import Path


FilePath = Path
FolderPath = Path

@dataclass
class ConversionOutcome():
    input_file: FilePath
    output_files: List[FilePath]
    warning_messages: List[str]
    
    successful: bool = True
    error_message: str = ""
    go_on: bool = True
    skipped: bool = False

        
class SingleFileConversionFunction():
    def __init__(self, func: Callable[[FilePath, FolderPath, bool], List[ConversionOutcome]]) -> None:
        self.func = func
        self.is_batchable = False

    def __call__(self, input_file: FilePath, output_folder: FolderPath, overwrite: bool = True) -> List[ConversionOutcome]:
            return self.func(input_file, output_folder, overwrite)  


class BatchConversionFunction():
    def __init__(self, func: Callable[[FilePath | FolderPath, FolderPath, bool, bool], List[ConversionOutcome]]) -> None:
        self.func = func
        self.is_batchable = True
    
    @overload
    def __call__(self, input_path: FilePath, output_folder: FolderPath, do_batch: Literal[False], overwrite: bool = True) -> List[ConversionOutcome]: ...
    
    @overload
    def __call__(self, input_path: FolderPath, output_folder: FolderPath, do_batch: Literal[True], overwrite: bool = True) -> List[ConversionOutcome]: ...
    
    def __call__(self, input_path: FilePath | FolderPath, output_folder: FolderPath, do_batch: bool = True, overwrite: bool = True) -> List[ConversionOutcome]:
        return self.func(input_path, output_folder, do_batch, overwrite)
    

ConversionFunction = SingleFileConversionFunction | BatchConversionFunction

class Generics:
    @staticmethod
    def str_error(e) -> str:
        return f"{type(e).__name__}: {str(e)}"
    
    @staticmethod
    def find_same_name_outcomes(input_file: FilePath, output_folder: FolderPath) -> List[FilePath]:
        return [file for file in output_folder.glob(f"{input_file.stem}*.*")]


    @staticmethod
    def same_name_skip(input_file: FilePath, output_folder: FolderPath, single_occurence: bool = False) -> List[ConversionOutcome]:
        res: List[FilePath] =  Generics.find_same_name_outcomes(input_file, output_folder)

        return [ConversionOutcome(input_file=input_file, output_files=res, warning_messages=[], skipped=True)] if res else []
    


    SingleFileInterpreter = Callable[[int, str, str, FilePath, FolderPath], List[ConversionOutcome]]
    BatchInterpreter = Callable[[int, str, str, FolderPath, FolderPath], List[ConversionOutcome]]
    SingleFileCleanUp = Callable[[FilePath, FolderPath], None]
    BatchCleanUp = Callable[[FolderPath, FolderPath], None]
    Skip = Callable[[FilePath, FolderPath], List[FilePath]]

    @overload
    @staticmethod
    def generic_subprocess_conversion( 
        input_path: FilePath,
        output_folder: FolderPath,
        command: List[str],
        interpreter: SingleFileInterpreter,
        batch: Literal[False],
        clean_up: Optional[SingleFileCleanUp] = None
        ) -> List[ConversionOutcome]: ...
         

    @overload
    @staticmethod
    def generic_subprocess_conversion( 
        input_path: FolderPath,
        output_folder: FolderPath,
        command: List[str],
        interpreter: BatchInterpreter,
        batch: Literal[True],
        clean_up: Optional[BatchCleanUp] = None
        ) -> List[ConversionOutcome]: ...
    

    @staticmethod
    def generic_subprocess_conversion( 
        input_path: FilePath | FolderPath,
        output_folder: FolderPath,
        command: List[str],
        interpreter: SingleFileInterpreter | BatchInterpreter,
        batch: bool,
        clean_up: Optional[SingleFileCleanUp | BatchCleanUp] = None
        ) -> List[ConversionOutcome]: 
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    return interpreter(result.returncode, result.stderr.strip(), result.stdout.strip(), input_path, output_folder)
                    
                except Exception as e:
                    return [ConversionOutcome(
                        input_file=input_path,
                        output_files=[],
                        warning_messages=[],
                        successful=False,
                        error_message=Generics.str_error(e),
                        go_on=False
                    )]
                finally:
                    if clean_up:
                        clean_up(input_path, output_folder)
                

    @staticmethod
    def generic_music21_conversion(
        input_file: FilePath, 
        output_file: FilePath, 
        func: Callable[[FilePath, FilePath], None]
        ) -> List[ConversionOutcome]:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                func(input_file, output_file)
                # print([str(w.message) for w in caught_warnings])
                return [ConversionOutcome(
                    input_file=input_file,
                    output_files=[output_file],
                    successful=True,
                    warning_messages=[str(w.message) for w in caught_warnings],
                    go_on=True
                )]
            # except music21.exceptions21. as e:
            #     return [ConversionOutcome(
            #         input_file=input_file,
            #         output_files=[output_file],
            #         warning_messages=[],
            #         successful=False,
            #         error_message=Generics.str_error(e),
            #         go_on=False  # serious crash
            #     )]
            except Exception as e:
                return [ConversionOutcome(
                    input_file=input_file,
                    output_files=[output_file],
                    warning_messages=[],
                    successful=False,
                    error_message=Generics.str_error(e),
                    go_on=True
                )]
            


def mxl_to_musicxml() -> SingleFileConversionFunction: 
    def music21_func(input_file: FilePath, output_file: FilePath) -> None:
        score = music21.converter.parse(input_file)
        score.write('musicxml', fp=output_file)
    
    def func(input_file: FilePath, output_folder: FolderPath, overwrite: bool) -> List[ConversionOutcome]:
        if not overwrite:
            if outcome := Generics.same_name_skip(input_file, output_folder):
                return outcome
        return Generics.generic_music21_conversion(input_file, output_folder.joinpath(input_file.stem + ".musicxml"), music21_func)

    return SingleFileConversionFunction(func)
    
def pdf_to_mxl(audiveris_app_folder: FolderPath, do_clean_up: bool = True) -> BatchConversionFunction:
    def single_file_interpreter(returncode: int, stdout: str, stderr: str, input_file: FilePath, output_folder: FolderPath) -> List[ConversionOutcome]:

        output_files = [file for file in output_folder.glob(f"{input_file.stem}*.mxl")]
        
        if returncode == 0:

            warnings = [line.strip().split("]")[1] for line in stdout.split('\n') if line.strip().startswith("WARN")]

            return [ConversionOutcome(input_file=input_file, output_files=output_files, successful=True, warning_messages=warnings, go_on=True)]
        
        if "FATAL" in stderr:
            return [ConversionOutcome(input_file=input_file, output_files=output_files, warning_messages=[],successful=False, error_message=stderr, go_on=False)]
        
        return [ConversionOutcome(input_file=input_file, output_files=output_files, warning_messages=[],successful=False, error_message=stderr, go_on=True)]
    
    def batch_interpreter(returncode: int, stderr: str, stdout: str, input_folder: FolderPath, output_folder: FolderPath) -> List[ConversionOutcome]:
        if returncode == 0:
            res = []
            
            for input_file in input_folder.glob("*.pdf"):
                temp = f"INFO  [{input_file.stem}]"
                stdout_section = stdout[stdout.index(temp) : stdout.rfind(temp) + len(temp)]
                res.extend(single_file_interpreter(0, stdout_section, "", input_file, output_folder))
            
            return res
   
        if "FATAL" in stderr:
            return [ConversionOutcome(input_file=Path(""), output_files=[], warning_messages=[],successful=False, error_message=stderr, go_on=False)]
        
        return [ConversionOutcome(input_file=Path(""), output_files=[], warning_messages=[],successful=False, error_message=stderr, go_on=True)]


    
    def batch_clean_up(input_folder: FolderPath, output_folder: FolderPath) -> None:
        for file in output_folder.iterdir():
            if file.suffix!= '.mxl':
                file.unlink() if file.is_file() else file.rmdir()

    def single_file_clean_up(input_file: FilePath, output_folder: FolderPath) -> None:
        batch_clean_up(Path(""), output_folder)

    
    def func(input_path: FilePath | FolderPath, output_folder: FolderPath, do_batch: bool, overwrite: bool) -> List[ConversionOutcome]:            
        
        classpath = ";".join([str(jar_file) for jar_file in audiveris_app_folder.glob("*.jar")])
        
        if do_batch:
            input_files = [str(f) for f in input_path.glob("*.pdf")]

            command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", classpath, "Audiveris", "-batch", "-export", "-output", str(output_folder), "--", *input_files]
            
            return Generics.generic_subprocess_conversion(
                input_path=input_path, 
                output_folder=output_folder, 
                command=command, 
                interpreter=batch_interpreter, 
                batch=do_batch, 
                clean_up=batch_clean_up if do_clean_up else None)
        else:
            if not overwrite:
                if outcome := Generics.same_name_skip(input_path, output_folder):
                    return outcome
            
            command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", classpath, "Audiveris", "-batch", "-export", "-output", str(output_folder), "--", str(input_path)]
            
            return Generics.generic_subprocess_conversion(
                input_path=input_path, 
                output_folder=output_folder, 
                command=command, 
                interpreter=single_file_interpreter, 
                batch=do_batch, 
                clean_up=single_file_clean_up if do_clean_up else None)

    return BatchConversionFunction(func)


# def musicxml_to_pdf(input_file: FilePath, output_file: FilePath) -> ConversionFunction:
#     pass

if __name__ == "__main__":
    pass