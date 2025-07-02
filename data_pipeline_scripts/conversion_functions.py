import warnings, music21, subprocess, json
from typing import *
from pathlib import Path
from conversion_func_infrastructure import *
       
class Generics:
    @staticmethod
    def str_error(e) -> str:
        return f"{type(e).__name__}: {str(e)}"
    
    @staticmethod
    def find_same_name_outcomes(input_file: FilePath, output_folder: FolderPath) -> List[FilePath]:
        return [file for file in output_folder.glob(f"{input_file.stem}*.*")]


    @staticmethod
    def same_name_skip(input_file: FilePath, output_folder: FolderPath) -> List[ConversionOutcome]:
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
                        halt=True
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
                return [ConversionOutcome(
                    input_file=input_file,
                    output_files=[output_file],
                    successful=True,
                    warning_messages=[str(w.message) for w in caught_warnings]
                )]
            except Exception as e:
                return [ConversionOutcome(
                    input_file=input_file,
                    output_files=[output_file],
                    warning_messages=[],
                    successful=False,
                    error_message=Generics.str_error(e)
                )]
            
    
class pdf_to_mxl(BatchConversionFunction):
    
    def __init__(self, audiveris_app_folder: FolderPath, do_clean_up: bool = True):
        self.audiveris_app_folder = audiveris_app_folder
        self.do_clean_up = do_clean_up

    def single_file_interpreter(self, returncode: int, stdout: str, stderr: str, input_file: FilePath, output_folder: FolderPath) -> List[ConversionOutcome]:

        output_files = [file for file in output_folder.glob(f"{input_file.stem}*.mxl")]
        
        if returncode == 0:

            warnings = [line.strip().split("]")[1] for line in stdout.split('\n') if line.strip().startswith("WARN")]

            return [ConversionOutcome(input_file=input_file, output_files=output_files, successful=True, warning_messages=warnings)]
        
        if "FATAL" in stderr:
            return [ConversionOutcome(input_file=input_file, output_files=output_files, warning_messages=[],successful=False, error_message=stderr)]
        
        return [ConversionOutcome(input_file=input_file, output_files=output_files, warning_messages=[],successful=False, error_message=stderr)]
    
    def batch_interpreter(self, returncode: int, stderr: str, stdout: str, input_folder: FolderPath, output_folder: FolderPath) -> List[ConversionOutcome]:
        if returncode == 0:
            res = []
            
            for input_file in input_folder.glob("*.pdf"):
                temp = f"INFO  [{input_file.stem}]"
                stdout_section = stdout[stdout.index(temp) : stdout.rfind(temp) + len(temp)]
                res.extend(self.single_file_interpreter(0, stdout_section, "", input_file, output_folder))
            
            return res
   
        if "FATAL" in stderr:
            return [ConversionOutcome(input_file=Path(""), output_files=[], warning_messages=[],successful=False, error_message=stderr, halt=True)]
        
        return [ConversionOutcome(input_file=Path(""), output_files=[], warning_messages=[], successful=False, error_message=stderr)]

    def batch_clean_up(self, input_folder: FolderPath, output_folder: FolderPath) -> None:
        for file in output_folder.iterdir():
            if file.suffix!= '.mxl':
                file.unlink() if file.is_file() else file.rmdir()

    def single_file_clean_up(self, input_file: FilePath, output_folder: FolderPath) -> None:
        self.batch_clean_up(Path(""), output_folder)

    def __call__(self, input_path: FilePath | FolderPath, output_folder: FolderPath, do_batch: bool = True, overwrite: bool = True) -> List[ConversionOutcome]:            
        
        classpath = ";".join([str(jar_file) for jar_file in self.audiveris_app_folder.glob("*.jar")])
        
        if do_batch:
            input_files = [str(f) for f in input_path.glob("*.pdf")]

            command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", classpath, "Audiveris", "-batch", "-export", "-output", str(output_folder), "--", *input_files]
            
            return Generics.generic_subprocess_conversion(
                input_path=input_path, 
                output_folder=output_folder, 
                command=command, 
                interpreter=self.batch_interpreter, 
                batch=do_batch, 
                clean_up=self.batch_clean_up if self.do_clean_up else None)
        
        else:
            if not overwrite:
                if outcome := Generics.same_name_skip(input_path, output_folder):
                    return outcome
            
            command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", classpath, "Audiveris", "-batch", "-export", "-output", str(output_folder), "--", str(input_path)]
            
            return Generics.generic_subprocess_conversion(
                input_path=input_path, 
                output_folder=output_folder, 
                command=command, 
                interpreter=self.single_file_interpreter, 
                batch=do_batch, 
                clean_up=self.single_file_clean_up if self.do_clean_up else None)

    

class mxl_to_musicxml(SingleFileConversionFunction): 
    def music21_func(self, input_file: FilePath, output_file: FilePath) -> None:
        score = music21.converter.parse(input_file)
        score.write('musicxml', fp=output_file)
    
    def __call__(self, input_file: FilePath, output_folder: FolderPath, overwrite: bool = True) -> List[ConversionOutcome]:
        if not overwrite:
            if outcome := Generics.same_name_skip(input_file, output_folder):
                return outcome
        return Generics.generic_music21_conversion(input_file, output_folder.joinpath(input_file.stem + ".musicxml"), self.music21_func)

    

class musicxml_to_midi(SingleFileConversionFunction):
    
    
    def music21_func(self, input_file: FilePath, output_file: FilePath) -> None:
        from tokenisation import extract_metadata
        
        metadata_folder: FolderPath = output_file.parent.parent.joinpath("metadata_files") 
        metadata_folder.mkdir(parents=True, exist_ok=True)  # Create metadata folder if it doesn't exist
        
        metadata_file: FilePath = metadata_folder.joinpath(input_file.stem + ".meta.json")  
        
        score = music21.converter.parse(input_file)
        score.write('midi', fp=output_file)
        

        if isinstance(score, music21.stream.base.Score): 
            metadata = extract_metadata(score)

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
    
    def __call__(self, input_file: FilePath, output_folder: FilePath, overwrite: bool = True) -> List[ConversionOutcome]:
        output_folder = output_folder.joinpath("midi_files")
        output_folder.mkdir(parents=True, exist_ok=True) 
        
        if not overwrite:
            if outcome := Generics.same_name_skip(input_file, output_folder):
                return outcome
        return Generics.generic_music21_conversion(input_file, output_folder.joinpath(input_file.stem + ".midi"), self.music21_func)

if __name__ == "__main__":
    print(type(music21.converter.parse(Path(r"C:\Users\marlo\sightreading_ai\data_pipeline\data\musicxml_in\C._Schfer_A._Sartorio_Op._45_-_Volume_2_-_Melodious_Exercises_Piano.mvt1.musicxml"))))