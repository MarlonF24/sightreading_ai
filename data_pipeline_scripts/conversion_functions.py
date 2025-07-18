from typing import *
from pathlib import Path
from pyparsing import cached_property
from data_pipeline_scripts.conversion_func_infrastructure import *
from tokeniser.tokeniser import MyTokeniser, Metadata
       
class Generics:
    """
    A utility class containing generic methods used in the data pipeline scripts.
    """

    @staticmethod
    def str_error(e) -> str:
            """
            Generate a string representation of an exception.

            Parameters:
                e (Exception): The exception to be represented as a string.
    
            Returns:
                str: A string representation of the exception in the format "type(e).__name__: str(e)".
            """
            return f"{type(e).__name__}: {str(e)}"
    
    @staticmethod
    def clear_n_terminal_lines(n: int = 1):
        """
        Clear the last n lines in the terminal.
        
        Parameters:
            n (int): The number of lines to clear, defaults to 1.
        """
        import sys
        
        sys.stdout.write(n * "\033[F\033[K")
        sys.stdout.flush()

    @staticmethod
    def mute_decorator(func: Callable):
        """
        A decorator to mute the output of the function.
        """
        def wrapper(*args, **kwargs):
            import io, sys
            text_trap = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = text_trap
                sys.stderr = text_trap
                return func(*args, **kwargs)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        return wrapper


    @staticmethod
    def invalid_metadata_skip(input_file: FilePath, tokenised_data: Dict[str, str], tokeniser: MyTokeniser) -> List[ConversionOutcome]:
        b, err = tokeniser.valid_metadata(tokenised_data)
        if not b:
            return [ConversionOutcome(
                input_file=input_file,
                skipped=True,
                error_message=f"Invalid metadata: {err}",
            )]
        
        return []


    @staticmethod
    def find_same_name_outcomes(input_file: FilePath, output_folder: FolderPath) -> List[FilePath]:
        """
        This function finds all files in the output folder whose name starts with the name of the input file.
        It uses the file's stem (name without extension) to match files.

        Parameters:
            input_file (FilePath): The input file for which to find matching files in the output folder.
            output_folder (FolderPath): The folder in which to search for matching files.

        Returns:
            List[FilePath]: A list of file paths that match the input file's name in the output folder.
        """
        return [file for file in output_folder.glob(f"{input_file.stem}*.*")]


    @staticmethod
    def same_name_skip(input_file: FilePath, output_folder: FolderPath) -> List[ConversionOutcome]:
        """
        This function checks if there are any existing files in the output folder whose name starts with the name of the input file.
        If such files exist, it returns a list containing a ConversionOutcome object indicating that the conversion was skipped.
        If no matching files are found, it returns an empty list.

        Attributes:
            input_file (FilePath): The input file for which to check for matching files in the output folder.
            output_folder (FolderPath): The folder in which to search for matching files.
    
        Returns:
            List[ConversionOutcome]: A list containing a ConversionOutcome object if matching files are found, or an empty list otherwise.
        """
        res: List[FilePath] = Generics.find_same_name_outcomes(input_file, output_folder)
    
        return [ConversionOutcome(input_file=input_file, output_files=res, skipped=True)] if res else []
    

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
        """
        This function is a generic utility for executing subprocess commands, interpreting their output, and handling cleanup.

        Attributes:
            input_path (FilePath | FolderPath): The input file or folder for the conversion process.
            output_folder (FolderPath): The folder where the output files will be saved.
            command (List[str]): The command to be executed as a list of strings.
            interpreter (SingleFileInterpreter | BatchInterpreter): The function that interprets the subprocess output.
            batch (bool): A flag indicating whether the conversion is batch or single file.
            clean_up (Optional[SingleFileCleanUp | BatchCleanUp]): An optional function to perform cleanup after the conversion.

        Returns:
            List[ConversionOutcome]: A list of ConversionOutcome objects representing the outcome of the conversion process.
        """
        import subprocess
        
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            return interpreter(result.returncode, result.stderr.strip(), result.stdout.strip(), input_path, output_folder)
            
        except Exception as e:
            return [ConversionOutcome(
                input_file=input_path,
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
        func: Callable[[FilePath, FilePath], None | List[ConversionOutcome]]
        ) -> List[ConversionOutcome]:
        """
        This function is a generic utility for converting music files using the music21 library.
        It catches warnings during the conversion process and handles exceptions.

        Attributes:
            input_file: (FilePath): The path to the input music file.
            output_file (FilePath): The path where the converted music file will be saved.
            func (Callable[[FilePath, FilePath], None]): The function that performs the actual conversion.

        Returns:
            List[ConversionOutcome]: A list containing a single ConversionOutcome object representing the outcome of the conversion process.
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                if t := func(input_file, output_file):
                    return t
                
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
                    successful=False,
                    error_message=Generics.str_error(e)
                )]
            
    
class pdf_to_mxl(BatchConversionFunction):   
    """
    A class representing a batch conversion function for converting PDF files to MXL files using Audiveris.

    Inherits from BatchConversionFunction.

    Attributes:
        audiveris_app_folder (FolderPath): The path to the folder containing the Audiveris application's jar files.
        do_clean_up (bool): A flag indicating whether to perform cleanup after the conversion process.
    """

    def __init__(self, audiveris_app_folder: FolderPath, do_clean_up: bool = True):
        """
        Initialise a new instance of the pdf_to_mxl class.

        Parameters:
            audiveris_app_folder (FolderPath): _description_
            do_clean_up (bool): _description_, defaults to True
    
        """        
        self._audiveris_app_folder = audiveris_app_folder
        self.do_clean_up = do_clean_up

    @property
    def audiveris_app_folder(self) -> FolderPath:
        return self._audiveris_app_folder

    @cached_property
    def classpath(self) -> str:   
        return ";".join([str(jar_file) for jar_file in self.audiveris_app_folder.glob("*.jar")])

    def skip_single_file(self, input_file, output_folder):
        return Generics.same_name_skip(input_file, output_folder)

    def single_file_interpreter(self, returncode: int, stdout: str, stderr: str, input_file: FilePath, output_folder: FolderPath) -> List[ConversionOutcome]:
        """
        Interprets the output of a single file conversion process.

        Parameters:
            returncode (int): The return code of the conversion process.
            stdout (str): The standard output of the conversion process.
            stderr (str): The standard error output of the conversion process.
            input_file (FilePath): The input file for the conversion process.
            output_folder (FolderPath): The folder where the output files are saved.

        Returns:
            List[ConversionOutcome]: A list containing a single ConversionOutcome object representing the outcome of the conversion process.
        """
        if temp := "Could not export since transcription did not complete successfully" in stdout:
            return [ConversionOutcome(
                input_file=input_file,
                successful=False,
                error_message=temp
            )]

        output_files = [file for file in output_folder.glob(f"{input_file.stem}*.mxl")]
        
        if returncode == 0:

            warnings = [f"{line.strip().split("]")[1]}" for line in stdout.split('\n') if line.strip().startswith("WARN")]

            return [ConversionOutcome(input_file=input_file, output_files=output_files, successful=True, warning_messages=warnings)]
        
        if "FATAL" in stderr:
            return [ConversionOutcome(input_file=input_file, output_files=output_files, successful=False, error_message=stderr)]
        
        return [ConversionOutcome(input_file=input_file, output_files=output_files, successful=False, error_message=stderr)]
    
    def batch_interpreter(self, returncode: int, stderr: str, stdout: str, input_folder: FolderPath, output_folder: FolderPath) -> List[ConversionOutcome]:
        """
        Interprets the output of a batch conversion process.

        Parameters:
            returncode (int): The return code of the conversion process.
            stderr (str): The standard error output of the conversion process.
            stdout (str): The standard output of the conversion process.
            input_folder (FolderPath): The folder containing the input files for the conversion process.
            output_folder (FolderPath): The folder where the output files will be saved.

        Returns:
            List[ConversionOutcome]: A list of ConversionOutcome objects representing the outcome of the conversion process.
        """
        if returncode == 0:
            res = []
            
            for input_file in input_folder.glob("*.pdf"):
                temp = f"INFO  [{input_file.stem}]"
                stdout_section = stdout[stdout.index(temp) : stdout.rfind(temp) + len(temp)]
                res.extend(self.single_file_interpreter(0, stdout_section, "", input_file, output_folder))
            
            return res
   
        if "FATAL" in stderr:
            return [ConversionOutcome(input_file=input_folder, successful=False, error_message=stderr, halt=True)]
        
        return [ConversionOutcome(input_file=input_folder, successful=False, error_message=stderr)]

    def batch_clean_up(self, input_folder: FolderPath, output_folder: FolderPath) -> None:
        for file in output_folder.iterdir():
            if file.suffix != '.mxl':
                file.unlink() if file.is_file() else file.rmdir()

    def single_file_clean_up(self, input_file: FilePath, output_folder: FolderPath) -> None:
        self.batch_clean_up(Path(""), output_folder)

    def single_file_conversion(self, input_file: FilePath, output_folder: FolderPath, overwrite: bool = True):  
        command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", self.classpath, "Audiveris", "-batch", "-export", "-output", str(output_folder), "--", str(input_file)]
         
        return Generics.generic_subprocess_conversion(
            input_path=input_file, 
            output_folder=output_folder, 
            command=command, 
            interpreter=self.single_file_interpreter, 
            batch=False)

    def batch_conversion(self, input_folder, output_folder, overwrite = True):
            input_files = [str(f) for f in input_folder.glob("*.pdf")]

            command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", self.classpath, "Audiveris", "-batch", "-export", "-output", str(output_folder), "--", *input_files]
            
            return Generics.generic_subprocess_conversion(
                input_path=input_folder, 
                output_folder=output_folder, 
                command=command, 
                interpreter=self.batch_interpreter, 
                batch=True)
        

class mxl_to_musicxml_music21(SingleFileConversionFunction): 
    """
    This class is responsible for converting a MXL file to a MusicXML file using the music21 library.

    Inherits from SingleFileConversionFunction.
    """
    
    def skip_single_file(self, input_file, output_folder):
        return Generics.same_name_skip()

    def music21_func(self, input_file: FilePath, output_file: FilePath) -> None:
        """
        This function is responsible for converting a MXL file to a MusicXML file using the music21 library.

        Parameters:
            input_file (FilePath): The path to the input MXL file.
            output_file (FilePath): The path where the converted MusicML file will be saved.
        """
        import music21

        score = music21.converter.parse(input_file)
        score.write('musicxml', fp=output_file)
    
    def conversion(self, input_file: FilePath, output_folder: FolderPath, overwrite: bool = True) -> List[ConversionOutcome]:
        return Generics.generic_music21_conversion(input_file, output_folder.joinpath(input_file.stem + ".musicxml"), self.music21_func)

    
class mxl_to_musicxml_unzip(SingleFileConversionFunction):
    """
    This class is responsible for converting a MXL file to a MusicXML file via unzipping.

    Inherits from SingleFileConversionFunction.
    """
    
    def skip_single_file(self, input_file, output_folder):
        return Generics.same_name_skip()
    
    def clean_up(self, input_file: FilePath, output_folder: FolderPath):
        import shutil

        for file in output_folder.iterdir():
                if file.suffix != '.musicxml':
                    file.unlink() if file.is_file() else shutil.rmtree(output_folder)
    
    def conversion(self, input_file, output_folder, overwrite = True):    
        import zipfile, os
        
        try:
            with zipfile.ZipFile(input_file, 'r') as archive:
                name = archive.namelist()[0]
                extracted_path = archive.extract(name, output_folder)
                
        except zipfile.BadZipFile:
            return [ConversionOutcome(
                input_file=input_file, 
                successful=False,
                error_message="Input file is not a valid ZIP archive.",
                halt=False)]
        
        else:
            new_path = output_folder / (input_file.stem + ".musicxml")

            if new_path.exists():
                new_path.unlink()

            os.rename(extracted_path, new_path)
            
            return [ConversionOutcome( 
                input_file=input_file, 
                output_files=[Path(input_file.stem + ".musicxml")],

                successful=True)]


class musicxml_to_midi(SingleFileConversionFunction):
    """
    This class is responsible for converting a MusicXML file to a MIDI file using the music21 library.
    It also extracts metadata from the parsed music score and saves it to a JSON file in a separate folder.

    Inherits from SingleFileConversionFunction.
    """

    def __init__(self, tokeniser: MyTokeniser):
        """
        Initialise a new instance of the musicxml_to_midi class.
        """
        self.tokeniser = tokeniser

    def skip_single_file(self, input_file, output_folder):
        return Generics.same_name_skip()

    def clean_up(self, input_file, output_folder):
        pass

    def music21_func(self, input_file: FilePath, output_file: FilePath) -> None | List[ConversionOutcome]:
        """
        This function is responsible for converting a MusicXML file to a MIDI file using the music21 library.
        It also extracts metadata from the parsed music score and saves it to a JSON file.

        Parameters:
            input_file (FilePath): The path to the input MusicXML file.
            output_file (FilePath): The path where the converted MIDI file shall be saved. !! This address is changed to save the MIDI file and metadata in separate folders inside the original folder address.
        """
        from tokeniser.tokeniser import Metadata
        import music21, json
        
        metadata_folder: FolderPath = output_file.parent.joinpath("metadata_files") 
        metadata_folder.mkdir(parents=True, exist_ok=True)  # Create metadata folder if it doesn't exist
        
        metadata_file: FilePath = metadata_folder.joinpath(input_file.stem + ".meta.json")  
        
        score = music21.converter.parse(input_file)
        
        if not isinstance(score, music21.stream.Score):
            raise ValueError("Input file is not a valid MusicXML file.")
        
        metadata = Metadata(score)

        if t := Generics.invalid_metadata_skip(input_file, metadata.tokenised_data, self.tokeniser):
            return t

        lh = score.parts[1]

        lh.removeByClass(music21.instrument.Instrument)
       
        piano_lh = music21.instrument.ElectricPiano()
        lh.insert(0, piano_lh)
        
        score.write("midi", fp=output_file)
        
        with open(metadata_file, "w") as f:
            json.dump(metadata.tokenised_data, f, indent=4)

        return None
    
    def conversion(self, input_file: FilePath, output_folder: FilePath, overwrite: bool = True) -> List[ConversionOutcome]:
        return Generics.generic_music21_conversion(input_file, output_folder.joinpath(input_file.stem + ".midi"), self.music21_func)
    
    
class midi_to_tokens(SingleFileConversionFunction):
    
    def __init__(self, tokeniser: MyTokeniser):
        """
        Initialise a new instance of the midi_to_tokens class.
        
        Parameters:
            tokeniser (MyTokeniser): An instance of MyTokeniser used for encoding MIDI files into tokens.
        """
        self.tokeniser = tokeniser

    def skip_single_file(self, input_file, output_folder):
        return Generics.same_name_skip(input_file, output_folder)

    def conversion(self, input_file: FilePath, output_folder: FolderPath, overwrite: bool = True) -> List[ConversionOutcome]:    
        import miditok, json

        with input_file.parent.joinpath("metadata_files", input_file.stem + ".meta.json").open() as f:
            metadata = json.load(f)

        if t := Generics.invalid_metadata_skip(input_file, metadata, self.tokeniser):
            return t

        token_seq = self.tokeniser.encode(input_file)
        Generics.clear_n_terminal_lines(3)

        if not isinstance(token_seq, miditok.TokSequence):
            raise ValueError("Tokenisation failed. The output is not a valid TokSequence object.")

        jso = self.tokeniser.create_training_json(metadata, token_seq)

        output_path = output_folder.joinpath(input_file.stem + ".tokens.json")
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(jso, f)

        return [ConversionOutcome(
            input_file=input_file,
            output_files=[output_path]
        )]


    def clean_up(self, input_file, output_folder):
        pass


if __name__ == "__main__":
    pass
