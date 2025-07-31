import traceback
from typing import *
from pathlib import Path
from pyparsing import cached_property
from data_pipeline_scripts.conversion_func_infrastructure import *
from tokeniser.tokeniser import MyTokeniser, Metadata
import constants 
       
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
        """Mute the output of the decorated function.

        :param func: The function whose output is to be muted.
        :type func: Callable
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
    def invalid_metadata_skip(input_file: FilePath, tokenised_data: Metadata | Metadata.TokenisedMetadata | Dict[str, str], tokeniser: MyTokeniser) -> List[ConversionOutcome]:
        """Check if the tokenised data contains valid metadata.

        :param input_file: The input file being processed.
        :type input_file: FilePath
        :param tokenised_data: The tokenised data to check.
        :type tokenised_data: Metadata | Metadata.TokenisedMetadata | Dict[str, str]
        :param tokeniser: The tokeniser used for validation.
        :type tokeniser: MyTokeniser
        :return: A list of ConversionOutcome objects indicating the result of the validation.
        :rtype: List[ConversionOutcome]
        """
        b, err = tokeniser.valid_metadata(tokenised_data)
        if not b:
            return [ConversionOutcome(
                input_file=input_file,
                skipped=True,
                error_message=f"Invalid metadata: {err}",
            )]
        
        return []


    @staticmethod
    def find_same_name_outcomes(input_file: FilePath, output_dir: DirPath) -> List[FilePath]:
        """
        This function finds all files in the output dir whose name starts with the name of the input file.
        It uses the file's stem (name without extension) to match files.

        Parameters:
            input_file (FilePath): The input file for which to find matching files in the output dir.
            output_dir (DirPath): The dir in which to search for matching files.

        Returns:
            List[FilePath]: A list of file paths that match the input file's name in the output dir.
        """
        return [file for file in output_dir.glob(f"{input_file.stem}*.*")]


    @staticmethod
    def same_name_skip(input_file: FilePath, output_dir: DirPath) -> List[ConversionOutcome]:
        """
        This function checks if there are any existing files in the output dir whose name starts with the name of the input file.
        If such files exist, it returns a list containing a ConversionOutcome object indicating that the conversion was skipped.
        If no matching files are found, it returns an empty list.

        Attributes:
            input_file (FilePath): The input file for which to check for matching files in the output dir.
            output_dir (DirPath): The dir in which to search for matching files.
    
        Returns:
            List[ConversionOutcome]: A list containing a ConversionOutcome object if matching files are found, or an empty list otherwise.
        """
        res: List[FilePath] = Generics.find_same_name_outcomes(input_file, output_dir)
    
        return [ConversionOutcome(input_file=input_file, output_files=res, skipped=True)] if res else []
    

    SingleFileInterpreter = Callable[[int, str, str, FilePath, DirPath], List[ConversionOutcome]]
    BatchInterpreter = Callable[[int, str, str, DirPath, DirPath], List[ConversionOutcome]]
    SingleFileCleanUp = Callable[[FilePath, DirPath], None]
    BatchCleanUp = Callable[[DirPath, DirPath], None]
    Skip = Callable[[FilePath, DirPath], List[FilePath]]

    @overload
    @staticmethod
    def generic_subprocess_conversion( 
        input_path: FilePath,
        output_dir: DirPath,
        command: List[str],
        interpreter: SingleFileInterpreter,
        batch: Literal[False],
        clean_up: Optional[SingleFileCleanUp] = None
        ) -> List[ConversionOutcome]: ...
         

    @overload
    @staticmethod
    def generic_subprocess_conversion( 
        input_path: DirPath,
        output_dir: DirPath,
        command: List[str],
        interpreter: BatchInterpreter,
        batch: Literal[True],
        clean_up: Optional[BatchCleanUp] = None
        ) -> List[ConversionOutcome]: ...
    

    @staticmethod
    def generic_subprocess_conversion( 
        input_path: FilePath | DirPath,
        output_dir: DirPath,
        command: List[str],
        interpreter: SingleFileInterpreter | BatchInterpreter,
        batch: bool
        ) -> List[ConversionOutcome]: 
        """
        This function is a generic utility for executing subprocess commands, interpreting their output, and handling cleanup.

        Attributes:
            input_path (FilePath | DirPath): The input file or dir for the conversion process.
            output_dir (DirPath): The dir where the output files will be saved.
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
                
        except Exception as e:
            return [ConversionOutcome(
                input_file=input_path,
                successful=False,
                error_message=Generics.str_error(e),
                halt=True
            )]
       
        else:
            return interpreter(result.returncode, result.stderr, result.stdout, input_path, output_dir)  

    @staticmethod
    def generic_music21_conversion(
        input_file: FilePath, 
        output_dir: FilePath, 
        func: (Callable[[FilePath, DirPath], List[FilePath]])
        ) -> List[ConversionOutcome]:
        """
        This function is a generic utility for converting music files using the music21 library.
        It catches warnings during the conversion process and handles exceptions.

        Attributes:
            input_file (FilePath): The path to the input music file.
            output_dir (DirPath): The path where the converted music file will be saved.
            func (Callable[[FilePath, DirPath], ]): The function that performs the actual conversion.

        Returns:
            List[ConversionOutcome]: A list containing a single ConversionOutcome object representing the outcome of the conversion process.
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                output_files = func(input_file, output_dir)  
            
            except Exception as e:
                return [ConversionOutcome(
                    input_file=input_file,
                    successful=False,
                    error_message=Generics.str_error(e)
                )]
            
            else:
                return [ConversionOutcome(
                        input_file=input_file,
                        output_files=output_files,
                        successful=True,
                        warning_messages=[str(w.message) for w in caught_warnings])]


class pdf_preprocessing(SingleFileConversionFunction):
    """
    A class representing a single file conversion function for preprocessing PDF files.
    """
    def __init__(self, pages_per_split: int = 1):
        self.pages_per_split = pages_per_split

    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> List[FilePath]:
        return Generics.same_name_skip(input_file, output_dir)
    
    def clean_up(self, input_file: FilePath, output_dir: DirPath) -> None:
        pass

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> List[ConversionOutcome]:
        from PyPDF2 import PdfReader, PdfWriter
        
        try:
            # Read the input PDF
            reader = PdfReader(str(input_file))
            total_pages = len(reader.pages)

            output_paths = []
            
            # Calculate number of splits needed
            num_splits = (total_pages + self.pages_per_split - 1) // self.pages_per_split  # Ceiling division

            for split_num in range(num_splits):
                start_page = split_num * self.pages_per_split
                end_page = min(start_page + self.pages_per_split, total_pages)

                # Create new PDF writer for this split
                writer = PdfWriter()
                
                # Add pages to this split
                for page_num in range(start_page, end_page):
                    writer.add_page(reader.pages[page_num])
                
                # Generate output filename
                stem = input_file.stem
                output_filename = f"{stem}_part_{split_num + 1:03d}.pdf"
                output_path = output_dir / output_filename
                
                # Write the split PDF
                with open(output_path, 'wb') as output_file:
                    writer.write(output_file)
                
                output_paths.append(output_path)
        
        except Exception as e:
            return [ConversionOutcome(
                input_file=input_file,
                successful=False,
                error_message=Generics.str_error(e),
                halt=False
            )]
        
        else:
            return [ConversionOutcome(
                input_file=input_file,
                output_files=output_paths,
                successful=True
            )]


class pdf_to_mxl(BatchConversionFunction):   
    """
    A class representing a batch conversion function for converting PDF files to MXL files using Audiveris.

    Inherits from BatchConversionFunction.

    Attributes:
        audiveris_app_dir (DirPath): The path to the dir containing the Audiveris application's jar files.
        do_clean_up (bool): A flag indicating whether to perform cleanup after the conversion process.
    """

    def __init__(self, audiveris_app_dir: DirPath, do_clean_up: bool = True):
        """
        Initialise a new instance of the pdf_to_mxl class.

        Parameters:
            audiveris_app_dir (DirPath): _description_
            do_clean_up (bool): _description_, defaults to True
    
        """        
        self._audiveris_app_dir = audiveris_app_dir
        self.do_clean_up = do_clean_up

    @property
    def audiveris_app_dir(self) -> DirPath:
        return self._audiveris_app_dir

    @cached_property
    def classpath(self) -> str:   
        return ";".join([str(jar_file) for jar_file in self.audiveris_app_dir.glob(f"*{constants.JAR_EXTENSION}")])

    def skip_single_file(self, input_file, output_dir):
        return Generics.same_name_skip(input_file, output_dir)

    def single_file_interpreter(self, returncode: int, stdout: str, stderr: str, input_file: FilePath, output_dir: DirPath) -> List[ConversionOutcome]:
        """
        Interprets the output of a single file conversion process.

        Parameters:
            returncode (int): The return code of the conversion process.
            stdout (str): The standard output of the conversion process.
            stderr (str): The standard error output of the conversion process.
            input_file (FilePath): The input file for the conversion process.
            output_dir (DirPath): The dir where the output files are saved.

        Returns:
            List[ConversionOutcome]: A list containing a single ConversionOutcome object representing the outcome of the conversion process.
        """

        critical_messages = ["Could not export since transcription did not complete successfully",
                             "resolution is too low"]
        
        if stdout:
            for temp in critical_messages:
                if temp in stdout:
                    return [ConversionOutcome(
                        input_file=input_file,
                        successful=False,
                        error_message=temp
                    )]

        if stderr:
            for temp in critical_messages:
                if temp in stderr:
                    return [ConversionOutcome(
                        input_file=input_file,
                        successful=False,
                        error_message=temp
                    )]        

        output_files = [file for file in output_dir.glob(f"{input_file.stem}*{constants.MXL_EXTENSION}")]
        
        if returncode == 0:

            warnings = [f"{line.strip().split(']')[1]}" for line in stdout.split('\n') if line.strip().startswith("WARN")]

            return [ConversionOutcome(input_file=input_file, output_files=output_files, successful=True, warning_messages=warnings)]
        
        if "FATAL" in stderr:
            return [ConversionOutcome(input_file=input_file, output_files=output_files, successful=False, error_message=stderr)]
        
        return [ConversionOutcome(input_file=input_file, output_files=output_files, successful=False, error_message=stderr)]
    
    def batch_interpreter(self, returncode: int, stderr: str, stdout: str, input_dir: DirPath, output_dir: DirPath) -> List[ConversionOutcome]:
        """
        Interprets the output of a batch conversion process.

        Parameters:
            returncode (int): The return code of the conversion process.
            stderr (str): The standard error output of the conversion process.
            stdout (str): The standard output of the conversion process.
            input_dir (DirPath): The dir containing the input files for the conversion process.
            output_dir (DirPath): The dir where the output files will be saved.

        Returns:
            List[ConversionOutcome]: A list of ConversionOutcome objects representing the outcome of the conversion process.
        """
        if returncode == 0:
            res = []

            for input_file in input_dir.glob(f"*{constants.PDF_EXTENSION}"):
                temp = f"INFO  [{input_file.stem}]"
                stdout_section = stdout[stdout.index(temp) : stdout.rfind(temp) + len(temp)]
                res.extend(self.single_file_interpreter(0, stdout_section, "", input_file, output_dir))
            
            return res
   
        if "FATAL" in stderr:
            return [ConversionOutcome(input_file=input_dir, successful=False, error_message=stderr, halt=True)]
        
        return [ConversionOutcome(input_file=input_dir, successful=False, error_message=stderr)]

    def batch_clean_up(self, input_dir: DirPath, output_dir: DirPath) -> None:
        for file in output_dir.iterdir():
            if file.suffix != '.mxl':
                file.unlink() if file.is_file() else file.rmdir()

    def single_file_clean_up(self, input_file: FilePath, output_dir: DirPath) -> None:
        self.batch_clean_up(Path(""), output_dir)

    def single_file_conversion(self, input_file: FilePath, output_dir: DirPath, overwrite: bool = True):  
        from data_pipeline_scripts import enhance_resolution 
        # resolution.convert(input_file, input_file)
        command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", self.classpath, "Audiveris", "-batch", "-export", "-output", str(output_dir), "--", str(input_file)]
          
        return Generics.generic_subprocess_conversion(
            input_path=input_file, 
            output_dir=output_dir, 
            command=command, 
            interpreter=self.single_file_interpreter, 
            batch=False)

    def batch_conversion(self, input_dir, output_dir, overwrite = True):
            input_files = [str(f) for f in input_dir.glob(f"*{constants.PDF_EXTENSION}")]

            command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", self.classpath, "Audiveris", "-batch", "-export", "-output", str(output_dir), "--", *input_files]
            
            return Generics.generic_subprocess_conversion(
                input_path=input_dir, 
                output_dir=output_dir, 
                command=command, 
                interpreter=self.batch_interpreter, 
                batch=True)
        

class mxl_to_musicxml_music21(SingleFileConversionFunction): 
    """
    This class is responsible for converting a MXL file to a MusicXML file using the music21 library.

    Inherits from SingleFileConversionFunction.
    """
    
    def skip_single_file(self, input_file, output_dir):
        return Generics.same_name_skip(input_file, output_dir)

    def music21_func(self, input_file: FilePath, output_dir: DirPath) -> List[FilePath]:
        """
        This function is responsible for converting a MXL file to a MusicXML file using the music21 library.

        Parameters:
            input_file (FilePath): The path to the input MXL file.
            output_file (FilePath): The path where the converted MusicML file will be saved.
        """
        import music21

        score = music21.converter.parse(input_file)
        score.write('musicxml', fp=output_dir / (input_file.stem + constants.MUSICXML_EXTENSION))
    
    def conversion(self, input_file: FilePath, output_dir: DirPath) -> List[ConversionOutcome]:
        return Generics.generic_music21_conversion(input_file, output_dir, self.music21_func)

    
class mxl_to_musicxml_unzip(SingleFileConversionFunction):
    """
    This class is responsible for converting a MXL file to a MusicXML file via unzipping.

    Inherits from SingleFileConversionFunction.
    """
    
    def skip_single_file(self, input_file, output_dir):
        return Generics.same_name_skip()
    
    def clean_up(self, input_file: FilePath, output_dir: DirPath):
        import shutil

        for file in output_dir.iterdir():
                if file.suffix != '.musicxml':
                    file.unlink() if file.is_file() else shutil.rmtree(output_dir)
    
    def conversion(self, input_file, output_dir):    
        import zipfile, os
        
        try:
            with zipfile.ZipFile(input_file, 'r') as archive:
                name = archive.namelist()[0]
                extracted_path = archive.extract(name, output_dir)
        
        except Exception as e:
            return [ConversionOutcome(
                input_file=input_file, 
                successful=False,
                error_message=Generics.str_error(e),
                halt=False)]
        
        else:
            new_path = output_dir / (input_file.stem + ".musicxml")

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
    It also extracts metadata from the parsed music score and saves it to a JSON file in a separate dir.

    Inherits from SingleFileConversionFunction.
    """

    def __init__(self, tokeniser: MyTokeniser):
        """Initialise a new instance of the musicxml_to_midi class.

        :param tokeniser: An instance of MyTokeniser used for tokenising MIDI files later. So that we can skip files that do not match the tokeniser vocab by their metadata.
        :type tokeniser: MyTokeniser
        """

        self.tokeniser = tokeniser

    def skip_single_file(self, input_file, output_dir):
        return Generics.same_name_skip()

    def clean_up(self, input_file, output_dir):
        pass

    def music21_func(self, input_file: FilePath, output_dir: DirPath) ->  List[FilePath]:
        """
        This function is responsible for converting a MusicXML file to a MIDI file using the music21 library.
        It also extracts metadata from the parsed music score and saves it to a JSON file.

        Parameters:
            input_file (FilePath): The path to the input MusicXML file.
            output_file (FilePath): The path where the converted MIDI file shall be saved. !! This address is changed to save the MIDI file and metadata in separate dirs inside the original dir address.
        """
        from tokeniser.tokeniser import Metadata
        import music21, json, warnings

        metadata_dir: DirPath = output_dir / constants.data_pipeline_constants.METADATA_DIR_NAME
        metadata_dir.mkdir(parents=True, exist_ok=True)  # Create metadata dir if it doesn't exist

        #try:
        score = music21.converter.parse(input_file)
        # except ZeroDivisionError as e:
        #     traceback.print_exc()
        #     raise ZeroDivisionError(f"artificial error")

        if not isinstance(score, music21.stream.Score):
            raise ValueError("Input file is not a valid MusicXML file.")

        measures = score.parts[0].getElementsByClass(music21.stream.Measure)
        splits = [1]  
        
        for i, measure in enumerate(measures, start=1):
            lol = measure.rightBarline
            if measure.finalBarline:
                splits.append(i)
        
        if not len(measures) in splits:
            splits.append(len(measures))

        split_scores = [score.measures(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]

        output_files: List[FilePath] = []
        error_counter = 0
        _warnings = []

        for i, score in enumerate(split_scores, start=1):
            try:
                metadata = Metadata(score)


                if t := Generics.invalid_metadata_skip(input_file, metadata, self.tokeniser):
                    raise ValueError(t[0].error_message)
                

                lh = score.parts[1]

                lh.removeByClass(music21.instrument.Instrument)
            
                piano_lh = music21.instrument.ElectricPiano()
                lh.insert(0, piano_lh)

                if len(split_scores) > 1:
                    _output_file = output_dir / (input_file.stem + f"_{i}{constants.MIDI_EXTENSION}")
                else:
                    _output_file = output_dir / (input_file.stem + constants.MIDI_EXTENSION)

                metadata_file: FilePath = metadata_dir / (_output_file.stem + constants.METADATA_EXTENSION)
                
                output_files.append(_output_file)

                score.write("midi", fp=_output_file)
                
                with open(metadata_file, "w") as f:
                    json.dump(metadata.tokenised_metadata.to_dict(), f, indent=4)

            # note that part of this score (one exercise) couldn't be processed
            except Exception as e:
                _warnings.append(f"{i}. exercise failed processing: {Generics.str_error(e)}\n")
                warnings.warn(_warnings[-1])
                error_counter += 1

            if error_counter == len(split_scores):
                raise ValueError(f"All {len(split_scores)} exercises failed processing: {_warnings}")

        return output_files

    def conversion(self, input_file: FilePath, output_dir: FilePath) -> List[ConversionOutcome]:
        return Generics.generic_music21_conversion(input_file, output_dir, self.music21_func)


class midi_to_tokens(SingleFileConversionFunction):
    
    def __init__(self, tokeniser: MyTokeniser):
        """
        Initialise a new instance of the midi_to_tokens class.
        
        Parameters:
            tokeniser (MyTokeniser): An instance of MyTokeniser used for encoding MIDI files into tokens.
        """
        self.tokeniser = tokeniser

    def skip_single_file(self, input_file, output_dir):
        return Generics.same_name_skip(input_file, output_dir)

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> List[ConversionOutcome]:    
        import json

        with input_file.parent.joinpath(constants.data_pipeline_constants.METADATA_DIR_NAME, input_file.stem + constants.METADATA_EXTENSION).open() as f:
            metadata = json.load(f)

        if t := Generics.invalid_metadata_skip(input_file, metadata, self.tokeniser):
            return t

        try: 
            jso = self.tokeniser.encode_with_metadata(input_file, metadata)
            
        except Exception as e:
            return [ConversionOutcome(
                input_file=input_file,
                successful=False,
                error_message=Generics.str_error(e),
                halt=False
            )]
        
        else:
            Generics.clear_n_terminal_lines(3)
            
            output_path = output_dir.joinpath(input_file.stem + constants.TOKENS_EXTENSION)
            
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(jso, f)

            return [ConversionOutcome(
                input_file=input_file,
                output_files=[output_path]
            )]


    def clean_up(self, input_file, output_dir):
        pass


class tokens_to_midi(SingleFileConversionFunction):
    
    def __init__(self, tokeniser: MyTokeniser):
        """
        Initialise a new instance of the tokens_to_midi class.
        
        Parameters:
            tokeniser (MyTokeniser): An instance of MyTokeniser used for decoding tokens into MIDI files.
        """
        self.tokeniser = tokeniser

    def skip_single_file(self, input_file, output_dir):
        return Generics.same_name_skip(input_file, output_dir)

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> List[ConversionOutcome]:
        import json

        try:
            with input_file.open("r", encoding="utf-8") as f:
                jso = json.load(f)
            
            output_path = output_dir / (input_file.stem + ".midi")
            
            
            midi.write("midi", fp=output_path)

        except Exception as e:
            return [ConversionOutcome(
                input_file=input_file,
                successful=False,
                error_message=Generics.str_error(e),
                halt=False
            )]

        try:
            midi = self.tokeniser.decode(jso)
        
        except Exception as e:
            return [ConversionOutcome(
                input_file=input_file,
                successful=False,
                error_message=Generics.str_error(e),
                halt=False
            )]
        
        else:
            

            return [ConversionOutcome(
                input_file=input_file,
                output_files=[output_path],
                successful=True
            )]



if __name__ == "__main__":
    pass
