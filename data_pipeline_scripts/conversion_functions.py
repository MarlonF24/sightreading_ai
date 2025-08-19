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
    import music21
    
    @staticmethod
    def transpose_score_to_key_sig(score: music21.stream.Score, target_key_sharps: int) -> music21.stream.Score:
        """
        Transpose the given score to the target key signature.

        :param score: The score to transpose.
        :type score: Score
        :param target_key_sharps: The target key signature in sharps.
        :type target_key_sharps: int
        :return: The transposed score.
        :rtype: Score
        """
        import copy
        
        # unfortunately the music21 score.transpose does not let you specify the desired key_signature, instead it will take an interval and can pick key_sigs like G# instead of enharmonic Ab. So, here we change the key_sig manually first, and then transpose the notes afterwards. music21 says that it will adapt the note transpositionto the given key_signature 

        original_key_sigs = Metadata(score).key_signatures
        new_score = copy.deepcopy(score)
                        
        new_metadata = Metadata(new_score)
        new_key_sigs = new_metadata.key_signatures
        for new_key_sig in new_key_sigs:
            new_key_sig.sharps = target_key_sharps

        interval = Generics.music21.interval.Interval(original_key_sigs[0].asKey('major').tonic, new_key_sigs[0].asKey('major').tonic)
        # for some reason only giving it the halfsteps makes it consider the key to determine the best accidental
        for n in new_metadata.notes:
            n.transpose(interval.semitones, inPlace=True)

        return new_score

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
    def invalid_metadata_skip(input_file: FilePath, tokenised_data: Metadata | Metadata.TokenisedMetadata | Dict[str, str], tokeniser: MyTokeniser) -> Optional[ConversionOutcome]:
        """Check if the tokenised data contains valid metadata.

        :param input_file: The input file being processed.
        :type input_file: FilePath
        :param tokenised_data: The tokenised data to check.
        :type tokenised_data: Metadata | Metadata.TokenisedMetadata | Dict[str, str]
        :param tokeniser: The tokeniser used for validation.
        :type tokeniser: MyTokeniser
        :return: A ConversionOutcome object indicating the result of the validation, or None if valid.
        :rtype: Optional[ConversionOutcome]
        """
        b, err = tokeniser.valid_metadata(tokenised_data)
        if not b:
            return ConversionOutcome(
                input_file=input_file,
                skipped=True,
                error_message=f"Invalid metadata: {err}",
            )

        return None

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
    def same_name_skip(input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        """
        This function checks if there are any existing files in the output dir whose name starts with the name of the input file.
        If such files exist, it returns a ConversionOutcome object indicating that the conversion was skipped.
        If no matching files are found, it returns None.

        Parameters:
            input_file (FilePath): The input file for which to check for matching files in the output dir.
            output_dir (DirPath): The dir in which to search for matching files.

        Returns:
            Optional[ConversionOutcome]: A ConversionOutcome object if matching files are found, or None otherwise.
        """
        res: List[FilePath] = Generics.find_same_name_outcomes(input_file, output_dir)

        return ConversionOutcome(input_file=input_file, output_files=res, skipped=True) if res else None

    SingleFileInterpreter = Callable[[int, str, str, FilePath, DirPath], ConversionOutcome]
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
        ) -> ConversionOutcome: ...
         
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
        ) -> ConversionOutcome | List[ConversionOutcome]: 
        """
        Executes a subprocess command and interprets the output.

        :param input_path: The input file or dir for the conversion process.
        :type input_path: FilePath | DirPath
        :param output_dir: The dir where the output files will be saved.
        :type output_dir: DirPath
        :param command: The command to be executed as a list of strings.
        :type command: List[str]
        :param interpreter: The function that interprets the subprocess output.
        :type interpreter: SingleFileInterpreter | BatchInterpreter
        :param batch: A flag indicating whether the conversion is batch or single file.
        :type batch: bool
        :return: The outcome of the conversion process - single ConversionOutcome for single file, list for batch.
        :rtype: ConversionOutcome | List[ConversionOutcome]
        """

        import subprocess
        if batch and not isinstance(input_path, DirPath):
            raise TypeError("input_path must be a DirPath when batch is True.")

        try:
            result = subprocess.run(command, capture_output=True, text=True)
                
        except Exception as e:
            if batch:
                res = [ConversionOutcome(
                    input_file=input_file,
                    successful=False,
                    error_message=Generics.str_error(e),
                ) for input_file in input_path.glob("*.pdf")]
            
            else:
                res = ConversionOutcome(
                    input_file=input_path,
                    successful=False,
                    error_message=Generics.str_error(e),
                )
            
            return res

        else:
            return interpreter(result.returncode, result.stderr, result.stdout, input_path, output_dir)  

    @staticmethod
    def generic_music21_conversion(
        input_file: FilePath, 
        output_dir: FilePath, 
        func: (Callable[[FilePath, DirPath], List[FilePath]])
        ) -> ConversionOutcome:
        """
        This function is a generic utility for converting music files using the music21 library.
        It catches warnings during the conversion process and handles exceptions.

        Parameters:
            input_file (FilePath): The path to the input music file.
            output_dir (DirPath): The path where the converted music file will be saved.
            func (Callable[[FilePath, DirPath], List[FilePath]]): The function that performs the actual conversion.

        Returns:
            ConversionOutcome: A ConversionOutcome object representing the outcome of the conversion process.
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                output_files = func(input_file, output_dir)  
            
            except Exception as e:
                return ConversionOutcome(
                    input_file=input_file,
                    successful=False,
                    error_message=Generics.str_error(e)
                )

            else:
                return ConversionOutcome(
                    input_file=input_file,
                    output_files=output_files,
                    successful=True,
                    warning_messages=[str(w.message) for w in caught_warnings]
                )


class pdf_preprocessing(SingleFileConversionFunction):
    """
    A class representing a single file conversion function for preprocessing PDF files.
    """
    def __init__(self, pages_per_split: int = 1):
        self.pages_per_split = pages_per_split

    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file, output_dir)
    
    def clean_up(self, input_file: FilePath, output_dir: DirPath) -> None:
        pass

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:
        """
        Convert a PDF file by splitting it into multiple files based on pages_per_split.

        Parameters:
            input_file (FilePath): The input PDF file to convert.
            output_dir (DirPath): The directory where the output files will be saved.

        Returns:
            ConversionOutcome: The outcome of the conversion process.
        """
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
            return ConversionOutcome(
                input_file=input_file,
                successful=False,
                error_message=Generics.str_error(e),
                halt=False
            )
        
        else:
            return ConversionOutcome(
                input_file=input_file,
                output_files=output_paths,
                successful=True
            )


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
            audiveris_app_dir (DirPath): The directory containing the Audiveris application's jar files.
            do_clean_up (bool): A flag indicating whether to perform cleanup, defaults to True
        """        
        self._audiveris_app_dir = audiveris_app_dir
        self.do_clean_up = do_clean_up

    @property
    def audiveris_app_dir(self) -> DirPath:
        return self._audiveris_app_dir

    @cached_property
    def classpath(self) -> str:   
        return ";".join([str(jar_file) for jar_file in self.audiveris_app_dir.glob(f"*{constants.JAR_EXTENSION}")])

    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file, output_dir)

    def single_file_interpreter(self, returncode: int, stdout: str, stderr: str, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:
        """
        Interprets the output of a single file conversion process.

        Parameters:
            returncode (int): The return code of the conversion process.
            stdout (str): The standard output of the conversion process.
            stderr (str): The standard error output of the conversion process.
            input_file (FilePath): The input file for the conversion process.
            output_dir (DirPath): The dir where the output files are saved.

        Returns:
            ConversionOutcome: A ConversionOutcome object representing the outcome of the conversion process.
        """

        critical_messages = ["Could not export since transcription did not complete successfully",
                             "resolution is too low", "FATAL"]
        
        if stdout:
            for temp in critical_messages:
                if temp in stdout:
                    return ConversionOutcome(
                        input_file=input_file,
                        successful=False,
                        error_message=temp
                    )

        if stderr:
            for temp in critical_messages:
                if temp in stderr:
                    return ConversionOutcome(
                        input_file=input_file,
                        successful=False,
                        error_message=temp
                    )        

        output_files = [file for file in output_dir.glob(f"{input_file.stem}*{constants.MXL_EXTENSION}")]
        
        if not output_files:
            return ConversionOutcome(
                input_file=input_file,
                successful=False,
                error_message=f"No output files found in {output_dir} for {input_file.stem}. Some uncaught error might have occurred. Check audiveris log for details and add any indication of the error to the critical_messages in {self.__class__.__name__} in {self.__class__.__module__}."
            )
        
        if returncode == 0:
            warnings = [f"{line.strip().split(']')[1]}" for line in stdout.split('\n') if line.strip().startswith("WARN")]

            return ConversionOutcome(input_file=input_file, output_files=output_files, successful=True, 
            warning_messages=warnings)
        
        return ConversionOutcome(input_file=input_file, output_files=output_files, successful=False, 
        error_message=stderr)
    
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
                res.append(self.single_file_interpreter(0, stdout_section, "", input_file, output_dir))
            
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

    def single_file_conversion(self, input_file: FilePath, output_dir: DirPath, overwrite: bool = True) -> ConversionOutcome:  
        """
        Convert a single PDF file to MXL format using Audiveris.

        Parameters:
            input_file (FilePath): The input PDF file to convert.
            output_dir (DirPath): The directory where the output files will be saved.
            overwrite (bool): Whether to overwrite existing files, defaults to True.

        Returns:
            ConversionOutcome: The outcome of the conversion process.
        """
        from data_pipeline_scripts import enhance_resolution 
        # resolution.convert(input_file, input_file)
        command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", self.classpath, "Audiveris", "-batch", "-export", "-output", str(output_dir), "--", str(input_file)]
          
        return Generics.generic_subprocess_conversion(
            input_path=input_file, 
            output_dir=output_dir, 
            command=command, 
            interpreter=self.single_file_interpreter, 
            batch=False)

    def batch_conversion(self, input_dir: DirPath, output_dir: DirPath, overwrite: bool = True) -> List[ConversionOutcome]:
        """
        Convert multiple PDF files to MXL format using Audiveris in batch mode.

        Parameters:
            input_dir (DirPath): The directory containing input PDF files.
            output_dir (DirPath): The directory where the output files will be saved.
            overwrite (bool): Whether to overwrite existing files, defaults to True.

        Returns:
            List[ConversionOutcome]: A list of ConversionOutcome objects representing the outcome of the conversion process.
        """
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
    
    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file, output_dir)

    def music21_func(self, input_file: FilePath, output_dir: DirPath) -> List[FilePath]:
        """
        This function is responsible for converting a MXL file to a MusicXML file using the music21 library.

        Parameters:
            input_file (FilePath): The path to the input MXL file.
            output_dir (DirPath): The path where the converted MusicXML file will be saved.

        Returns:
            List[FilePath]: A list containing the path to the converted MusicXML file.
        """
        import music21

        score = music21.converter.parse(input_file)
        output_path = output_dir / (input_file.stem + constants.MUSICXML_EXTENSION)
        score.write('musicxml', fp=output_path)
        return [output_path]
    
    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:
        """
        Convert a MXL file to MusicXML format using music21.

        Parameters:
            input_file (FilePath): The input MXL file to convert.
            output_dir (DirPath): The directory where the output files will be saved.

        Returns:
            ConversionOutcome: The outcome of the conversion process.
        """
        return Generics.generic_music21_conversion(input_file, output_dir, self.music21_func)

    
class mxl_to_musicxml_unzip(SingleFileConversionFunction):
    """
    This class is responsible for converting a MXL file to a MusicXML file via unzipping.

    Inherits from SingleFileConversionFunction.
    """
    
    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file=input_file, output_dir=output_dir)
    
    def clean_up(self, input_file: FilePath, output_dir: DirPath):
        import shutil

        for file in output_dir.iterdir():
                if file.suffix != '.musicxml':
                    file.unlink() if file.is_file() else shutil.rmtree(file)
    
    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:    
        """
        Convert a MXL file to MusicXML format by unzipping.

        Parameters:
            input_file (FilePath): The input MXL file to convert.
            output_dir (DirPath): The directory where the output files will be saved.

        Returns:
            ConversionOutcome: The outcome of the conversion process.
        """
        import zipfile, os
        
        try:
            with zipfile.ZipFile(input_file, 'r') as archive:
                name = archive.namelist()[0]
                extracted_path = archive.extract(name, output_dir)
        
        except Exception as e:
            return ConversionOutcome(
                input_file=input_file, 
                successful=False,
                error_message=Generics.str_error(e),
                halt=False)
        
        else:
            new_path = output_dir / (input_file.stem + ".musicxml")

            if new_path.exists():
                new_path.unlink()

            os.rename(extracted_path, new_path)
            
            return ConversionOutcome( 
                input_file=input_file, 
                output_files=[new_path],
                successful=True)


class mxl_to_midi(SingleFileConversionFunction):
    """
    This class is responsible for converting a MusicXML file to a MIDI file using the music21 library.
    It also extracts metadata from the parsed music score and saves it to a JSON file in a separate dir.

    Inherits from SingleFileConversionFunction.
    """

    def __init__(self, tokeniser: MyTokeniser, split: bool = True, transpose: bool = True):
        """Initialise a new instance of the mxl_to_midi class.

        :param tokeniser: An instance of MyTokeniser used for tokenising MIDI files later. So that we can skip files that do not match the tokeniser vocab by their metadata.
        :type tokeniser: MyTokeniser
        :param split: Whether to split the score into smaller segments at each bar end line.
        :type split: bool
        :param transpose: Whether to transpose the score to C major.
        :type transpose: bool
        """

        self.tokeniser = tokeniser
        self.split = split
        self.transpose = transpose

    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file=input_file, output_dir=output_dir)

    def clean_up(self, input_file: FilePath, output_dir: DirPath):
        pass

    def music21_func(self, input_file: FilePath, output_dir: DirPath) -> List[FilePath]:
        """
        This function is responsible for converting a MusicXML file to a MIDI file using the music21 library.
        It also extracts metadata from the parsed music score and saves it to a JSON file.

        Parameters:
            input_file (FilePath): The path to the input MusicXML file.
            output_dir (DirPath): The path where the converted MIDI file shall be saved. This address is changed to save the MIDI file and metadata in separate dirs inside the original dir address.

        Returns:
            List[FilePath]: A list of paths to the generated MIDI files.
        """
        from tokeniser.tokeniser import Metadata
        import music21, json, warnings

        metadata_dir: DirPath = output_dir / constants.data_pipeline_constants.METADATA_DIR_NAME
        metadata_dir.mkdir(parents=True, exist_ok=True)  # Create metadata dir if it doesn't exist

        score = music21.converter.parse(input_file)

        if not isinstance(score, music21.stream.Score):
            raise ValueError("Input file is not a valid MusicXML file.")
        
        # Split hand into different instrument programs to separate them for tokenisation later
        if len(score.parts) != 2:
            raise ValueError("Expected two staves (RH and LH), got something else.")

        lh = score.parts[1]

        lh.removeByClass(music21.instrument.Instrument)
            
        piano_lh = music21.instrument.ElectricPiano()
        lh.insert(0, piano_lh)

        score_stack = [score]
        
        # note that this also splits at every end-repeat line
        if self.split:
            measures = score.parts[0].getElementsByClass(music21.stream.Measure)
            splits = [0]  
            
            for i, measure in enumerate(measures, start=1):
                
                if t := measure.finalBarline:
                    if t.__class__ == music21.bar.Barline:
                        splits.append(i)
            
            if len(splits) == 1:
                warnings.warn("Input file does not contain any bar end lines. Either this is an exercise fragment or it is not properly formatted.")

            if not len(measures) in splits:
                splits.append(len(measures))

            score_stack = [score.measures(splits[i] + 1, splits[i + 1]) for i in range(len(splits) - 1)]
            print(f"Splitting completed.", end="\r")

        i = 0
        initial_length = len(score_stack)
        output_files: List[FilePath] = []
        error_counter = 0
        _warnings = []

        while score_stack:
            try:
                i += 1
                score = score_stack.pop(0)
                
                metadata = Metadata(score)
                
                if t := Generics.invalid_metadata_skip(input_file, metadata, self.tokeniser):
                    raise ValueError(t.error_message)
                
                # transpose everything to C_major so that model doesnt have to learn how key signatures work
                score = Generics.transpose_score_to_key_sig(score, 0) if self.transpose else score
                
                if initial_length > 1:
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

        if error_counter == i:
            raise ValueError(f"All {i} exercises failed processing: {_warnings}")

        return output_files

    def conversion(self, input_file: FilePath, output_dir: FilePath) -> ConversionOutcome:
        """
        Convert a MusicXML file to MIDI format with metadata extraction.

        Parameters:
            input_file (FilePath): The input MusicXML file to convert.
            output_dir (DirPath): The directory where the output files will be saved.

        Returns:
            ConversionOutcome: The outcome of the conversion process.
        """
        return Generics.generic_music21_conversion(input_file, output_dir, self.music21_func)


class midi_to_tokens(SingleFileConversionFunction):
    """
    A class responsible for converting MIDI files to token sequences using a tokeniser.

    Inherits from SingleFileConversionFunction.
    """
    
    def __init__(self, tokeniser: MyTokeniser):
        """
        Initialise a new instance of the midi_to_tokens class.
        
        Parameters:
            tokeniser (MyTokeniser): An instance of MyTokeniser used for encoding MIDI files into tokens.
        """
        self.tokeniser = tokeniser

    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        import json
        
        if temp := Generics.same_name_skip(input_file, output_dir):
            with temp.output_files[0].open() as f:
                tokens = json.load(f)

            if tokens[constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY] == self.tokeniser.hexa_hash:
                return temp

        return None

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:    
        """
        Convert a MIDI file to token sequence format.

        Parameters:
            input_file (FilePath): The input MIDI file to convert.
            output_dir (DirPath): The directory where the output files will be saved.

        Returns:
            ConversionOutcome: The outcome of the conversion process.
        """
        import json, shutil

        with input_file.parent.joinpath(constants.data_pipeline_constants.METADATA_DIR_NAME, input_file.stem + constants.METADATA_EXTENSION).open() as f:
            metadata = json.load(f)

        if t := Generics.invalid_metadata_skip(input_file, metadata, self.tokeniser):
            return t

        try: 
            jso = self.tokeniser.encode_with_metadata(input_file, metadata)
            
        except Exception as e:
            return ConversionOutcome(
                input_file=input_file,
                successful=False,
                error_message=Generics.str_error(e),
                halt=False
            )
        
        else:
            # Get terminal width (fallback to 80 if not available)
            try:
                terminal_width = shutil.get_terminal_size().columns
            except (AttributeError, OSError):
                terminal_width = 80
            
            # Calculate how many lines the filename will take
            lines_needed = (len(input_file.name) + len("") + terminal_width - 1) // terminal_width  # Ceiling division
            
            # Clear the calculated number of lines
            Generics.clear_n_terminal_lines(lines_needed)
            
            output_path = output_dir.joinpath(input_file.stem + constants.TOKENS_EXTENSION)
            
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(jso, f)

            return ConversionOutcome(
                input_file=input_file,
                output_files=[output_path],
                successful=True
            )

    def clean_up(self, input_file: FilePath, output_dir: DirPath):
        pass


class tokens_to_midi(SingleFileConversionFunction):
    """
    A class responsible for converting token sequences back to MIDI files.

    Inherits from SingleFileConversionFunction.
    """
    import music21, symusic, dataclasses

    metadata_length = len(dataclasses.fields(Metadata.TokenisedMetadata))
    
    def __init__(self, tokeniser: MyTokeniser):
        """
        Initialise a new instance of the tokens_to_midi class.
        
        Parameters:
            tokeniser (MyTokeniser): An instance of MyTokeniser used for decoding tokens into MIDI files.
        """
        self.tokeniser = tokeniser

    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file, output_dir)
    
    def clean_up(self, input_file: FilePath, output_dir: DirPath):
        pass

    # deprecated
    @staticmethod
    def symusic_to_music21(score_tick: symusic.core.ScoreTick = None) -> music21.stream.Score:
        import music21
        m21_score = music21.stream.Score()
        rh_part = music21.stream.Part()
        lh_part = music21.stream.Part()
        
        tokens_to_midi.symusic.core.TrackTick

        for track in score_tick.tracks:
            print(track)
            if not isinstance(track, tokens_to_midi.symusic.core.TrackTick):
                raise TypeError(f"Expected symusic.core.TrackTick, got {type(track)}")
            # remember, we set rh to program 0 and lh to program 2
            if track.program == 0:
                part = rh_part
            else:
                part = lh_part

            for tick in track.notes:
                if not isinstance(tick, tokens_to_midi.symusic.core.NoteTick):
                    raise TypeError(f"Expected symusic.core.NoteTick, got {type(tick)}")

                note = music21.note.Note(tick.pitch, quarterLength=tick.duration)

        for event in score_tick:
            if event.is_chord:
                c = music21.chord.Chord(event.pitches, quarterLength=event.duration)
                part.append(c)
            else:
                n = music21.note.Note(event.pitch, quarterLength=event.duration)
                part.append(n)

        m21_score.insert(0, rh_part)
        m21_score.insert(1, lh_part)
        return m21_score

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:
        """
        Convert a token sequence file back to MIDI format.

        Parameters:
            input_file (FilePath): The input token sequence file to convert.
            output_dir (DirPath): The directory where the output files will be saved.

        Returns:
            ConversionOutcome: The outcome of the conversion process.
        """
        import json, miditok, symusic, dataclasses
        metadata_dir = output_dir / constants.data_pipeline_constants.METADATA_DIR_NAME
        metadata_dir.mkdir(parents=True, exist_ok=True)

        try:
            with input_file.open("r", encoding="utf-8") as f:
                jso = json.load(f)

            if (a := self.tokeniser.hexa_hash) != (b := jso[constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY]):
                return ConversionOutcome(
                    input_file=input_file,
                    successful=False,
                    error_message=f"Tokeniser hexa hash mismatch: {a} != {b}. The model that generated these tokens has a different tokeniser than the one given.",
                    halt=True
                )
            
            ids = jso[constants.tokeniser_constants.TOKENS_INPUT_IDS_KEY]

            tok_seq = miditok.TokSequence(ids=ids)
            
            if self.tokeniser.is_trained:
                tok_seq.are_ids_encoded = True
                self.tokeniser.decode_token_ids(tok_seq)

            self.tokeniser.complete_sequence(tok_seq)
            
            if "Bar_None" not in tok_seq.tokens:
                raise ValueError("The token sequence does not contain a Bar_None token. This is required for the conversion to MIDI.")
            print(tok_seq.tokens)
            
            clean_seq = miditok.TokSequence(tokens=tok_seq.tokens[tokens_to_midi.metadata_length + 1:])

            output_path = output_dir / (input_file.stem + ".midi")

            if output_path.exists():
                output_path.unlink()

            score = self.tokeniser.decode(clean_seq)
            
            # set both hands to piano program as we split them onto different programs before
            for track in score.tracks:
                track.program = 0 
                track.name = "Piano"

            if not isinstance(score, symusic.core.ScoreTick):
                raise ValueError("The decoded score is not a valid symusic Score object.")
            
            score.dump_midi(output_path.as_posix())

            with open(metadata_dir / (output_path.stem + constants.METADATA_EXTENSION), "w") as f:
                json.dump({constants.tokeniser_constants.TOKENS_KEY_SIGNATURE_KEY: jso[constants.tokeniser_constants.TOKENS_KEY_SIGNATURE_KEY], constants.tokeniser_constants.TOKENS_METADATA_KEY: jso[constants.tokeniser_constants.TOKENS_METADATA_KEY]}, f, indent=4)

        except Exception as e:
            return ConversionOutcome(
                input_file=input_file,
                successful=False,
                error_message=Generics.str_error(e),
                halt=False
            )

        else:
            return ConversionOutcome(
                input_file=input_file,
                output_files=[output_path],
                successful=True
            )


class midi_to_musicxml(SingleFileConversionFunction):
    """
    This class is responsible for converting a MIDI file to a MusicXML file using the music21 library.
    
    Inherits from SingleFileConversionFunction.
    """
    
    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file, output_dir)

    def clean_up(self, input_file: FilePath, output_dir: DirPath):
        pass

    def music21_func(self, input_file: FilePath, output_dir: DirPath) -> List[FilePath]:
        """
        Convert a MIDI file to MusicXML format using music21.

        Parameters:
            input_file (FilePath): The path to the input MIDI file.
            output_dir (DirPath): The directory where the converted MusicXML file will be saved.

        Returns:
            List[FilePath]: A list containing the path to the converted MusicXML file.
        """
        import music21, json
        score = music21.converter.parse(input_file)
        
        with open(input_file.parent / constants.data_pipeline_constants.METADATA_DIR_NAME / (input_file.stem + constants.METADATA_EXTENSION), "r") as f:
            jso = json.load(f)
            key_signature = jso[constants.tokeniser_constants.TOKENS_KEY_SIGNATURE_KEY]
        
        # transpose everything to the original key signature
        score = Generics.transpose_score_to_key_sig(score, key_signature)

        output_path = output_dir / (input_file.stem + constants.MUSICXML_EXTENSION)
        score.write('musicxml', fp=output_path)
        return [output_path]

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:
        """
        Convert a MIDI file to MusicXML format.

        Parameters:
            input_file (FilePath): The input MIDI file to convert.
            output_dir (DirPath): The directory where the output files will be saved.

        Returns:
            ConversionOutcome: The outcome of the conversion process.
        """
        return Generics.generic_music21_conversion(input_file, output_dir, self.music21_func)
        
        

if __name__ == "__main__":
    pass


