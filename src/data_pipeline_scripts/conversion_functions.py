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
        Transpose a music21 score to a specified key signature.
        
        This method manually sets the key signature first, then transposes all notes
        to match the new key. This approach avoids music21's default behavior of
        potentially choosing enharmonic equivalents (e.g., G# instead of Ab).
        
        Args:
            score: The music21 Score object to transpose
            target_key_sharps: The target key signature represented as number of sharps
                              (negative values indicate flats)
        
        Returns:
            A new transposed Score object with the specified key signature
            
        Note:
            The original score is not modified; a deep copy is created and returned.
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
        Generate a standardized string representation of an exception.
        
        This utility function creates a consistent error message format for logging
        and debugging purposes throughout the data pipeline.
        
        Args:
            e: The exception object to convert to string
            
        Returns:
            A formatted string in the format "ExceptionType: exception message"
            
        Example:
            >>> str_error(ValueError("Invalid input"))
            'ValueError: Invalid input'
        """
        return f"{type(e).__name__}: {str(e)}"
    
    @staticmethod
    def clear_n_terminal_lines(n: int = 1):
        """
        Clear the last n lines from the terminal output.
        
        This function uses ANSI escape sequences to move the cursor up and clear
        lines, useful for updating progress displays without cluttering the terminal.
        
        Args:
            n: The number of lines to clear from the terminal. Defaults to 1.
            
        Note:
            This function works with terminals that support ANSI escape sequences.
            On unsupported terminals, the escape sequences may be displayed as text.
        """
        import sys
        
        sys.stdout.write(n * "\033[F\033[K")
        sys.stdout.flush()

    @staticmethod
    def mute_decorator(func: Callable):
        """
        Decorator that suppresses stdout and stderr output from the decorated function.
        
        This decorator captures all output to stdout and stderr during function execution,
        preventing it from appearing in the terminal. Useful for silencing verbose
        library functions while preserving their return values.
        
        Args:
            func: The function to decorate and mute
            
        Returns:
            A wrapper function that executes the original function silently
            
        Example:
            @mute_decorator
            def verbose_function():
                print("This won't appear")
                return "but this is returned"
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
        """
        Validate tokenized metadata against the tokenizer's requirements.
        
        This function checks if the provided tokenized data contains valid metadata
        according to the tokenizer's validation rules. If invalid, it returns a
        ConversionOutcome indicating the file should be skipped.
        
        Args:
            input_file: The file being processed
            tokenised_data: The tokenized metadata to validate. Can be Metadata object,
                           TokenisedMetadata object, or dictionary
            tokeniser: The tokenizer instance used for validation
            
        Returns:
            ConversionOutcome with skip=True and error details if invalid,
            None if the metadata is valid
            
        Note:
            This is typically used early in conversion pipelines to filter out
            files with incompatible metadata before expensive processing.
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
        Find all output files that correspond to a given input file.
        
        This function searches for files in the output directory whose names start
        with the input file's stem (filename without extension). It matches both
        exact stems and stems with suffixes (e.g., file_001, file_part1).
        
        Args:
            input_file: The input file to find corresponding outputs for
            output_dir: The directory to search for matching output files
            
        Returns:
            List of file paths that match the input file's naming pattern
            
        Example:
            For input file "song.pdf", this might return:
            ["song.mxl", "song_001.midi", "song_part1.json"]
        """
        import re

        return [file for file in output_dir.glob(f"{input_file.stem}.*") if re.match(rf"{input_file.stem}(_\d+)?", file.name)] # either find same stem with different extension or _<digits>

    @staticmethod
    def same_name_skip(input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        """
        Check if output files already exist for the given input file.
        
        This function implements a skip mechanism to avoid reprocessing files
        that have already been converted. It searches for existing output files
        with names matching the input file's pattern.
        
        Args:
            input_file: The input file to check for existing outputs
            output_dir: The directory containing potential output files
            
        Returns:
            ConversionOutcome with skip=True and list of existing files if found,
            None if no matching output files exist
            
        Note:
            This is commonly used to implement incremental processing where
            only new or modified files are processed.
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
        batch: bool,
        clean_up: Optional[SingleFileCleanUp | BatchCleanUp] = None
    ) -> ConversionOutcome | List[ConversionOutcome]:
        """
        Execute a subprocess command and interpret its results for file conversion.
        
        This generic function provides a standardized way to run external conversion
        tools via subprocess, handle their output, and interpret results using
        provided interpreter functions. It supports both single-file and batch processing.
        
        The function executes the given command, captures stdout/stderr, and delegates
        result interpretation to the provided interpreter function. Exception handling
        creates appropriate ConversionOutcome objects with error details.
        
        Args:
            input_path: Path to input file (single mode) or directory (batch mode).
                       Must be a directory when batch=True.
            output_dir: Directory where output files will be saved
            command: List of command arguments to execute (e.g., ["java", "-jar", "tool.jar"]).
                    Command should be ready to run without modification.
            interpreter: Function to interpret subprocess results and create ConversionOutcome(s).
                        Signature for single file: (int, str, str, FilePath, DirPath) -> ConversionOutcome
                        Signature for batch: (int, str, str, DirPath, DirPath) -> List[ConversionOutcome]
            batch: If True, process directory of files; if False, process single file.
                  When True, input_path must be a directory.
            
        Returns:
            Single ConversionOutcome for file processing (batch=False), or 
            List of ConversionOutcomes for batch processing (batch=True)
            
        Raises:
            TypeError: If batch=True but input_path is not a directory
            
        Note:
            - The interpreter function receives (returncode, stderr, stdout, input_path, output_dir)
            - All subprocess exceptions are caught and converted to failed ConversionOutcome objects
            - For batch processing with exceptions, a ConversionOutcome is created for each file
              in the input directory using glob("*")
            - The command is executed with capture_output=True and text=True for string handling
            
        Example:
            >>> command = ["java", "-jar", "converter.jar", str(input_file)]
            >>> outcome = generic_subprocess_conversion(
            ...     input_file, output_dir, command, my_interpreter, False
            ... )
        """

        import subprocess
        if batch and not input_path.is_dir():
            raise TypeError("input_path must be a directory when batch is True.")

        try:
            result = subprocess.run(command, capture_output=True, text=True)
                
        except Exception as e:
            if batch:
                res = [ConversionOutcome(
                    input_file=input_file,
                    successful=False,
                    error_message=Generics.str_error(e),
                ) for input_file in input_path.glob("*")]
            
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
        Execute a music21-based conversion with standardized error and warning handling.
        
        This wrapper function provides consistent error handling and warning capture
        for music21-based conversions. It catches music21-specific warnings and
        exceptions, returning standardized ConversionOutcome objects.
        
        Args:
            input_file: Path to the input music file to convert
            output_dir: Directory where converted files will be saved
            func: Conversion function that performs the actual music21 operations.
                  Must accept (FilePath, DirPath) and return List[FilePath]
        
        Returns:
            ConversionOutcome object containing:
            - Success/failure status
            - List of output files (if successful)
            - Error message (if failed)
            - Warning messages captured during conversion
            
        Note:
            This function is designed to be extended in the future to handle
            specific music21 error types and provide more detailed error information.
            All warnings are captured and included in the outcome for debugging.
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
    Preprocesses PDF files by splitting them into smaller chunks based on page count.
    
    This converter splits multi-page PDF files into separate PDF files, each containing
    a specified number of pages. Useful for breaking down large sheet music collections
    or preparing PDFs for batch processing with tools that work better on smaller files.
    
    Attributes:
        pages_per_split (int): Number of pages to include in each output PDF file
        
    Example:
        >>> preprocessor = pdf_preprocessing(pages_per_split=2)
        >>> outcome = preprocessor.conversion(input_pdf, output_dir)
    """
    
    def __init__(self, pages_per_split: int = 1):
        """
        Initialize the PDF preprocessing converter.
        
        Args:
            pages_per_split: Number of pages to include in each split PDF file.
                           Defaults to 1 (one page per output file).
        """
        self.pages_per_split = pages_per_split

    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file, output_dir)
    
    def clean_up(self, input_file: FilePath, output_dir: DirPath) -> None:
        pass

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:
        from PyPDF2 import PdfReader, PdfWriter
        
        try:
            reader = PdfReader(str(input_file))
            total_pages = len(reader.pages)

            output_paths = []
            
            num_splits = (total_pages + self.pages_per_split - 1) // self.pages_per_split  # Ceiling division

            for split_num in range(num_splits):
                start_page = split_num * self.pages_per_split
                end_page = min(start_page + self.pages_per_split, total_pages)

                writer = PdfWriter()
                
                for page_num in range(start_page, end_page):
                    writer.add_page(reader.pages[page_num])
                
                stem = input_file.stem
                output_filename = f"{stem}_part_{split_num + 1:03d}.pdf"
                output_path = output_dir / output_filename
                
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
    Converts PDF sheet music files to MXL format using Audiveris optical music recognition.
    
    This converter uses the Audiveris OMR engine to analyze PDF images of sheet music
    and convert them to structured MXL (MusicXML) format. Supports both single-file
    and batch processing modes for efficient handling of multiple PDFs.
    
    The converter requires Audiveris to be installed and accessible via its JAR files.
    It handles Audiveris-specific error messages and warnings, providing detailed
    feedback about conversion success or failure.
    
    Attributes:
        audiveris_app_dir (DirPath): Directory containing Audiveris JAR files
        do_clean_up (bool): Whether to remove non-MXL files after conversion
        classpath (str): Computed Java classpath for Audiveris execution
        
    Note:
        Requires Java runtime environment and Audiveris OMR software.
        Performance depends on PDF image quality and complexity of musical notation.
        
    Example:
        >>> converter = pdf_to_mxl(Path("audiveris/app"), do_clean_up=True)
        >>> outcomes = converter.batch_conversion(input_dir, output_dir)
    """

    def __init__(self, audiveris_app_dir: DirPath, do_clean_up: bool = True):
        """
        Initialize the PDF to MXL converter.

        Args:
            audiveris_app_dir: Directory containing Audiveris application JAR files
            do_clean_up: Whether to remove non-MXL files after conversion
        """        
        self.audiveris_app_dir = audiveris_app_dir
        self.do_clean_up = do_clean_up


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
        command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", self.classpath, "Audiveris", "-batch", "-export", "-output", str(output_dir), "--", str(input_file)]
          
        return Generics.generic_subprocess_conversion(
            input_path=input_file, 
            output_dir=output_dir, 
            command=command, 
            interpreter=self.single_file_interpreter, 
            batch=False)

    def batch_conversion(self, input_dir: DirPath, output_dir: DirPath, overwrite: bool = True) -> List[ConversionOutcome]:
        input_files = [str(f) for f in input_dir.glob(f"*{constants.PDF_EXTENSION}")]

        command = ["java","--add-opens", "java.base/java.nio=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED", "-cp", self.classpath, "Audiveris", "-batch", "-export", "-output", str(output_dir), "--", *input_files]
        
        return Generics.generic_subprocess_conversion(
            input_path=input_dir, 
            output_dir=output_dir, 
            command=command, 
            interpreter=self.batch_interpreter, 
            batch=True)

# deprecated
class mxl_to_musicxml_music21(SingleFileConversionFunction):
    """
    Converts MXL (compressed MusicXML) files to uncompressed MusicXML using music21.
    
    **DEPRECATED**: This class is no longer recommended for use.
    
    Uses the music21 library to parse MXL files and write them as standard MusicXML.
    This approach provides music21's parsing capabilities but may be slower than
    direct unzipping for simple format conversion.
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
        return Generics.generic_music21_conversion(input_file, output_dir, self.music21_func)

# deprecated 
class mxl_to_musicxml_unzip(SingleFileConversionFunction):
    """
    Converts MXL files to MusicXML format by direct ZIP extraction.
    
    **DEPRECATED**: This class is no longer recommended for use.
    
    MXL files are essentially ZIP archives containing MusicXML files. This converter
    extracts the MusicXML content directly without music21 parsing, making it faster
    but potentially less robust for malformed files.
    """
    
    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file=input_file, output_dir=output_dir)
    
    def clean_up(self, input_file: FilePath, output_dir: DirPath):
        import shutil

        for file in output_dir.iterdir():
                if file.suffix != '.musicxml':
                    file.unlink() if file.is_file() else shutil.rmtree(file)
    
    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:     
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


class to_midi(SingleFileConversionFunction):
    """
    Converts music21-compatible files (MusicXML, MXL) to MIDI with metadata extraction.
    
    This converter processes musical scores through several stages:
    1. Parses input files using music21
    2. Refurbishes scores (handles piano parts, voice merging)
    3. Optionally splits scores at bar end lines into separate exercises
    4. Optionally transposes to C major for consistent training data
    5. Extracts and validates metadata using the provided tokenizer
    6. Saves MIDI files and corresponding JSON metadata
    
    The converter is designed for machine learning pipelines where consistent
    format and metadata are crucial for training music generation models.
    
    Attributes:
        tokenizer (MyTokenizer): Used for metadata validation and skipping incompatible files
        split (bool): Whether to split scores into separate exercises at bar lines
        transpose (bool): Whether to transpose all output to C major
        
    Note:
        - Handles 2-part and 4-part scores, converting them to standardized piano format
        - Left hand parts are assigned different MIDI programs for tokenizer distinction
        - Failed exercises within a score generate warnings but don't fail entire conversion
        
    Example:
        >>> converter = to_midi(my_tokenizer, split=True, transpose=True)
        >>> outcome = converter.conversion(musicxml_file, output_dir)
    """
    
    def __init__(self, tokeniser: MyTokeniser, split: bool = True, transpose: bool = True):
        """
        Initialize the MusicXML to MIDI converter.

        Args:
            tokeniser: Tokenizer instance for metadata validation and vocabulary checking
            split: Whether to split scores into separate exercises at bar end lines
            transpose: Whether to transpose all scores to C major
        """
        self.tokeniser = tokeniser
        self.split = split
        self.transpose = transpose

    def clean_up(self, input_file, output_dir):
        pass

    def skip_single_file(self, input_file, output_dir):
        return Generics.same_name_skip(input_file, output_dir)

    import music21
    @staticmethod
    def refurbish_score(score: music21.stream.Score, split: bool) -> list[music21.stream.Score]:
        """
        Prepare and standardize a music21 score for MIDI conversion.
        
        This method performs several transformations:
        - Ensures all parts use Piano instruments
        - Handles 4-part scores by merging voice parts into piano parts
        - Assigns different MIDI programs to left/right hands for tokenizer distinction
        - Optionally splits score into separate exercises at bar end lines
        - Inserts missing key signatures and time signatures via Metadata constructor
        
        Args:
            score: The music21 Score object to refurbish
            split: Whether to split the score at bar end lines
            
        Returns:
            List of refurbished Score objects (single score if not splitting)
            
        Raises:
            ValueError: If score has unexpected number of parts (not 2 or 4)
            ValueError: If attempting to merge parts with conflicting note data
            
        Note:
            - Right hand remains program 0 (Piano), left hand becomes program 2 (Electric Piano)
            - Splitting creates separate exercises that can be processed independently
            - Bar end lines include repeat markers and final barlines
        """
        import music21, warnings

        num_parts = len(score.parts)
        # Sometimes scores are corrupted, where a section has voice parts instead of piano. thus, we accept scores with 4 parts and try to merge the voice parts into the piano parts
        if num_parts == 2:
            for part in score.parts:
                if not all(isinstance(instrument, music21.instrument.Piano) for instrument in part.getInstruments()):
                    part.removeByClass(music21.instrument.Instrument)
                
                part.insert(0, music21.instrument.Piano())
                part.partName = "Piano"

        elif num_parts == 4:
            piano_parts: list[music21.stream.Part] = []
            non_piano_parts: list[music21.stream.Part] = []

            # here we rely on that both piano and non_piano parts are sorted as rh, lh
            for part in score.parts:
                if len(piano_parts) < 2 and isinstance(part.getInstrument(), music21.instrument.Piano):
                    piano_parts.append(part) 
                    # clean off any stray instruments in the part
                    for instrument in part.getInstruments():
                        if not isinstance(instrument, music21.instrument.Piano):
                            part.remove(instrument)
                else:
                    non_piano_parts.append(part)

            for j, non_piano_part in enumerate(non_piano_parts):
                corresponding_piano_part = piano_parts[j]

                score.remove(non_piano_part)

                if len(non_piano_measures := non_piano_part.getElementsByClass(music21.stream.Measure)) != len(piano_measures := corresponding_piano_part.getElementsByClass(music21.stream.Measure)):
                    raise ValueError("Attempted to merge non-piano parts into piano parts but they have different numbers of measures.")

                measure_zip = zip(non_piano_measures, piano_measures)
                
                for non_piano_measure, piano_measure in measure_zip:
                    if (temp := non_piano_measure.notes) and piano_measure.notes:
                        raise ValueError(f"Attempted to merge non-piano part into piano part but they both have notes in the same measure -> corrupt input data.")
                
                    if temp:
                        piano_measure.removeByClass(music21.note.GeneralNote)
                        piano_measure.append(list(non_piano_measure.notesAndRests))

        else:
            raise ValueError(f"{num_parts} parts found in score, {__class__}.refurbish_score can only attempt to refurbish scores with 2 or 4 staves.")



        # change left hand to different program so that our tokeniser can distinguish the hands when given the midi
        lh = score.parts[1]

        lh.removeByClass(music21.instrument.Instrument)
            
        piano_lh = music21.instrument.ElectricPiano()
        lh.insert(0, piano_lh)

        scores = [score]

        # note that this also splits at every end-repeat line
        if split:
            measures = score.parts[0].getElementsByClass(music21.stream.Measure)
            splits = [0]  
            
            for i, measure in enumerate(measures, start=1):
                
                if t := measure.finalBarline:
                    #if t.__class__ == music21.bar.Barline:
                    splits.append(i)
            
            if len(splits) == 1:
                warnings.warn("Input file does not contain any bar end lines. Either this is an exercise fragment or it is not properly formatted.")

            if not len(measures) in splits:
                splits.append(len(measures))

            scores = [score.measures(splits[i] + 1, splits[i + 1]) for i in range(len(splits) - 1)]
            # print(f"Splitting completed.", end="\r")

        for score in scores:
            Metadata(score) # this is to insert missing key or time signatures
        
        return scores

    
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
        metadata_dir.mkdir(parents=True, exist_ok=True) 

        score = music21.converter.parse(input_file)
        
        if not isinstance(score, music21.stream.Score):
            raise ValueError("Input file is not a valid MusicXML file.")

        score_stack = to_midi.refurbish_score(score, split=self.split)

        i = 0
        initial_length = len(score_stack)
        output_files: List[FilePath] = []
        error_counter = 0
        _warnings = []

        while score_stack:
            try:
                i += 1
                score = score_stack.pop(0)
                
                # deleting repeats (remember the tokeniser's measure limit: this makes longer score accessible)
                for repeat in score.recurse().getElementsByClass(music21.repeat.RepeatMark):
                    repeat.activeSite.remove(repeat)

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
                    json.dump(metadata.tokenised_metadata.as_dict, f, indent=4)

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
    Converts MIDI files to tokenized sequences using a specified tokenizer.
    
    This converter transforms MIDI files into token sequences suitable for machine
    learning models. It loads corresponding metadata from JSON files and validates
    compatibility with the tokenizer's vocabulary before encoding.
    
    The converter implements intelligent skipping - if output tokens already exist
    and were generated by the same tokenizer (verified by hash), conversion is skipped
    for efficiency in incremental processing pipelines.
    
    Attributes:
        tokenizer (MyTokenizer): The tokenizer instance used for MIDI encoding
        
    Note:
        - Requires metadata JSON files to exist alongside MIDI files
        - Only processes files compatible with tokenizer vocabulary
        - Skips reprocessing if existing tokens match current tokenizer
        
    Example:
        >>> converter = midi_to_tokens(my_tokenizer)
        >>> outcome = converter.conversion(midi_file, output_dir)
    """
    
    def __init__(self, tokeniser: MyTokeniser, split: bool = True):
        """
        Initialize the MIDI to tokens converter.
        
        Args:
            tokeniser: Tokenizer instance for encoding MIDI files into token sequences
        """
        self.tokeniser = tokeniser
        self.split = split

    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        import json
        
        if temp := Generics.same_name_skip(input_file, output_dir):
            for file in temp.output_files:
                with file.open() as f:
                    tokens = json.load(f)

                if tokens[constants.tokeniser_constants.TOKENS_TOKENISER_HASH_KEY] != self.tokeniser.hexa_hash:
                   return None
            return temp
        return None

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:    
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
            terminal_width = shutil.get_terminal_size().columns

            # Calculate how many lines the filename will take
            lines_needed = 1 + ((len(str(input_file)) + len("[read_file(fs::path)] Input path:")) // terminal_width)  # Ceiling division

            if lines_needed > 1:
                pass

            # Clear the calculated number of lines
            Generics.clear_n_terminal_lines(lines_needed + 1) # + 1 for extra line: "[read_file(fs::path)] _wfopen_s returned: 0" 

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
    Converts tokenized sequences back to MIDI files using a specified tokenizer.
    
    This converter reverses the tokenization process, transforming token sequences
    back into playable MIDI files. It validates tokenizer compatibility, decodes
    token sequences, and generates both MIDI output and corresponding metadata.
    
    The conversion process includes:
    1. Loading and validating token files against tokenizer hash
    2. Decoding token IDs if from a trained model
    3. Completing sequences with missing tokens
    4. Converting to MIDI via symusic library
    5. Standardizing MIDI programs to Piano for both hands
    
    Attributes:
        tokenizer (MyTokenizer): The tokenizer instance used for decoding
        metadata_length (int): Number of metadata tokens to skip in sequences
        
    Note:
        - Requires token files to match the tokenizer that generated them
        - Expects "Bar_None" token presence for proper MIDI structure
        - Restores both hands to piano program after tokenizer's hand distinction
        
    Example:
        >>> converter = tokens_to_midi(my_tokenizer)
        >>> outcome = converter.conversion(token_file, output_dir)
    """
    
    def __init__(self, tokeniser: MyTokeniser):
        """
        Initialize the tokens to MIDI converter.
        
        Args:
            tokeniser: Tokenizer instance for decoding token sequences into MIDI
        """
        self.tokeniser = tokeniser

    def skip_single_file(self, input_file: FilePath, output_dir: DirPath) -> Optional[ConversionOutcome]:
        return Generics.same_name_skip(input_file, output_dir)
    
    def clean_up(self, input_file: FilePath, output_dir: DirPath):
        pass

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:
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

            clean_seq = miditok.TokSequence(tokens=tok_seq.tokens[1:])

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


class midi_to_mxl(SingleFileConversionFunction):
    """
    Converts MIDI files to MXL (MusicXML) format using music21.
    
    This converter transforms MIDI files back into structured musical notation format.
    It can optionally transpose the output back to original key signatures using
    metadata stored during the initial conversion process.
    
    The converter is typically used at the end of machine learning pipelines to
    convert generated MIDI back to readable sheet music format for evaluation
    or distribution.
    
    Attributes:
        transpose_to_desired_key (bool): Whether to restore original key signatures
        
    Note:
        - Requires metadata JSON files for key signature restoration
        - Output MXL files can be opened in music notation software
        - Transposition uses metadata stored during initial MIDI conversion
        
    Example:
        >>> converter = midi_to_mxl(transpose_to_desired_key=True)
        >>> outcome = converter.conversion(midi_file, output_dir)
    """
    
    def __init__(self, transpose_to_desired_key: bool):
        """
        Initialize the MIDI to MXL converter.

        Args:
            transpose_to_desired_key: Whether to transpose output back to original key signature
        """
        self.transpose_to_desired_key = transpose_to_desired_key

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

        if self.transpose_to_desired_key:
            with open(input_file.parent / constants.data_pipeline_constants.METADATA_DIR_NAME / (input_file.stem + constants.METADATA_EXTENSION), "r") as f:
                jso = json.load(f)
                key_signature = jso[constants.tokeniser_constants.TOKENS_KEY_SIGNATURE_KEY]

            # transpose everything to the original key signature
            score = Generics.transpose_score_to_key_sig(score, key_signature)

        output_path = output_dir / (input_file.stem + constants.MXL_EXTENSION)
        score.write('mxl', fp=output_path)
        return [output_path]

    def conversion(self, input_file: FilePath, output_dir: DirPath) -> ConversionOutcome:
        return Generics.generic_music21_conversion(input_file, output_dir, self.music21_func)
        
        

if __name__ == "__main__":
    pass


