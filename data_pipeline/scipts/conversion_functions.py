from file import *
import warnings, music21, subprocess, os
from typing import *

ConversionFunction = Callable[[File, File], ConversionOutcome]


class GenericConverter:
    @staticmethod
    def generic_subprocess_conversion(
        input_file: File, 
        output_file: File, 
        command: List[str], 
        interpreter: Callable[[subprocess.CompletedProcess, File, File], ConversionOutcome],
        clean_up: Optional[Callable[[File, File], None]] = None
        ) -> ConversionOutcome: 
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            return interpreter(result, input_file, output_file)
        except Exception as e:
            return ConversionOutcome(
                input_file=input_file,
                output_file=output_file,
                successful=False,
                error_message=f"Subprocess exception: {e}",
                go_on=False
            )
        finally:
            if clean_up:
                clean_up(input_file, output_file)  # clean up after conversion

    @staticmethod
    def generic_music21_conversion(
        input_file: File, 
        output_file: File, 
        func: ConversionFunction
        ) -> ConversionOutcome:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                func(input_file, output_file)
                # print([str(w.message) for w in caught_warnings])
                return ConversionOutcome(
                    input_file=input_file,
                    output_file=output_file,
                    successful=True,
                    warning_messages=tuple([str(w.message) for w in caught_warnings]),
                    go_on=True
                )
            except ValueError as e:
                return ConversionOutcome(
                    input_file=input_file,
                    output_file=output_file,
                    successful=False,
                    error_message=str(e),
                    go_on=True
                )
            except Exception as e:
                return ConversionOutcome(
                    input_file=input_file,
                    output_file=output_file,
                    successful=False,
                    error_message=str(e),
                    go_on=False  # serious crash
                )



def mxl_to_musicxml() -> ConversionFunction: 
    def music21_func(input_file: File, output_file: File):
        score = music21.converter.parse(input_file.path)
        score.write('musicxml', fp=output_file.path)
    
    def func(input_file: File, output_file: File) -> ConversionOutcome:
        return GenericConverter.generic_music21_conversion(input_file, output_file, music21_func)

    return func
    
def pdf_to_mxl(audiveris_path: Path, do_clean_up: bool = True) -> ConversionFunction:
    def interpreter(result: subprocess.CompletedProcess[str], input_file: File, output_file: File) -> ConversionOutcome:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()

        if result.returncode == 0:
            warnings = (stderr,) if "WARN" in stderr else ()
            return ConversionOutcome(input_file=input_file, output_file=output_file, successful=True, warning_messages=warnings, go_on=True)
        
        if "FATAL" in stderr:
            return ConversionOutcome(input_file=input_file, output_file=output_file,successful=False, error_message=stderr, go_on=False)
        
        return ConversionOutcome(input_file=input_file, output_file=output_file, successful=False, error_message=stderr, go_on=True)
    
    def clean_up(input_file: File, output_file: File) -> None:
        for root, _, files in os.walk(output_file.folder_path, topdown=False):
            for file in files:
                if not file.endswith('.mxl'):
                    os.remove(os.path.join(root, file))
                    

    def func(input_file: File, output_file: File) -> ConversionOutcome:
        command = [audiveris_path,"-batch","-export", "-output", output_file.folder_path, "--", input_file.path]

        return GenericConverter.generic_subprocess_conversion(input_file, output_file, command, interpreter, clean_up=clean_up if do_clean_up else None)

    return func          


# def musicxml_to_pdf(input_file: File, output_file: File) -> ConversionFunction:
#     pass


