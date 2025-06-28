from file import *
import warnings, music21, subprocess
from typing import *

ConversionFunction = Callable[[File, File], ConversionOutcome]


class GenericConverter:
    @staticmethod
    def generic_subprocess_conversion(
        input_file: File, output_file: File, command: List[str], interpreter: Callable[[subprocess.CompletedProcess, File, File], ConversionOutcome]) -> ConversionOutcome: 
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

    @staticmethod
    def generic_music21_conversion(input_file: File, output_file: File, func: ConversionFunction) -> ConversionOutcome:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                func(input_file, output_file)
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
    
def pdf_to_musicxml(audiveris_path: str) -> ConversionFunction:
    def interpreter(result: subprocess.CompletedProcess[str], input_file: File, output_file: File) -> ConversionOutcome:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()

        if result.returncode == 0:
            warnings = tuple(stderr) if "WARN" in stderr else ()
            return ConversionOutcome(input_file=input_file, output_file=output_file, successful=True, warning_messages=warnings, go_on=True)
        
        if "FATAL" in stderr:
            return ConversionOutcome(input_file=input_file, output_file=output_file,successful=False, error_message=stderr, go_on=False)
        
        return ConversionOutcome(input_file=input_file, output_file=output_file, successful=False, error_message=stderr, go_on=True)
    
    def func(input_file, output_file) -> ConversionOutcome:
        command = [audiveris_path, "-batch", "-export", input_file.path]
        return GenericConverter.generic_subprocess_conversion(input_file, output_file, command, interpreter)

    return func          


# def musicxml_to_pdf(input_file: File, output_file: File) -> ConversionFunction:
#     pass


