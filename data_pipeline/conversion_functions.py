from file import File, ConversionOutcome
import warnings, music21, subprocess

def mxl_to_musicxml(): 
    def wrapper(input_file: File, output_file: File) -> ConversionOutcome:
        """
        Converts all .mxl files in the input folder to .musicxml.
        The output files are saved in the specified output folder.
        """
        
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            score = music21.converter.parse(input_file.path)
            score.write('musicxml', fp=output_file.path)

        warning_list = [f"MusicXMLWarning: {w.message}" for w in caught_warnings]

        return ConversionOutcome(input_file, output_file, successful=True, warning_messages=tuple(warning_list), error_message=None, go_on=True)

    return wrapper

def pdf_to_musicxml(audiveris_path: str):
    def wrapper(input_file: File, output_file: File) -> ConversionOutcome:
        subprocess.run([audiveris_path, "-batch", "-export", "-output", output_file.folder, "--", input_file.path]) 
    
    return wrapper          

def musicxml_to_pdf(input_file: File, output_file: File) -> ConversionOutcome:
     pass