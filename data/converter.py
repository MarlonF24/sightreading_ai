from music21 import converter, environment
import os
import subprocess
from pipeline_stage import Pipeline_stage, construct_pipeline


class Converter():
    """
    A class to handle the conversion of files between .pdf, .mxl, .musicxml, .midi, tokens.
    It uses MuseScore for the conversion process.
    """
    conversion_map = {}
    pipeline_stages = construct_pipeline()

    def __init__(self, folders: dict, musescore_path=r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'):
        environment.set('musescoreDirectPNGPath', musescore_path)
        self.own_path = os.path.dirname(os.path.abspath(__file__)) 
        self.assign_folders(folders)
        
        
    def assign_folders(self, folders: dict) -> None:
        # Enforce that folders dict has a folder for each Pipeline_stage
        missing = [stage.name for stage in Converter.pipeline_stages if stage.name not in folders]
        if missing:
            raise ValueError(f"Missing folders for stages: {', '.join(missing)}")
        else:
            for stage_name, folder in folders.items():
                Converter.pipeline_stages[stage_name].folder = folder
                os.makedirs(folder, exist_ok=True)

    
    def conversion_decorator(start: str, target:str) -> function:
        def decorator(func: function) -> function:
            func.start = start
            func.target = target
            Converter.conversion_map[(start, target)] = func
            return func
        return decorator

    def convert(start_stage: str, target_stage: str) -> None:
        pass
    
    @conversion_decorator("mxl", "musicxml")
    def mxl_to_musicxml(self, input_folder: str, output_folder:str) -> None:
        """
        Converts all .mxl files in the input folder to .musicxml.
        The output files are saved in the specified output folder.
        """
        for filename in os.listdir(input_folder):
            if filename.endswith('.mxl'):
                file_path = os.path.join(input_folder, filename)
                score = converter.parse(file_path)

                # Write to .musicxml
                xml_filename = os.path.splitext(filename)[0] + '.musicxml'
                xml_path = os.path.join(output_folder, xml_filename)
                score.write('musicxml', fp=xml_path)

    @conversion_decorator("musicxml", "pdf")
    def musicxml_to_pdf(self, input_folder: str, output_folder:str) -> None:
        """
        Converts all .musicxml files in the input folder to PDF using MuseScore.
        The output PDFs are saved in the specified output folder.
        """

        for filename in os.listdir(input_folder):
            if filename.endswith('.musicmxl'):
                file_path = os.path.join(input_folder, filename)

                # Convert .musicxml to PDF using MuseScore
                pdf_filename = os.path.splitext(filename)[0] + '.pdf'
                pdf_path = os.path.join(output_folder, pdf_filename)
                subprocess.run([self.musescore_path, file_path, '-o', pdf_path], check=True)

if __name__ == "__main__":
    # Example usage
    folders = {
        "mxl": "path/to/mxl",
        "musicxml": "path/to/musicxml",
        "pdf": "path/to/pdf",
        "tokens": "path/to/tokens",
        "midi": "path/to/midi"
    }
    
    converter = Converter(folders)
    converter.mxl_to_musicxml(folders["mxl"], folders["musicxml"])
    converter.musicxml_to_pdf(folders["musicxml"], folders["pdf"])
