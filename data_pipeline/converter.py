import os, subprocess, datetime, warnings, music21, logging
from typing import *
from data_pipeline.pipeline import *
from file import *


class Converter():
    """
    A class to handle the conversion of files between .pdf, .mxl, .musicxml, .midi, tokens.
    It uses MuseScore for the conversion process.
    """
    own_path = os.path.abspath(__file__)
    own_directory = os.path.dirname(own_path)
    #pipeline_stages = construct_pipeline()
    
    
    def __init__(self, pipeline: Pipeline, data_folder_path: str=f"{own_directory}\\data", logs_folder_path: str= f"{own_directory}\\logs", musescore_path: str=r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe', audiveris_path: str=r"C:\Program Files\Audiveris\Audiveris.exe", ):
        # environment.set('musescoreDirectPNGPath', musescore_path)
        self.musescore_path = musescore_path
        self.audiveris_path = audiveris_path
        self.logs_folder_path = logs_folder_path
        self.data_folder_path = data_folder_path
        os.makedirs(self.logs_folder_path, exist_ok=True)
        os.makedirs(self.data_folder_path, exist_ok=True)
        
        self.pipeline = pipeline
        self.assign_folders()    
        self.conversion_map = {
        (".pdf", ".musicxml"): self.pdf_to_musicxml,
        (".mxl", ".musicxml"): self.mxl_to_musicxml,
        (".musicxml", ".pdf"): self.musicxml_to_pdf,
        }

        self.log_folder_map = {}

        for function in self.conversion_map.values():
            path = os.path.join(self.logs_folder_path, f"{function}")
            os.makedirs(path, exist_ok=True)
            self.log_folder_map[function] = path


    def acceptable_pipeline(self):
        

        print("Pipeline accepted, all conversions can be performed by converter.")



    def assign_folders(self) -> None:
        for stage in self.pipeline:
            path = os.path.join(self.data_folder_path, folder)
            self.pipeline[stage_name].folder_path = path
            os.makedirs(path, exist_ok=True)


    def find_conversion_route(self, start_stage: Pipeline_stage, target_stage: Pipeline_stage) -> List[Callable]:
        
        result = []
        
        current = start_stage
        
        while current != target_stage:
            if current.child:
                result.append(self.conversion_map[(current.extention, current.child.extention)])
            else:
                raise ValueError(f"No converion route from stage: {start_stage.name} to stage: {target_stage.name}")
            
            current = current.child
        
        return result


    def multi_stage_conversion(self, start_stage: Pipeline_stage, target_stage: Pipeline_stage, overwrite: bool = True) -> None:
                     
        route = self.find_conversion_route(start_stage, target_stage)

        current = start_stage
        for func in route:
            self.single_stage_conversion(current, current.child, func, overwrite)
            current = current.child


    def single_stage_conversion(self, start_stage: Pipeline_stage, target_stage: Pipeline_stage, conversion_function: Callable, overwrite: bool=True) -> None: 
        if start_stage.child != target_stage:
                raise ValueError(f"Cannot convert from stage: {start_stage.name} to stage: {target_stage.name}")

        log = Log(start_stage, target_stage, self.log_folder_map[conversion_function])
        
        for file_name in os.listdir(start_stage.folder_path):
            if file_name.endswith(start_stage.extention):            
                input_file = File(os.path.join(start_stage.folder_path, file_name))
                
                output_file = File(os.path.join(target_stage.folder_path, os.path.splitext(file_name)[0] + target_stage.extention))

                self.logged_single_file_conversion(conversion_function, input_file, output_file, log, overwrite)
        
        log.commit()

                
    def logged_single_file_conversion(self, func, input_file: File, output_file: File, log: Log, overwrite: bool):
        log.num_attempted += 1
        
        if os.path.exists(output_file.path) and not overwrite:
            log.skip(input_file, output_file)
        else:
            try: 
                outcome = func(input_file, output_file)
                log.log(outcome)

            except Exception as e:
                log.halt(input_file, e)
                log.commit()
                raise RuntimeError(f"Critical failure since conversion of {input_file.name}. All upcoming conversion aborted\n" + str(e))
    

    def mxl_to_musicxml(self, input_file: File, output_file: File) -> ConversionOutcome:
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

    def pdf_to_musicxml(self, input_file: File, output_file: File) -> ConversionOutcome:
        subprocess.run([self.audidveris_path, "-batch", "-export", "-output", os.path.dirname(output_file.path), "--", input_file.path])            

    def musicxml_to_pdf(self, input_file: File, output_file: File) -> ConversionOutcome:
        """
        Converts all .musicxml files in the input folder to PDF using MuseScore.
        The output PDFs are saved in the specified output folder.
        """
        subprocess.run([self.musescore_path, input_file.path, '-o', output_file.path], check=True)

    conversion_map = {
            (".pdf", ".musicxml"): pdf_to_musicxml,
            (".mxl", ".musicxml"): mxl_to_musicxml,
            (".musicxml", ".pdf"): musicxml_to_pdf,
            }

if __name__ == "__main__":
    # Example usage
    folders = {
        "pdf_in": "pdf_in",
        "mxl_in": "mxl_in",
        "musicxml_in": "musicxml_in",
        "midi_in": "midi_in",
        "tokens": "tokens",
    }
    print(os.getcwd())
    pipeline = Converter.pipeline_stages
    converter = Converter(folders)
    converter.multi_stage_conversion(pipeline["mxl_in"], pipeline["musicxml_in"], overwrite=True)
