from __future__ import annotations
from typing import *
from file import File, ConversionOutcome
import subprocess, music21, warnings, os

class Pipeline_stage():    
    def __init__(self, name: str, extension: str, children: dict[Pipeline_stage, Callable[[File, File], ConversionOutcome]]):
        self.name: str = name
        self.extension: str = extension
        self.children: dict[Pipeline_stage, Callable[[File, File], ConversionOutcome]] = children

class Pipeline():
    def __init__(self, *args: Pipeline_stage):
        self.stages = {*args} 
        self.graph = {stage: stage.children for stage in self.stages}  
        self.num_stages = len(self.stages)
    
    def shortest_conversion_route(self, start_stage: Pipeline_stage, target_stage: Pipeline_stage) -> List[Pipeline_stage]:
        """_summary_

        Args:
            start_stage (Pipeline_stage): _description_
            target_stage (Pipeline_stage): _description_

        Returns:
            List[Pipeline_stage]: empty if no route
        """        
        visited = set()
        res = []
        shortest = [float("inf")]

        def dfs(current: List[Pipeline_stage]):
            if current[-1] == target_stage:
                shortest[0] = len(current)
                res.clear()
                res.extend(current)
            elif len(current) < shortest[0] - 1:
                for neighbour in current.children:
                    if neighbour not in visited:
                        dfs(current + [neighbour])
                            
        dfs([start_stage])

        if not res:
            raise ValueError(f"No conversion route from {start_stage.name} to {target_stage.name} in pipeline.")
        
        return res
    

def construct_music_pipeline(musescore_path: str=r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe', audiveris_path: str=r"C:\Program Files\Audiveris\Audiveris.exe"):
    pdf_out = Pipeline_stage("musicxml_out" , ".musicxml", None)
    musicxml_out = Pipeline_stage("musicxml_out" , ".musicxml", midi_in)
    midi_out = Pipeline_stage("midi_out", ".midi", musicxml_out)
    
    tokens = Pipeline_stage("tokens",".json", None)
    midi_in = Pipeline_stage("midi_in",".midi", tokens)
    musicxml_in = Pipeline_stage("musicxml_in" , ".musicxml", midi_in)
    mxl_in = Pipeline_stage("mxl_in", ".mxl", musicxml_in)
    pdf_in = Pipeline_stage("pdf_in", ".pdf", musicxml_in)

    def mxl_to_musicxml(input_file: File, output_file: File) -> ConversionOutcome:
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

    def pdf_to_musicxml(input_file: File, output_file: File) -> ConversionOutcome:
        subprocess.run([audiveris_path, "-batch", "-export", "-output", output_file.folder, "--", input_file.path])            

    def musicxml_to_pdf(input_file: File, output_file: File) -> ConversionOutcome:
        """
        Converts all .musicxml files in the input folder to PDF using MuseScore.
        The output PDFs are saved in the specified output folder.
        """
        subprocess.run([musescore_path, input_file.path, '-o', output_file.path], check=True)
    
    
    