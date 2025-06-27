from __future__ import annotations
from typing import *
from file import File, ConversionOutcome

class Pipeline_stage():    
    def __init__(self, name: str, extention: str, children: dict[Pipeline_stage, Callable[[File, File], ConversionOutcome]]):
        self.name: str = name
        self.extention: str = extention
        self.children: dict[Pipeline_stage, Callable[[File, File], ConversionOutcome]] = children

class Pipeline():
    def __init__(self, *args: Pipeline_stage):
        self.stages = {*args} 
        self.graph = {stage: stage.children for stage in self.stages}  
        self.num_stages = len(self.stages)
    
    def have_conversion_route(self, start_stage: Pipeline_stage, target_stage: Pipeline_stage) -> List[Pipeline_stage]:
        visited = set()
        res = []
        shortest = float("inf")

        def dfs(current: List[Pipeline_stage]):
            if current[-1] == target_stage:
                shortest = len(current)
                res = current
            elif len(current) < shortest - 1:
                for neighbour in current.children:
                    if neighbour not in visited:
                        dfs(current + [neighbour])
                            
        dfs(start_stage)
        return res
    

def construct_pipeline(folders: dict):
    tokens = Pipeline_stage("tokens",".json", None, None)
    midi_in = Pipeline_stage("midi_in",".midi", tokens, None)
    musicxml_in = Pipeline_stage("musicxml_in" , ".musicxml", midi_in, None)
    mxl_in = Pipeline_stage("mxl_in", ".mxl", musicxml_in, None)
    pdf_in = Pipeline_stage("pdf_in", ".pdf", musicxml_in, None)
    
    
    return Pipeline_stage.pipeline