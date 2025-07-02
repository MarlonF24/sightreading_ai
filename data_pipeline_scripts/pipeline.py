import conversion_functions
from __future__ import annotations
from typing import *
from pathlib import Path
from conversion_func_infrastructure import *


class PipelineStage():    
    def __init__(self, name: str, extension: str, children: dict[PipelineStage, ConversionFunction] = {}) -> None:
        self.name: str = name
        self.extension: str = extension
        self.children: dict[PipelineStage, ConversionFunction] = children


    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', extension='{self.extension}', children={self.children}')"
    

    def set_conversion_function(self, target_stage: PipelineStage, conversion_function: ConversionFunction) -> None:
        self.children[target_stage] = conversion_function
    

    def add_child_stage(self, child_stage: PipelineStage, conversion_function: ConversionFunction) -> None:
        self.children[child_stage] = conversion_function


    def remove_child_stage(self, child_stage: PipelineStage) -> None:
        del self.children[child_stage]
    
   
class Pipeline():
    def __init__(self, *args: PipelineStage):
        self.stages = {stage for stage in args}  
        
    @property
    def stages(self) -> Set[PipelineStage]:
        return self._stages
    
    @stages.setter
    def stages(self, stages: Set[PipelineStage]):
        self._stages = stages
        self.check_unique_stage_names()
        self.check_all_children_contained()

    @property
    def stage_name_map(self) -> Dict[str, PipelineStage]:
        return {stage.name : stage for stage in self.stages} 
    
    @property
    def graph(self) -> Dict[PipelineStage, Dict[PipelineStage, ConversionFunction]]:
        return {stage: stage.children for stage in self.stages}  
    
    def check_unique_stage_names(self):
        if len(self.stage_name_map) < len(self.stages):
                    stage_names = [stage.name for stage in self.stages]
                    duplicate_stages = set([name for name in stage_names if stage_names.count(name) > 1])
                    raise ValueError(f"Cannot have pipeline whith duplicate pipeline stage names.\nDuplicate stage names found: {duplicate_stages}")


    def check_all_children_contained(self):
        uncontained_children = {child.name for children in self.graph.values() for child in children if child not in self}
        
        if uncontained_children:                
            raise ValueError(f"Cannot create pipeline with uncontained children: {uncontained_children}")

    def __repr__(self):
        stages_representation = ",\n\n".join([repr(stage) for stage in self.stages])
        return f"{type(self).__name__}(stages={stages_representation})"

    def __contains__(self, stage: PipelineStage | str) -> bool:
        if isinstance(stage, str):
            return stage in self.stage_name_map

        return stage in self.stages
    
    
    def __getitem__(self, stage_name: str) -> PipelineStage:
        return self.stage_name_map[stage_name]


    def __iter__(self) -> Iterator[PipelineStage]:
        return iter(self.stages)


    def __len__(self) -> int:
        return len(self.stages)


    @property
    def size(self) -> int:
        return len(self)
    
    def to_stage(self, *stages: str | PipelineStage) -> tuple[PipelineStage, ...]:
        return tuple(self[stage] if isinstance(stage, str) else stage for stage in stages)


    def remove_stage(self, *args: PipelineStage | str) -> None:
        self.stages = self.stages - set(self.to_stage(*args))
            
        
    def add_stage(self, *args: PipelineStage) -> None:
        self.stages = self.stages.union(set(self.to_stage(*args)))
    

    def shortest_conversion_route(self, start_stage: PipelineStage |str, target_stage: PipelineStage | str) -> List[PipelineStage]:
        start_stage, target_stage = self.to_stage(start_stage, target_stage)

        visited: Set[PipelineStage] = set()
        res: List[PipelineStage] = []
        shortest = float("inf")

        def dfs(current: List[PipelineStage]):
            nonlocal shortest, res
            
            if current[-1] == target_stage:
                shortest = len(current)
                res = current

            elif len(current)  + 1 < shortest:
                for neighbour in current[-1].children:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        dfs(current + [neighbour])
                            
        visited.add(start_stage)
        dfs([start_stage])

        if not res:
            raise ValueError(f"No conversion route from {start_stage.name} to {target_stage.name} in {self}.")
        
        return res
    

def construct_music_pipeline(musescore_path: str = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe', audiveris_app_folder: str = r"C:\Program Files\Audiveris\app") -> Pipeline:
    #pdf_out = PipelineStage("musicxml_out" , ".musicxml", None)
    #musicxml_out = PipelineStage("musicxml_out" , ".musicxml", midi_in)
    #midi_out = PipelineStage("midi_out", ".midi", musicxml_out)
    
    tokens = PipelineStage("tokens",".json", {})
    midi_in = PipelineStage("midi_in",".midi", {})
    musicxml_in = PipelineStage("musicxml_in" , ".musicxml", {midi_in: conversion_functions.musicxml_to_midi()})
    mxl_in = PipelineStage("mxl_in", ".mxl", {musicxml_in: conversion_functions.mxl_to_musicxml()})
    pdf_in = PipelineStage("pdf_in", ".pdf", {mxl_in: conversion_functions.pdf_to_mxl(audiveris_app_folder=Path(audiveris_app_folder))})
    return Pipeline(tokens, midi_in, musicxml_in, mxl_in, pdf_in)

if __name__ == "__main__":
    pipeline = construct_music_pipeline()
    print(pipeline)
    #print(pipeline.shortest_conversion_route("midi_in", "tokens"))

    
    
    