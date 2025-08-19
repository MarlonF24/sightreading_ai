from __future__ import annotations
import data_pipeline_scripts.conversion_functions as conversion_functions, constants as constants
from dataclasses import dataclass
from typing import *
from pathlib import Path
from data_pipeline_scripts.conversion_func_infrastructure import *
from data_pipeline_scripts.conversion_func_infrastructure import _ConversionFunction
from tokeniser.tokeniser import MyTokeniser

# @dataclass(unsafe_hash=True)
class PipelineStage():
    """
    A class representing a stage in a data processing pipeline.

    Attributes:
    - name (str): The name of the pipeline stage. This should be a unique identifier.
    - extension (str): The file extension associated with this stage (e.g., ".musicxml", ".midi").
    - children (dict[PipelineStage, _ConversionFunction]): A dictionary mapping child PipelineStage objects to the conversion function required to reach them from the current stage.
    """

    def __init__(self, name: str, extension: str, children: dict[PipelineStage, _ConversionFunction] = {}, extra_dirs: list[str] = []) -> None:
        """
        Initializes a PipelineStage object.

        Args:
            name (str): The name of the pipeline stage. This should be a unique identifier.
            extension (str): The file extension associated with this stage (e.g., ".musicxml", ".midi").
            children (dict[PipelineStage, _ConversionFunction], optional): A dictionary mapping child PipelineStage objects to the conversion function required to reach them from the current stage. Defaults to an empty dictionary.
            extra_dirs (list[str], optional): A list of names of additional directories that will be created in this stage's data directory.
        """
        self.name: str = name
        self.extension: str = extension
        self.children: dict[PipelineStage, _ConversionFunction] = children
        self.extra_dirs: list[str] = extra_dirs


    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', extension='{self.extension}', children={self.children}')"
    
    
    def add_child_stage(self, child_stage: PipelineStage, conversion_function: _ConversionFunction) -> None:
        """
        Adds a child stage and its associated conversion function to the current pipeline stage.

        Parameters:
            child_stage (PipelineStage): The child stage to be added.
            conversion_function (ConversionFunction): The conversion function required to transition from the current stage to the child stage.

        """
        self.children[child_stage] = conversion_function
    
    def remove_child_stage(self, child_stage: PipelineStage) -> None:
        """
        Removes a child stage from the current pipeline stage.

        Parameters:
            child_stage (PipelineStage): The child stage to be removed. This stage should exist in the current stage's children.
        """
        del self.children[child_stage]
    
    def set_conversion_function(self, target_stage: PipelineStage, conversion_function: _ConversionFunction) -> None:
        """
        Sets the conversion function for a specific child stage of the current pipeline stage.
    
        Parameters:
            target_stage (PipelineStage): The child stage for which the conversion function is being set.
            conversion_function (ConversionFunction): The conversion function to be used for transitioning from the current stage to the target stage.
        """
        self.add_child_stage(target_stage, conversion_function)
     
   
class Pipeline():
    """
    A class representing a data processing pipeline.

    Attributes:
        stages (Set[PipelineStage]): A set of PipelineStage objects representing the stages in the pipeline.
        stage_name_map (Dict[str, PipelineStage]): A dictionary mapping the names of the pipeline stages to their corresponding PipelineStage objects.
        graph (Dict[PipelineStage, List[PipelineStage]]): A dictionary representing the directed acyclic graph (DAG) of the pipeline stages.
        size (int): The total number of stages in the pipeline.
    """

    def __init__(self, *args: PipelineStage):
        """
        Initializes a Pipeline object with the provided PipelineStage instances.
    
        Parameters:
            *args (PipelineStage): Variable length argument list of PipelineStage instances. These instances will be added to the pipeline.
        """
        self.stages = {stage for stage in args}


    @property
    def stages(self) -> Set[PipelineStage]:
        return self._stages
    

    @stages.setter
    def stages(self, stages: Set[PipelineStage]):
        """
        Sets the stages of the pipeline.
    
        Parameters:
        - stages (Set[PipelineStage]): A set of PipelineStage instances to be added to the pipeline.
    
        This setter function performs the following tasks:
        1. Sets the internal stages attribute to the provided stages.
        2. Calls the check_unique_stage_names method to ensure that all stage names are unique.
        3. Calls the check_all_children_contained method to ensure that all child stages are contained within the pipeline.
        """
        self._stages = stages
        self.check_unique_stage_names()
        self.check_all_children_contained()


    @property
    def stage_name_map(self) -> Dict[str, PipelineStage]:
        """
        A property that returns a dictionary mapping the names of the pipeline stages to their corresponding PipelineStage objects.
    
        This dictionary is useful for quickly accessing a specific stage by its name.
    
        Returns:
        Dict[str, PipelineStage]: A dictionary where the keys are the names of the pipeline stages and the values are the corresponding PipelineStage objects.
        """
        return {stage.name : stage for stage in self.stages}
    

    @property
    def graph(self) -> Dict[PipelineStage, Dict[PipelineStage, _ConversionFunction]]:
        """
        Returns a dictionary representing the graph structure of the pipeline.
    
        The graph is a dictionary where each key is a PipelineStage instance, and the corresponding value is another dictionary.
        This inner dictionary maps child PipelineStage instances to the conversion functions required to reach them from the current stage.
    
        Returns:
        Dict[PipelineStage, Dict[PipelineStage, _ConversionFunction]]: A dictionary representing the graph structure of the pipeline.
        """
        return {stage: stage.children for stage in self.stages}
    

    def check_unique_stage_names(self):
        """
        Checks if all stage names in the pipeline are unique.

        Raises:
        ValueError: If any duplicate stage names are found.
        """
        if len(self.stage_name_map) < len(self.stages):
            stage_names = [stage.name for stage in self.stages]
            duplicate_stages = set([name for name in stage_names if stage_names.count(name) > 1])
            raise ValueError(f"Cannot have pipeline with duplicate pipeline stage names.\nDuplicate stage names found: {duplicate_stages}")


    def check_all_children_contained(self):
        """
        Checks if all child stages in the pipeline are contained within the pipeline.
    
        Raises:
            ValueError: If any child stage is not found within the pipeline.
        """
        uncontained_children = {child.name for children in self.graph.values() for child in children if child not in self}
        
        if uncontained_children:                
            raise ValueError(f"Cannot have pipeline with uncontained children: {uncontained_children}")


    def __repr__(self):
        stages_representation = ",\n\n".join([repr(stage) for stage in self.stages])
        return f"{type(self).__name__}(stages={stages_representation})"
    

    def to_stage(self, *stages: str | PipelineStage) -> tuple[PipelineStage, ...] | PipelineStage:
            """
            Converts a variable number of stage names or PipelineStage objects into a tuple of PipelineStage objects.
        
            Parameters:
                *stages (str | PipelineStage): Variable length argument list. Each element can be either a string representing a stage name or a PipelineStage object.
        
            Returns:
                tuple[PipelineStage, ...]: A tuple of PipelineStage objects. If a string is provided, it is converted into a PipelineStage object using the stage name map.
        
            This function is used to simplify the process of working with stages in the pipeline. It allows for the use of either stage names or PipelineStage objects as input parameters.
            """
            return tuple(self[stage] if isinstance(stage, str) else stage for stage in stages)
    

    def __contains__(self, stage: PipelineStage | str) -> bool:
        """
        Checks if a given stage is present in the pipeline.
    
        The function accepts either a PipelineStage object or a string representing the name of the stage.
        If a string is provided, the function checks if the corresponding PipelineStage object exists in the pipeline.
        If a PipelineStage object is provided, the function directly checks if it exists in the pipeline.
    
        Parameters:
            stage (PipelineStage | str): The stage to be checked. This can be either a PipelineStage object or a string representing the name of the stage.
    
        Returns:
            bool: Returns True if the stage is present in the pipeline, False otherwise.
        """
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
    
    
    def remove_stage(self, *args: PipelineStage | str) -> None:
        """
        Removes one or more stages from the pipeline.
    
        Parameters:
            *args (PipelineStage | str): Variable length argument list. Each element can be either a string representing a stage name or a PipelineStage object.
    
    
        This function uses the `to_stage` method to convert any string stage names into PipelineStage objects.
        It then removes the specified stages from the pipeline by updating the `stages` attribute.
        """
        self.stages = self.stages - set(self.to_stage(*args))
            
        
    def add_stage(self, *args: PipelineStage) -> None:
        """
        Adds one or more stages to the pipeline.
    
        Parameters:
            *args (PipelineStage): Variable length argument list. Each element should be a PipelineStage object.
        """
        self.stages = self.stages.union(set(args))
    

    def shortest_conversion_route(self, start_stage: PipelineStage |str, target_stage: PipelineStage | str) -> List[PipelineStage]:
        """
        Finds the shortest conversion route between two stages in the pipeline.

        Parameters:
            start_stage (PipelineStage | str): The starting stage for the conversion route. This can be either a PipelineStage object or a string representing the name of the stage.
            target_stage (PipelineStage | str): The target stage for the conversion route. This can be either a PipelineStage object or a string representing the name of the stage.

        Returns:
            List[PipelineStage]: A list of PipelineStage objects representing the shortest conversion route from the start stage to the target stage.

        The function uses a depth-first search (DFS) algorithm to find the shortest conversion route.
        It keeps track of visited stages, the current shortest route, and the length of the current route.
        If a shorter route is found, the function updates the shortest route and its length.
        If no conversion route is found, a ValueError is raised.
        """
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
    

def construct_music_pipeline(tokeniser: MyTokeniser, pdf_preprocess: bool, musescore_path: str = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe', audiveris_app_dir: str = r"C:\Program Files\Audiveris\app") -> Pipeline:

    musicxml_out = PipelineStage("musicxml_out", constants.MUSICXML_EXTENSION, {})
    
    midi_out = PipelineStage("midi_out", constants.MIDI_EXTENSION, {musicxml_out: conversion_functions.midi_to_musicxml()}, extra_dirs=[constants.data_pipeline_constants.METADATA_DIR_NAME])
    
    tokens_out = PipelineStage("tokens_out", constants.TOKENS_EXTENSION, {midi_out: conversion_functions.tokens_to_midi(tokeniser)})

    tokens_in = PipelineStage("tokens_in", constants.TOKENS_EXTENSION, {})
    
    midi_in = PipelineStage("midi_in", constants.MIDI_EXTENSION, children={tokens_in: conversion_functions.midi_to_tokens(tokeniser)}, extra_dirs=[constants.data_pipeline_constants.METADATA_DIR_NAME])
    
    #musicxml_in = PipelineStage("musicxml_in", constants.MUSICXML_EXTENSION, {midi_in: conversion_functions.mxl_to_midi(tokeniser)})

    mxl_in = PipelineStage("mxl_in", constants.MXL_EXTENSION, children={midi_in: conversion_functions.mxl_to_midi(tokeniser)})

    if pdf_preprocess:
        pdf_preprocessed = PipelineStage("pdf_preprocessed", constants.PDF_EXTENSION, children={mxl_in: conversion_functions.pdf_to_mxl(audiveris_app_dir=Path(audiveris_app_dir))})

    pdf_in = PipelineStage("pdf_in", constants.PDF_EXTENSION, {mxl_in: conversion_functions.pdf_to_mxl(audiveris_app_dir=Path(audiveris_app_dir))})

    pipeline = Pipeline(musicxml_out, midi_out, tokens_out, tokens_in, midi_in, mxl_in, pdf_in)

    if pdf_preprocess:
        pipeline.add_stage(pdf_preprocessed)
        pipeline["pdf_in"].remove_child_stage(pipeline["mxl_in"])
        pipeline["pdf_in"].add_child_stage(pdf_preprocessed, conversion_functions.pdf_preprocessing())
    
    return pipeline

if __name__ == "__main__":
    pipeline = construct_music_pipeline(tokeniser=MyTokeniser())
    print(pipeline)
    #print(pipeline.shortest_conversion_route("midi_in", "tokens"))

    
    
    