from __future__ import annotations
import data_pipeline_scripts.conversion_functions as conversion_functions, constants as constants
from dataclasses import dataclass
from typing import *
from pathlib import Path
from data_pipeline_scripts.conversion_func_infrastructure import *
from data_pipeline_scripts.conversion_func_infrastructure import _ConversionFunction
from tokeniser.tokeniser import MyTokeniser

@dataclass(unsafe_hash=False)
class PipelineStage():
    """
    Represents a stage in a file conversion pipeline.

    A pipeline stage defines a specific file format or processing state in the conversion workflow. Each stage can have multiple child stages representing possible conversion targets, with associated conversion functions to transform files from the current stage to each child stage.

    Attributes:
        name (str): Unique identifier for this pipeline stage (e.g., "pdf", "midi").
        extension (str): File extension associated with files at this stage (e.g., ".pdf", ".midi", ".json").
        children (dict[PipelineStage, _ConversionFunction]): Maps child stages to their conversion functions.
            Each key-value pair represents a possible conversion path from this stage.
        extra_dirs (list[Tuple[str, str]]): Additional directories this stage creates for auxiliary files.
            Each tuple contains (directory_name, file_extension) for files stored separately from main outputs.
            Example: [("metadata", ".metadata.json")] for stages that save metadata alongside main files.

    Example:
        >>> midi_stage = PipelineStage(
        ...     name="midi_in", 
        ...     extension=".midi",
        ...     extra_dirs=[("metadata", ".metadata.json")]
        ... )
        >>> tokens_stage = PipelineStage("tokens_in", ".json")
        >>> midi_stage.add_child_stage(tokens_stage, midi_to_tokens_converter)
    """

    name: str
    extension: str
    children: dict[PipelineStage, _ConversionFunction] = field(default_factory=dict)
    extra_dirs: list[Tuple[str, str]] = field(default_factory=list)

    def add_child_stage(self, child_stage: PipelineStage, conversion_function: _ConversionFunction) -> None:
        """
        Establishes a conversion path from this stage to a child stage.
        Creates a new edge in the pipeline graph, indicating that files at this stage can be converted to the child stage using the specified conversion function.

        Args:
            child_stage (PipelineStage): The target stage for the conversion.
            conversion_function (_ConversionFunction): Function that performs the conversion
                from this stage's file format to the child stage's format.

        Example:
            >>> pdf_stage.add_child_stage(mxl_stage, pdf_to_mxl_converter)
        """
        self.children[child_stage] = conversion_function
    
    def remove_child_stage(self, child_stage: PipelineStage) -> None:
        """
        Removes a conversion path to a child stage.

        Deletes the edge in the pipeline graph, making the child stage no longer
        directly reachable from this stage.

        Args:
            child_stage (PipelineStage): The child stage to disconnect from this stage.

        Raises:
            KeyError: If the child_stage is not currently a child of this stage.

        Example:
            >>> pdf_stage.remove_child_stage(mxl_stage)  # Remove direct pdf->mxl conversion
        """
        del self.children[child_stage]
    
    def set_conversion_function(self, target_stage: PipelineStage, conversion_function: _ConversionFunction) -> None:
        """
        Updates the conversion function for an existing child stage.

        Replaces the current conversion function used to reach a specific child stage
        with a new one. The child stage must already be connected to this stage.

        Args:
            target_stage (PipelineStage): Existing child stage whose conversion function will be updated.
            conversion_function (_ConversionFunction): New conversion function to use for this path.

        Raises:
            KeyError: If target_stage is not currently a child of this stage.

        Example:
            >>> # Switch from one PDF-to-MXL converter to another
            >>> pdf_stage.set_conversion_function(mxl_stage, better_pdf_to_mxl_converter)
        """
        if target_stage not in self.children:
            raise KeyError(f"'{target_stage.name}' is not a child of '{self.name}'. "
                          f"Use add_child_stage() to create new conversion paths.")
        self.children[target_stage] = conversion_function

    def __hash__(self):
        return hash(self.name)


class Pipeline():
    """
    Manages a directed graph of music file conversion stages.

    A Pipeline represents the complete workflow for converting between different music file formats.
    It maintains stages and their conversion relationships, validates the pipeline structure,
    and provides methods for finding optimal conversion routes between any two stages.

    The pipeline allows cycles, enabling round-trip conversions and flexible format transformations.
    For example: MIDI → Tokens → MIDI for validation, or PDF → MIDI → Tokens for training.

    The pipeline ensures data integrity by validating that:
    - All stage names are unique
    - All child stage references point to stages within the pipeline

    Attributes:
        stages (Set[PipelineStage]): All stages in the pipeline.
        stage_name_map (Dict[str, PipelineStage]): Maps stage names to stage objects for quick lookup.
        graph (Dict[PipelineStage, Dict[PipelineStage, _ConversionFunction]]): 
            Adjacency list representation of the pipeline graph.
        size (int): Total number of stages in the pipeline.

    Example:
        >>> pipeline = Pipeline(pdf_stage, mxl_stage, midi_stage, tokens_stage)
        >>> route = pipeline.shortest_conversion_route("pdf_in", "tokens_out")
        >>> print([stage.name for stage in route])
        ['pdf_in', 'mxl_in', 'midi_in', 'tokens_out']
        
        >>> # Round-trip example
        >>> route = pipeline.shortest_conversion_route("tokens_out", "tokens_in") 
        >>> print([stage.name for stage in route])
        ['tokens_out', 'midi_out', 'midi_in', 'tokens_in']
    """

    def __init__(self, *stages: PipelineStage):
        """
        Creates a new pipeline with the specified stages.

        Initializes the pipeline and validates its structure. All provided stages
        are added to the pipeline, and validation checks ensure the pipeline is
        properly formed.

        Args:
            *stages (PipelineStage): Variable number of pipeline stages to include.
                These stages define the nodes in the conversion graph.

        Raises:
            ValueError: If stage names are not unique or if child references are invalid.

        Example:
            >>> pipeline = Pipeline(pdf_stage, mxl_stage, midi_stage)
        """
        self.stages: Set[PipelineStage] = set()
        self.add_stages(*stages)


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
    

    def to_stage(self, *stages: str | PipelineStage) -> tuple[PipelineStage, ...]:
            """
            Converts a variable number of stage names or PipelineStage objects into a tuple of PipelineStage objects.
        
            Parameters:
                *stages (str | PipelineStage): Variable length argument list. Each element can be either a string representing a stage name or a PipelineStage object.
        
            Returns:
                tuple[PipelineStage, ...]: A tuple of PipelineStage objects. If a string is provided, it is converted into a PipelineStage object using the stage name map.
        
            This function is used to simplify the process of working with stages in the pipeline. It allows for the use of either stage names or PipelineStage objects as input parameters.
            """

            if any(not isinstance(stage, (PipelineStage, str)) for stage in stages):
                raise TypeError("All arguments must be either str or PipelineStage instances.")

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
    
    
    def remove_stages(self, *stages: PipelineStage | str) -> None:
        """
        Removes one or more stages from the pipeline.
    
        Parameters:
            *stages (PipelineStage | str): Variable length argument list. Each element can be either a string representing a stage name or a PipelineStage object.
    
    
        This function uses the `to_stage` method to convert any string stage names into PipelineStage objects.
        It then removes the specified stages from the pipeline by updating the `stages` attribute.
        """

        self.stages = self.stages - set(self.to_stage(*stages))
                  
    def add_stages(self, *stages: PipelineStage) -> None:
        """
        Adds one or more stages to the pipeline.
    
        Parameters:
            *stages (PipelineStage): Variable length argument list. Each element should be a PipelineStage object.
        """
        for stage in stages:
            if not isinstance(stage, PipelineStage):
                raise TypeError(f"Expected PipelineStage instance, got {type(stage)}")
            
            if stage.name in self.stage_name_map:
                raise ValueError(f"Stage with name '{stage.name}' already exists in the pipeline.")
            
            self.stages.add(stage)

        self.check_unique_stage_names()
        self.check_all_children_contained()

    def shortest_conversion_route(self, start_stage: PipelineStage | str, target_stage: PipelineStage | str) -> List[PipelineStage]:
        """
        Finds the shortest conversion path between two stages.

        Uses depth-first search to find the minimum number of conversion steps
        required to transform files from the start stage format to the target stage format.
        Returns the complete path including both start and target stages.

        Args:
            start_stage (PipelineStage | str): Starting stage for the conversion.
                Can be a stage object or stage name string.
            target_stage (PipelineStage | str): Desired end stage for the conversion.
                Can be a stage object or stage name string.

        Returns:
            List[PipelineStage]: Ordered list of stages representing the conversion path.
                Includes both start_stage and target_stage as first and last elements.

        Raises:
            ValueError: If no conversion path exists between the specified stages.
            KeyError: If either stage name is not found in the pipeline.

        Example:
            >>> route = pipeline.shortest_conversion_route("pdf_in", "tokens_out")
            >>> print([stage.name for stage in route])
            ['pdf_in', 'mxl_in', 'midi_in', 'tokens_out']
            
            >>> # This represents: PDF -> MXL -> MIDI -> Tokens (3 conversion steps)
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

            elif len(current) + 1 < shortest:
                for neighbour in current[-1].children:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        dfs(current + [neighbour])
                            
        visited.add(start_stage)
        dfs([start_stage])

        if not res:
            raise ValueError(f"No conversion route from '{start_stage.name}' to '{target_stage.name}' "
                           f"exists in this pipeline. Check that both stages are connected.")
        
        return res
    

def construct_music_pipeline(tokeniser: MyTokeniser, pdf_preprocess: bool, 
                           musescore_path: str = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe', 
                           audiveris_app_dir: str = r"C:\Program Files\Audiveris\app") -> Pipeline:
    """
    Constructs a complete music file processing pipeline.

    Creates a pipeline for converting between various music file formats including PDF,
    MXL, MIDI, and tokenized representations. Supports optional PDF preprocessing for
    improved OCR accuracy.

    The pipeline supports these conversion paths:
    - PDF → MXL → MIDI → Tokens (full pipeline)
    - MIDI(input) → MIDI(processed) → Tokens (for pre-converted MIDI files)  
    - Tokens → MIDI → MXL (for model output conversion back to readable formats)

    Args:
        tokeniser (MyTokeniser): Configured tokenizer for MIDI ↔ Tokens conversion.
        pdf_preprocess (bool): Whether to include PDF preprocessing stage for better OCR.
            When True, adds a pdf_preprocessed stage that splits/enhances PDFs before OCR.
        musescore_path (str): Path to MuseScore executable for audio playback/conversion.
            Defaults to standard Windows installation path.
        audiveris_app_dir (str): Path to Audiveris application directory for PDF OCR.
            Defaults to standard Windows installation path.

    Returns:
        Pipeline: Configured pipeline ready for music file conversion workflows.

    Example:
        >>> tokeniser = MyTokeniser()
        >>> pipeline = construct_music_pipeline(tokeniser, pdf_preprocess=True)
        >>> # Convert PDF sheet music to training tokens
        >>> route = pipeline.shortest_conversion_route("pdf_in", "tokens_in")
        
    Note:
        The pipeline includes extra_dirs for stages that generate auxiliary files:
        - MIDI stages: Include metadata directories for musical analysis data
        - Token stages: May include additional tokenization metadata
    """
    if not isinstance(tokeniser, MyTokeniser):
        raise TypeError("Expected tokeniser to be an instance of MyTokeniser")
    if not isinstance(pdf_preprocess, bool):
        raise TypeError("Expected pdf_preprocess to be a boolean")
    

    musicxml_out = PipelineStage("musicxml_out", constants.MUSICXML_EXTENSION, {})

    midi_out = PipelineStage("midi_out", constants.MIDI_EXTENSION, {musicxml_out: conversion_functions.midi_to_mxl(transpose_to_desired_key=True)}, extra_dirs=[(constants.data_pipeline_constants.METADATA_DIR_NAME, constants.METADATA_EXTENSION)])

    tokens_out = PipelineStage("tokens_out", constants.TOKENS_EXTENSION, {midi_out: conversion_functions.tokens_to_midi(tokeniser)})

    tokens_in = PipelineStage("tokens_in", constants.TOKENS_EXTENSION, {})

    

    midi_in = PipelineStage("midi_in", constants.MIDI_EXTENSION, children={tokens_in: conversion_functions.midi_to_tokens(tokeniser)}, extra_dirs=[(constants.data_pipeline_constants.METADATA_DIR_NAME, constants.METADATA_EXTENSION)])

    #musicxml_in = PipelineStage("musicxml_in", constants.MUSICXML_EXTENSION, {midi_in: conversion_functions.to_midi(tokeniser)})

    mxl_in = PipelineStage("mxl_in", constants.MXL_EXTENSION, children={midi_in: conversion_functions.to_midi(tokeniser)})
    
    midi_start = PipelineStage("midi_start", constants.MIDI_EXTENSION, {midi_in: conversion_functions.to_midi(tokeniser)})

    if pdf_preprocess:
        pdf_preprocessed = PipelineStage("pdf_preprocessed", constants.PDF_EXTENSION, children={mxl_in: conversion_functions.pdf_to_mxl(audiveris_app_dir=Path(audiveris_app_dir))})

    pdf_in = PipelineStage("pdf_in", constants.PDF_EXTENSION, {mxl_in: conversion_functions.pdf_to_mxl(audiveris_app_dir=Path(audiveris_app_dir))})

    pipeline = Pipeline(musicxml_out, midi_out, tokens_out, tokens_in, midi_in, mxl_in, pdf_in, midi_start)

    if pdf_preprocess:
        pipeline.add_stages(pdf_preprocessed)
        pipeline["pdf_in"].remove_child_stage(pipeline["mxl_in"])
        pipeline["pdf_in"].add_child_stage(pdf_preprocessed, conversion_functions.pdf_preprocessing())
    
    return pipeline

if __name__ == "__main__":
    pass



