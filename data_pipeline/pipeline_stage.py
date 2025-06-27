from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Pipeline_stage():    
    name: str
    extention: str
    child: Pipeline_stage | None
    folder_path: str | None
    

    pipeline = {}
    
    def __post_init__(self):
        Pipeline_stage.pipeline[self.name] = self

def construct_pipeline():
    tokens = Pipeline_stage("tokens",".json", None, None)
    midi_in = Pipeline_stage("midi_in",".midi", tokens, None)
    musicxml_in = Pipeline_stage("musicxml_in" , ".musicxml", midi_in, None)
    mxl_in = Pipeline_stage("mxl_in", ".mxl", musicxml_in, None)
    pdf_in = Pipeline_stage("pdf_in", ".pdf", musicxml_in, None)
    
    return Pipeline_stage.pipeline