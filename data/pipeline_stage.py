from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Pipeline_stage():    
    name: str
    type: str
    folder: str | None
    child: Pipeline_stage | None


def construct_pipeline():
    tokens = Pipeline_stage("tokens", None, None)
    midi_in = Pipeline_stage("midi", tokens, None)
    musicxml_in = Pipeline_stage("musicxml", midi_in, None)
    mxl_in = Pipeline_stage("mxl", musicxml_in, None)
    pdf_in = Pipeline_stage("pdf", musicxml_in, None)
    
    pipeline_stages = [tokens, midi_in, musicxml_in, mxl_in, pdf_in]
    return {stage.name: stage for stage in pipeline_stages}