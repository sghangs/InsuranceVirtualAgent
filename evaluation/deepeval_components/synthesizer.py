from deepeval.synthesizer import Synthesizer 
from deepeval.synthesizer.config import ContextConstructionConfig,StylingConfig
from src.utility.utils import list_file_paths
from src.exception.exception import InsuranceAgentException
import sys
from src.constant import (
    INPUT_FORMAT,
    EXPECTED_OUTPUT_FORMAT,
    TASK,
    SCENARIO,
    POLICY_DOCUMENTS_PATH,
    MAX_GOLDEN_PER_CONTEXT
)

import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the synthesizer with custom LLM and embedding models
def synthesize():
    """
    This function initializes the synthesizer 
    and generates golden outputs from documents.
    """
    try:
        styling_config=StylingConfig(
                input_format=INPUT_FORMAT,
                expected_output_format=EXPECTED_OUTPUT_FORMAT,
                task=TASK,
                scenario=SCENARIO
            )
        
        synthesizer = Synthesizer(cost_tracking=True,styling_config=styling_config)

        #generate file path for each policy file
        list_path = list_file_paths(POLICY_DOCUMENTS_PATH)

        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=list_path,
            include_expected_output=True,
            max_goldens_per_context=MAX_GOLDEN_PER_CONTEXT    
        )

        return synthesizer, goldens
    
    except Exception as e:
        raise InsuranceAgentException(sys,e)
