from deepeval.synthesizer import Synthesizer 
from deepeval.synthesizer.config import ContextConstructionConfig,StylingConfig

import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the synthesizer with custom LLM and embedding models
def synthesize():
    """
    This function initializes the synthesizer 
    and generates golden outputs from documents.
    """
    styling_config=StylingConfig(
            input_format="Questions about homewoner insurance policy. What cover/not cover depending on home damage situation",
            expected_output_format="Detail answer on what covered/not covered for that accident based on policy",
            task="RAG chatbot for homewner insurance policy documents",
            scenario="Customer asking queries about their policy information"

        )
    
    synthesizer = Synthesizer(cost_tracking=True,styling_config=styling_config)

    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=["evaluation/data/docs/sample_policy_doc_AU1234.pdf"],
        include_expected_output=True
      #  max_goldens_per_context=2,  # Limit to one golden per context
      #  context_construction_config=ContextConstructionConfig(
          #  max_contexts_per_document=1,  # Limit to one context per document 
     #   )     
    )

    return synthesizer, goldens
