from evaluation.deepeval_components.synthesizer import synthesize
from src.exception.exception import InsuranceAgentException
import sys
from src.loggers.logger import logging

class DatasetGenerator:
    """Generates synthetic queries and golden answers using DeepEval Synthesizer."""

    def __init__(self):
        self.synthesizer,self.goldens = synthesize()
        
    def save_dataset(self):
        """
        Save the synthetic dataset result as json
        """    
        try:
            self.synthesizer.save_as(
                file_type='json',
                directory="data/goldens",
                file_name="synthetic_dataset"
            )
            logging.info(f"✅ Synthetic dataset saved to data/goldens/synthetic_dataset.json")
        except Exception as e:
            raise InsuranceAgentException(sys,e)


