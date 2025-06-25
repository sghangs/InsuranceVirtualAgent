from evaluation.deepeval_components.synthesizer import synthesize

class DatasetGenerator:
    """Generates synthetic queries and golden answers using DeepEval Synthesizer."""
    
    def __init__(self):
        self.synthesizer,self.goldens = synthesize()
        
        
    def save_dataset(self):
        self.synthesizer.save_as(
            file_type='json',
            directory="data/goldens",
            file_name="synthetic_dataset"
        )
        print(f"✅ Synthetic dataset saved to data/goldens/synthetic_dataset.json")


