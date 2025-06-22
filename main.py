from evaluation.evaluate import ModelFactory, DatasetGenerator, EvaluationPipeline
import os



from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # Load environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Initialize model factory
    model_factory = ModelFactory(chatgroq_api_key=groq_api_key)
    chat_groq_llm, huggingface_embedding_model = model_factory.get_models()

    # Generate synthetic dataset
    dataset_generator = DatasetGenerator()
    #dataset_generator.save_dataset()

    # Run evaluation pipeline
    evaluation_pipeline = EvaluationPipeline()
    test_cases = evaluation_pipeline.generate_testcases()
    results = evaluation_pipeline.run_evaluation(test_cases,chat_groq_llm,huggingface_embedding_model)
    
    # Save evaluation results
    evaluation_pipeline.save_results(results)