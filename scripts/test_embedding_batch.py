from app.rag.embedding import SiliconFlowEmbeddingFunction
import logging

logging.basicConfig(level=logging.INFO)

def test_batching():
    fn = SiliconFlowEmbeddingFunction()
    # Generate 125 dummy strings (more than 2 full batches of 60)
    test_input = [f"This is test sentence number {i}" for i in range(125)]
    
    print(f"Testing embedding with {len(test_input)} items...")
    try:
        embeddings = fn(test_input)
        print(f"Successfully generated {len(embeddings)} embeddings.")
        assert len(embeddings) == 125
        print("Batching logic verified!")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_batching()
