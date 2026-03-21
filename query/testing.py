from query.encoder import encode

def test_query_encoding(q: str):
    query = q
    encoded = encode(query)

    print("Raw Query:", encoded.raw)
    print("Analysis:", encoded.analysis)
    print("Weights:", encoded.weights)
    print("Descriptor:", encoded.descriptor)
    print("Level Embeddings:", {k: [(t, c) for t, c, _ in v] for k, v in encoded.level_embeddings.items()})
    print("Pooled Vector (first 5 dims):", encoded.pooled_vector[:5])


list_of_queries = [
    "What are the health benefits of green tea?",
    "Compare the economic growth of China and India in the last decade.",
    "Who won the Nobel Prize in Physics in 2020?",
    "Explain the concept of quantum entanglement."
]

if __name__ == "__main__":
    for q in list_of_queries:
        print("\n--- Testing Query ---")
        test_query_encoding(q)