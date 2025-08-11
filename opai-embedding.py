import time
import os
import openai

# Set your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def benchmark_openai_embeddings(sentences=None, batch_size=32, model_name="text-embedding-3-small"):
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Dummy sentences if none provided
    if sentences is None:
        sentences = [f"This is sentence {i}" for i in range(1000)]

    def get_embeddings_batch(text_batch):
        response = openai.embeddings.create(
            model=model_name,
            input=text_batch
        )
        return [item.embedding for item in response.data]

    # Warm-up
    _ = get_embeddings_batch(sentences[:batch_size])

    # Throughput
    start_time = time.time()
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        all_embeddings.extend(get_embeddings_batch(batch))
    total_time = time.time() - start_time
    throughput = len(sentences) / total_time

    # Latency
    sample_sentences = sentences[:min(100, len(sentences))]
    latencies = []
    for sent in sample_sentences:
        t0 = time.time()
        _ = get_embeddings_batch([sent])
        t1 = time.time()
        latencies.append(t1 - t0)

    avg_latency = sum(latencies) / len(latencies)

    print(f"Model: {model_name}")
    print(f"Processed {len(sentences)} sentences in {total_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} sentences/second")
    print(f"Average latency per sentence: {avg_latency * 1000:.2f} ms")


if __name__ == "__main__":
    benchmark_openai_embeddings()
