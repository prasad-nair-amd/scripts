import argparse
import time
import os
import openai

# PyTorch Profiler imports
import torch
import torch.profiler

parser = argparse.ArgumentParser()
parser.add_argument("--instance_id", type=int, default=0, help="Instance number for logging")
parser.add_argument("--results_dir", type=str, default="results", help="Directory to store results")
parser.add_argument("--program_id", type=int, default=0, help="Program copy id")
args = parser.parse_args()

openai.api_key = os.getenv("OPENAI_API_KEY")

def benchmark_openai_embeddings(sentences=None, batch_size=32, model_name="text-embedding-3-small"):
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    os.makedirs(args.results_dir, exist_ok=True)

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

    # PyTorch Profiler context for embedding batch calls
    profiler_output = os.path.join(args.results_dir, f"profiler_{args.instance_id}_{args.program_id}.txt")
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.results_dir, worker_name=f"openai_{args.instance_id}_{args.program_id}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        start_time = time.time()
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            all_embeddings.extend(get_embeddings_batch(batch))
            prof.step()
        total_time = time.time() - start_time
        throughput = len(sentences) / total_time
        # Optionally export profiler events to file
        prof.export_stacks(profiler_output, "self_cuda_time_total")

    # Latency measurement
    sample_sentences = sentences[:min(100, len(sentences))]
    latencies = []
    for sent in sample_sentences:
        t0 = time.time()
        _ = get_embeddings_batch([sent])
        t1 = time.time()
        latencies.append(t1 - t0)
    avg_latency = sum(latencies) / len(latencies)

    # Print results
    print(f"\n[Instance {args.instance_id}_{args.program_id}] Model: {model_name}")
    print(f"[Instance {args.instance_id}_{args.program_id}] Processed {len(sentences)} sentences in {total_time:.4f} s")
    print(f"[Instance {args.instance_id}_{args.program_id}] Throughput: {throughput:.2f} sentences/sec")
    print(f"[Instance {args.instance_id}_{args.program_id}] Average latency: {avg_latency * 1000:.2f} ms\n")

    # Save results to file for aggregator
    with open(os.path.join(args.results_dir, f"metrics_{args.instance_id}_{args.program_id}.txt"), "w") as f:
        f.write(f"{throughput},{avg_latency}\n")

if __name__ == "__main__":
    benchmark_openai_embeddings()
