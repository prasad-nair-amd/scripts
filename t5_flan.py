import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load T5-Flan model and tokenizer
model_name = "google/flan-t5-xl"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load IMDb dataset
train_dataset, test_dataset = IMDB(split=('train', 'test'))

# Choose a subset of the test dataset for benchmarking
benchmark_data = list(test_dataset)[:100]

# Function to perform inference on a single input
def perform_inference(input_text):
    output_text = ''
    if device == 'cpu':
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    else:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    output_ids = model.generate(input_ids)
    for txt in output_ids:
       output_text = output_text + tokenizer.decode(txt, skip_special_tokens=True)
    return output_text

# Function to measure inference time on the benchmark data
def benchmark_inference(data):
    start_time = time.time()
    for example in data:
        input_text = example[1]  # IMDb reviews are in the second column
        output_text = perform_inference(input_text)
        #print(f"input text : {input_text}")
        #print(f"output text : {output_text}")
    end_time = time.time()
    return end_time - start_time

# Perform benchmarking
inference_time = benchmark_inference(benchmark_data)

# Measure memory usage
memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # in megabytes

# Display results
print(f"Inference time for {len(benchmark_data)} examples: {inference_time:.2f} seconds")
if memory_used == 0:
    print("CUDA : %s" %(torch.cuda.is_available()))
    print(f"GPU Memory used: {memory_used:.2f} MB")
