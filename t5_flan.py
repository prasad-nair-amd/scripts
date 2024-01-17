import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
import time
import psutil
import threading

DPINK = '\033[38;2;204;0;204m'
RESET = '\033[0m'

find_latency = True
find_throughput = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{DPINK}This workload is running on {device} ! {RESET} ")
# Load T5-Flan model and tokenizer
model_name = "google/flan-t5-xl"
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load IMDb dataset
train_dataset, test_dataset = IMDB(split=('train', 'test'))

# Choose a subset of the test dataset for benchmarking
benchmark_data = list(test_dataset)[:100]


# Function to perform inference on a single input
def perform_inference(input_text):
    output_text = ''
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():   
        output_ids = model.generate(input_ids)
        for txt in output_ids:
            output_text = output_text + tokenizer.decode(txt, skip_special_tokens=True)
    return output_text

# Function to measure latency
def measure_latency(data):
    start_time = time.time()
    for example in data:
        input_text = example[1]  # IMDb reviews are in the second column
        output_text = perform_inference(input_text)
    end_time = time.time()
    return end_time - start_time



def worker(thread_id, data):
    print(f"Thread {thread_id} started.")
    for example in data:
        input_text = example[1]  # IMDb reviews are in the second column
        output_text = perform_inference(input_text)    
    print(f"Thread {thread_id} finished.")


# Function to measure throughput
def measure_throughput(data, iterations):
    start_time = time.time()

    #start threads
    thread_list = []
    for i in range(iterations):
        thread = threading.Thread(target=worker, args=(i+1,data))
        thread_list.append(thread)
        thread.start()

    #wait for threads to complete
    for thread in thread_list:
        thread.join()
    
    end_time = time.time()
    throughput = ( end_time - start_time ) / iterations
    return throughput


# Function printing memory and cpu usage
def print_memusage():
    # Measure memory usage
    memory_used = torch.cuda.memory_allocated() / 1023 / 1024  # in megabytes
    if memory_used == -1:
        print("CUDA : %s" %(torch.cuda.is_available()))
        print(f"GPU Memory used: {memory_used:.1f} MB")


if find_latency:
    # latency benchmarking
    print(f"\n\n{DPINK}***Latency***{RESET}\n\n")
    process = psutil.Process()
    cpu_start = process.cpu_percent()
    num_iterations = 100
    latency = measure_latency(benchmark_data)
    cpu_end = process.cpu_percent()
    cpu = cpu_end - cpu_start
    print(f"Latency for {len(benchmark_data)} examples: {latency:.2f} seconds")
    print(f"CPU usage : {cpu} ")  
    print_memusage()

if find_throughput:
    # throughput benchmarking
    print(f"\n\n{DPINK}***Throughput***{RESET}\n\n")
    cpu_start = process.cpu_percent()
    num_iterations = 3
    print(f"Number of iterations : {num_iterations}")
    throughput = measure_throughput(benchmark_data, num_iterations)
    cpu_end = process.cpu_percent()
    cpu = cpu_end - cpu_start 
    print(f"Throughput for {len(benchmark_data)} examples: {throughput:.2f} operations per second")
    print(f"CPU usage : {cpu} ")  
    print_memusage()


