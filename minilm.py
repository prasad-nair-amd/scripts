import time
import torch
import tqdm
from sentence_transformers import SentenceTransformer
import datasets

#Load model 
model = SentenceTransformer('all-MiniLM-L6-v2')
model.max_seq_length = 512

#Preparing a dataset
dataset = datasets.load_dataset("IMDB")

data = dataset["train"]["text"][:10000]

#Function to measure latency
def run_inference(sentence):
    start_time = time.perf_counter()
    with torch.no_grad():
        sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    end_time = time.perf_counter()
    batch_latency = end_time - start_time
    return batch_latency


#Measure latency for each sentence in the dataset
latencies = []
for sentence in tqdm.tqdm(data):
    latency = run_inference(sentence)
    latencies.append(latency)



#Calculate average latency and throughput
average_latency = sum(latencies) / len(latencies)
print("average latency :", average_latency  , "seconds")
print("latency(10000 requests) :", average_latency * 10000  , "seconds")
throughput = len(data) / sum(latencies)
print("Throughput :", throughput, "samples per second")
