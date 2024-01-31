import time
import torch
import tqdm
from sentence_transformers import SentenceTransformer
import datasets

#Load model 
model = SentenceTransformer('all-MiniLM-L6-v2')


#Preparing a dataset
dataset = datasets.load_dataset("sst2")

data = dataset["train"]["sentence"][:10000]

print(len(data))

#Function to measure latency
def measure_latency(sentence):
    start_time = time.perf_counter()
    with torch.no_grad():
        sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    end_time = time.perf_counter()
    batch_latency = end_time - start_time
    return end_time - start_time


#Measure latency for each sentence in the dataset
latencies = []
for sentence in tqdm.tqdm(data):
    #print(sentence)
    latency = measure_latency(sentence[:512])
    latencies.append(latency)



#Calculate average latency and throughput
average_latency = sum(latencies) / len(latencies)
throughput = len(dataset) / sum(latencies)

print("Average latency(10000) :", average_latency * 10000 , "seconds")
print("Throughput :", throughput, "samples per second")
