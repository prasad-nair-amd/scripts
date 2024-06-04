import torch
import time
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the Llama 2 70B model and tokenizer

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


# Move the model to the GPU
#model.cuda()

# Set the model to evaluation mode
model.eval()

# Define the input prompt
# prompt = "Hello, how are you today?"
prompt = "Write something meaningful and see the output"
# Encode the prompt
#input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Measure the first token latency
with torch.no_grad():
    #start_time = torch.cuda.Event(enable_timing=True)
    #end_time = torch.cuda.Event(enable_timing=True)
    #start_time.record()
    start_time = time.time()
    output = model.generate(input_ids, max_length=input_ids.size(-1) + 1, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    #end_time.record()
    end_time = time.time()
    #torch.cuda.synchronize()
    #first_token_latency = start_time.elapsed_time(end_time) / 1000  # in seconds
    first_token_latency = (end_time - start_time)   # in seconds
    
    
# Measure the second token latency
with torch.no_grad():
    #start_time = torch.cuda.Event(enable_timing=True)
    #end_time = torch.cuda.Event(enable_timing=True)
    #start_time.record()
    start_time = time.time()
    output = model.generate(input_ids, max_length=input_ids.size(-1) + 2, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    #end_time.record()
    end_time = time.time()
    #torch.cuda.synchronize()
    #second_token_latency = start_time.elapsed_time(end_time) / 1000  # in seconds
    second_token_latency = (end_time - start_time) 

# Measure the throughput
num_tokens = 100
with torch.no_grad():
    #start_time = torch.cuda.Event(enable_timing=True)
    #end_time = torch.cuda.Event(enable_timing=True)
    #start_time.record()
    start_time = time.time()
    output = model.generate(input_ids, max_length=input_ids.size(-1) + num_tokens, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    print(output)
    #end_time.record()
    end_time = time.time()
    #torch.cuda.synchronize()
    #throughput = num_tokens / (end_time.elapsed_time(start_time) / 1000)  # tokens per second
    throughput = num_tokens / (end_time - start_time) 
    
# Print the results
print(f"First token latency: {first_token_latency:.4f} seconds")
print(f"Second token latency: {second_token_latency:.4f} seconds")
print(f"Throughput: {throughput:.2f} tokens/second")
