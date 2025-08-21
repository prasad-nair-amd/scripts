#!/bin/bash
RESULTS_DIR="results"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Launch 8 processes pinned to cores 0-7
for i in $(seq 0 7); do
    #sleep 0.05
    for j in $(seq 0 3); do
    	echo "Launching instance $i_$j on CPU core $i"
    	numactl --physcpubind=$i python3 openai-embedd_multi.py --instance_id $i --program_id $j --results_dir "$RESULTS_DIR" &
    done
done

wait
echo "All instances completed."

# Aggregate results
total_throughput=0
total_latency=0
count=0

for metrics_file in "$RESULTS_DIR"/metrics_*.txt; do
    if [ -f "$metrics_file" ]; then
        line=$(cat "$metrics_file")
        throughput=$(echo "$line" | cut -d',' -f1)
        latency=$(echo "$line" | cut -d',' -f2)
        total_throughput=$(echo "$total_throughput + $throughput" | bc -l)
        total_latency=$(echo "$total_latency + $latency" | bc -l)
        count=$((count + 1))
    fi
done

if [ $count -gt 0 ]; then
    avg_throughput=$(echo "$total_throughput / $count" | bc -l)
    avg_latency_ms=$(echo "($total_latency / $count) * 1000" | bc -l)
    echo "===================================="
    echo " Average Throughput: $avg_throughput sentences/sec"
    echo " Average Latency: ${avg_latency_ms} ms"
    echo " Across $count instances"
    echo "===================================="
else
    echo "No metrics found."
fi
