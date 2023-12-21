#!/bin/bash



file_to_delete="cpu_frequencies.csv"  # Replace with the file you want to delete

if [ -e "$file_to_delete" ]; then
  rm "$file_to_delete"
  echo "File '$file_to_delete' deleted."
else
  echo "File '$file_to_delete' does not exist."
fi


# Get the number of CPU cores
num_cores=$(lscpu | awk '/^CPU\(s\)/{print $2}')


# Loop through each CPU core to add core numbers to the header
for ((core=0; core < num_cores; core++)); do
    echo -n "CPU$core," >> cpu_frequencies.csv
done

echo -e '\n' >> cpu_frequencies.csv

while true; do
# Loop through each CPU core
    for ((core=0; core < num_cores; core++)); do
        # Get the current CPU frequency for the core
	freq=$(grep -E 'MHz' /proc/cpuinfo | awk -F ': ' '{print $2}' | tr '\n' ',' | sed 's/,$//')
	#freq="${freq}"$'\n'

        # Write the core number and frequency to the CSV file
        echo -e "$freq," >> cpu_frequencies.csv
	
	sleep 0.1
    done
done

