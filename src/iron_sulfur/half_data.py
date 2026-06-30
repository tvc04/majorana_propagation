import re
import json

input_file = "out.txt"
output_file = "Fe4S4_data/Fe4S4_aa_0.json"

n_qubits = 72
n_layers = 1
cutoff = 0

# Extract max bond values
data = []
latencies = []

with open(input_file, "r") as f:
    for line in f:
        match = re.search(r"max bond = ([0-9]+), latency = ([0-9.eE+-]+)", line)
        if match:
            data.append(int(match.group(1)))
            latencies.append(float(match.group(2)))

# Create JSON structure
result = {
    "n_qubits": n_qubits,
    "n_layers": n_layers,
    "cutoff": cutoff,
    "data": data,
    "latencies": latencies
}

# Write JSON file
with open(output_file, "w") as f:
    json.dump(result, f, indent=4)

print(f"Wrote {len(data)} bond values to {output_file}")
