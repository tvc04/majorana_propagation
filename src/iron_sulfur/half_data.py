import re
import json

input_file = "out.txt"
output_file = "Fe4S4_data/Fe4S4_sq_0.json"
#output_file = "forward_prop/forward_prop_LUCJ.json"

n_qubits = 72
n_layers = 1
cutoff = 0

# Extract max bond values
data = []
latencies = []

with open(input_file, "r") as f:
    for line in f:
        match = re.search(r"Op\s+(\d+)\s*/\s*(\d+),\s*max bond\s*=\s*(\d+),\s*latency\s*=\s*([\d.]+)", line)
        if match:
            data.append(int(match.group(3)))
            latencies.append(float(match.group(4)))

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
