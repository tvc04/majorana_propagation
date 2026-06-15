import re
import json

input_file = "out.txt"
output_file = "40_rcs.json"

n_qubits = 40
n_layers = 10

# Extract max bond values
data = []

pattern = re.compile(r"max bond\s*=\s*(\d+)")

with open(input_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            data.append(int(match.group(1)))

# Create JSON structure
result = {
    "n_qubits": n_qubits,
    "n_layers": n_layers,
    "data": data
}

# Write JSON file
with open(output_file, "w") as f:
    json.dump(result, f, indent=4)

print(f"Wrote {len(data)} bond values to {output_file}")
