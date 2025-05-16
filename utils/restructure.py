
path = r"C:\Users\pw\Desktop\reports\mnist_classifier\2025-05-16_13-38-34\halo2_prover_cpu.csv"

import csv

# Input and output file paths
input_csv = f'{path}\halo2_prover_cpu.csv'
output_csv = f'{path}\halo2_prover_formatted.csv'

# Read the original CSV
with open(input_csv, newline='') as infile:
    reader = csv.DictReader(infile)
    rows = list(reader)

# Prepare reformatted data
formatted_rows = []

for row in rows:
    for key in row:
        if key.endswith('_time'):
            phase = key.replace('_time', '')
            time_value = row.get(f"{phase}_time", "")
            cpu_value = row.get(f"{phase}_cpu", "")
            formatted_rows.append({
                "phase": phase,
                "time": time_value,
                "cpu": cpu_value
            })

# Write the reformatted CSV
with open(output_csv, 'w', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["phase", "time", "cpu"])
    writer.writeheader()
    writer.writerows(formatted_rows)

print(f"✅ Reformatted CSV written to: {output_csv}")
