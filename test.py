import csv
def read_csv_into_dict(file_path):
    data = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key, value in row.items():
                data[key] = value
    return data
circuit_info = read_csv_into_dict('halo2_circuit.csv')

dict = read_csv_into_dict('halo2_circuit.csv')
print(dict)
