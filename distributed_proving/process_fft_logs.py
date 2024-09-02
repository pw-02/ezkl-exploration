
import os
import csv
import json
import pandas as pd


def get_prover_logs(prover_metrics_file):
    prover_metrics = {}
    with open('halo2_prover.csv', mode='r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        for key, value in row.items():
                            key = f"halo2_{key}"
                            prover_metrics[key] = value

    return prover_metrics



def process_msm(msm_file='halo2_msm_times.csv'):
    if os.path.isfile(msm_file):
        msm_metrics = {}
        df = pd.read_csv(msm_file)  # Replace 'your_file.csv' with the actual file path
        # Calculate the total number of MSMs
        msm_metrics['total_msms'] = len(df)
        msm_metrics['total_msm_time(s)'] = df['device'].sum()
        msm_metrics['largest_msm'] = df['num_coeffs'].max()
        msm_metrics['avg_msm_duration'] = df['device'].mean()
        # Calculate the average duration

        # Convert the DataFrame to a dictionary
        data_dict = df.to_dict(orient='records')  # 'records' format creates a list of dictionaries
        msm_data = json.dumps(data_dict)
        msm_metrics['msm_data'] = msm_data

        return msm_metrics
    pass

def process_fft (fft_file='halo2_fft_times.csv'):
    if  os.path.isfile(fft_file):
        fft_metrics = {}
        df = pd.read_csv(fft_file)  # Replace 'your_file.csv' with the actual file path
        # Calculate the total number of FFTs
        fft_metrics['total_ffts'] = len(df)
        fft_metrics['total_fft_time(s)'] = df['duration(s)'].sum()
        fft_metrics['largest_fft'] = df['size'].max()
        fft_metrics['avg_fft_duration'] = df['duration(s)'].mean()
        # Calculate the average duration

        # Convert the DataFrame to a dictionary
        data_dict = df.to_dict(orient='records')  # 'records' format creates a list of dictionaries
        fft_data = json.dumps(data_dict)
        fft_metrics['fft_data'] = fft_data

        return fft_metrics
       
        pass
      
       
if __name__ == '__main__':
    prover_metrics =get_prover_logs("halo2_prover.csv")
    some_dict = {}
    some_dict.update(prover_metrics)