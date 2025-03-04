## Deployment

1. **Clone the repository**

   ```bash
   git clone -b simple-client https://github.com/pw-02/ezkl-exploration.git
   ```

2. **Install Python libraries**:

   ```
   pip install -r requirements.txt
   ```

3. **Make 'to_run.sh' executable:**

   ```bash
    chmod +x to_run.sh
   ```

4. **Execute 'to_run.sh':**

   ```bash
    ./to_run.sh
   ```

   The 'to_run.sh' contains the commands to generate proofs based on the specified configuration (see config section below for information)

## Configurations :

| Parameter             | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| **model**             | The model for which to generate a proof. This should match the name of a `.yaml` file in the `conf/model` folder. The YAML file includes default values for the following: <br /> - **onnx_file**: Path to the ONNX model used for proving. <br /> - **input_file**: Path to the input data for inference. <br /> - **split_group_size**: Number of sub-models per group when splitting the model. For example, a model with 100 nodes and `split_group_size = 5` will be divided into 20 sub-models. To prove the model without splitting, set this value to `null` or `0`. <br /><br /> These values can be overridden via the command line. For example, to set `split_group_size`, use: <br /> `model.split_group_size={your_value}` |
| **cache_setup_files** | A Boolean (True/False) value indicating whether to save setup files between runs. If enabled, the files will be stored in a folder named `Cache` at the root directory. **Default is False**. |

## Reporting

All reports are saved in a `/reports/model/{name}/{timestamp}` directory, where `{name}` corresponds to the model's name and `{timestamp}` represents the time of execution. The `reports` folder is located at the root of the project. There are several different reports:

| Report Name                   | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `models_to_prove_summary.csv` | A summary of the models to be proven as part of the job. This report is generated before proof generation begins and contains information from the EZKL settings API. |
| `msms_summary.csv`            | A summary of MSM operations across all proven sub-models. |
| `ntts_summary.csv`            | A summary of NTT operations across all proven sub-models. |
| `halo2_prover_summary.csv`    | Performance metrics collected directly from Halo2 for each proven sub-model. |
| `ezkl_perf_summary.csv`       | Performance metrics of EZKL APIs. Note: EZKL includes some additional overhead compared to Halo2. |
| `halo2_circuit_summary.csv`   | A summary of the Halo2 circuit for each sub-model.           |