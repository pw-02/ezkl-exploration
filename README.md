

1. **Download Docker Image:**

   ```bash
   docker pull pwatters991/ezkl-workloads:1.0
   ```

2. **Start a ZKP Worker Service:**

   1. Open a terminal and start the Docker container by running:

      ```bash
      docker run -it --rm pwatters991/ezkl-workloads:1.0
      ```

      Upon startup, the Docker container should Git clone branch 'pw-dev' of repo https://github.com/pw-02/ezkl-exploration.git.

   2. Run `hostname -i` and record the host IP of the container (this will be needed later).

   3. Run:

      ```bash
      python distributed_proving/worker.py --port 50052
      ```

      to start a worker service on port 50052. You can change the port number if necessary. If successful, a 'Worker started' message will appear in the terminal

   (Optional) An arbitrary number of workers can be launched by repeating the steps above. Workers can be run from any server/machine available in the cluster.

3. **Running the Dispatcher and Kicking off a Proof Generation Task:**

   1. Open another terminal and start another instance of the Docker container by running:

      ```bash
      docker run -it --rm pwatters991/ezkl-workloads:1.0
      ```

   2. Run:

      ```bash
      cat /ezkl-exploration/conf/config.yaml
      ```

      to view the config file. You do not need to make any changes to this file as the values can be set via the command line when launching a job, but there are two key settings to be aware of:

      - **model** - This is the name of the model to generate a proof for. The default is `mobilenet`. Other options include `mnist_classifier`, `mnist_gan`, `little_transformer`. The directory `/ezkl-exploration/config/model` contains a .yaml file for each of these models and within it is a file path pointing to the model ONNX file and an example input. You should not need to modify these.
      - **worker_addresses** - This is a list of worker addresses which the dispatcher will submit proving tasks to. By default, we assume that a single worker is running on `localhost:50052`. If the worker is running on a different machine/port, add the correct `hostname:port`. If the worker is running on the same machine but inside a different Docker container, get the hostname of that container (step 2.2) and set the worker address accordingly. For example, if the output of step 2.2 is `172.17.0.2` and the worker is running on port `50052`, then the address will be `172.17.0.2:50052`.
      - If there are multiple workers, add all their addresses to the list.

   3. To launch a simple end-to-end proof generation task and confirm everything is working, use the following command (remember to update the `worker_addresses`):

      ```bash
      python distributed_proving/dispatcher.py model=mnist_classifier worker_addresses='["172.17.0.3:50052"]'
      ```

      The dispatcher will first confirm connection to each of the workers and then submit the proving job to them. See the terminal of the worker for updates on the proof generation progress. 

   4. For more CPU and memory-intensive examples (which cause out-of-memory issues on a 60GB machine), try the following :

      ```bash
      python distributed_proving/dispatcher.py model=mnist_gan worker_addresses='["172.17.0.3:50052"]'
      ​
      python distributed_proving/dispatcher.py model=mobilenet worker_addresses='["172.17.0.3:50052"]'
      ```

      Here’s a refined version of your README section on running with model splitting:

---

### 4.  **Compute Proofs with Model Splitting**

To enable model splitting while running a proof, use the `model.split_group_size` parameter to your command. This setting asks, 'After splitting the global model into the maximum number of splits, how many should I now combine to process together? 

   - If `model.split_group_size` is set to `null` no splitting will occur, and the global model will be submitted for proving as a single entity.

 ```bash
     # Use 1 workers to prove MobileNet with splits being processed as pairs 
     # (maximum splits = 100, model_split_group_size = 2,  50 proofs to compute in total)  
     ​
     python distributed_proving/dispatcher.py model=mobilenet model.split_group_size=2 worker_addresses ["172.17.0.3:50052"]
     
     # Use 2 workers to prove mnist_gan with splits being processed as triplets 
     # (maximum splits = 12, model_split_group_size = 3, resulting in 4 proofs to compute)  
     python distributed_proving/dispatcher.py model=mnist_gan model.split_group_size=3 worker_addresses=["172.17.0.3:50052", "172.17.0.3:50053"]
 ```

### 5. **Testing / Investigation**

   - ***Specifying splits to combine:*** To instruct the dispatcher to group specific sets of splits you can provide a list of lists to configuration setting  `model.group_splits`. Each subsist should contain the IDs of the splits you wish to combine. The ID of a split corresponds to its position in the overall list of model nodes. For example, if you are processing MNIST GAN with a default split group size of 2, but you want to force the dispatcher to group splits [1, 2, 3] together as one group, while all other groups remain at the default size of 2, use the following:

   ```bash
   python distributed_proving/dispatcher.py model=mnist_gan model.split_group_size=2 model.group_splits=[[1,2,3]] worker_addresses=["localhost:50052"]
   ```

   - ***Spot Test:***
     To force the dispatcher to compute proofs only for the specified groups given `model.group_splits` you can set the `spot_test` configuration setting to `True`. For instance, the following command will only compute a proof for the group  `[1, 2, 3]`. This is useful for debugging as it allow us to target specific proving tasks within the overall collection for a model. 

   ```bash
   python distributed_proving/dispatcher.py model=mnist_gan model.split_group_size=2 model.group_splits=[[1,2,3]] worker_addresses=["localhost:50052"] spot_test=True
   ```

------

### 5. **Reporting**

- **Overall Performance Metrics:** The dispatcher will report metrics such as proving time, circuit size, resource usage to `exploration/logs/performance_logs.csv` on the dispatcher node. Each line in the report corresponds to a single proving operation performed by a worker. Note that if the exploration/logs/performance_logs.csv already exits, it will not be overwritten. The dispatcher will simply append a new row to the bottom of the file. 
- **FFT and MSMS**: The dispatcher will also report the size and duration of all FTT ad MSM operations that happened during the proving phase. These will be stored under `exploration/logs/ffts/{id of model for proving}.csv` and `exploration/logs/msms/{id of model for proving}.csv`. 