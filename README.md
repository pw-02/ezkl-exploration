

1. **Download Docker Image:**

   ```bash
   docker pull pwatters991/ezkl-workloads:1.0
   ```

2. **Start a ZKP Worker Service:**

   1. Open a terminal and start the Docker container by running:

      ```bash
      docker run -it --rm pwatters991/ezkl-workloads:1.0
      ```

      Upon startup, the Docker container should Git clone branch 'pw-dev' of the repo https://github.com/pw-02/ezkl-exploration.git.

   2. Run `hostname -i` and record the host IP of the container (this will be needed later).

   3. Run:

      ```bash
      python distrubuted_proving/worker.py --port 50052
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
      python distrubuted_proving/dispatcher.py model=mnist_classifier worker_addresses='["172.17.0.3:50052"]'
      ```

      The dispatcher will first confirm connection to each of the workers and then submit the proving job to them. See the terminal of the worker for updates on the proof generation progress. 

   4. For more CPU and memory-intensive examples (which cause out-of-memory issues on a 60GB machine), try the following :

      ```bash
      python distrubuted_proving/dispatcher.py model=mnist_gan worker_addresses='["172.17.0.3:50052"]'
      ​
      python distrubuted_proving/dispatcher.py model=mobilenet worker_addresses='["172.17.0.3:50052"]'
      ```

4. **Reporting**

   Once the proof has been computed the dispatcher will report all mettrics to `'/ezkl-exploration/distrubuted_proving/performance_logs.csv'` on the dispacther node.

   Sometimes the dispacther can lose connection with the worker (working on resolving this). If that happens, the worker may still run the proof job to completion and log its reporting metrics to `'/ezkl-exploration/distrubuted_proving/worker_log.csv'`. This same information is already in  `'/ezkl-exploration/distrubuted_proving/performance_logs.csv'` but we log twice for now for backup.



------

