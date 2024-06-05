1. Download docker image:

   `docker pull pwatters991/ezkl-workloads:1.0`

2.  Start a ZKP Worker Service:

   1. Open a terminal and start the docker container by running `docker run -it --rm pwatters991/ezkl-workloads:1.0`. Upon startup, the docker container should Git clone branch 'v2' of repo https://github.com/pw-02/ezkl-exploration.git

   2.  Run`python distrubuted_proving/worker.py --port 50052` to start a worker service. The worker will listen on port 50052 for messages from the dispatcher. You can change the port number if necessary.  If successful a 'Worker started' message  will appear in the terminal like below:

      ![image-20240604222557810](C:\Users\pw\AppData\Roaming\Typora\typora-user-images\image-20240604222557810.png)

   3. (Optional) An arbitrary number of workers can be launched by repeating the steps above. Workers can be run from any server/machine that is available in the cluster. 

3. Running the dispatcher:

   1. Open another terminal and start another instance of the docker container by running  `docker run -it --rm pwatters991/ezkl-workloads:1.0`. If using a separate machine to run the dispacther you may need to downlaod the docker image again by running `docker pull pwatters991/ezkl-workloads:1.0`.

   2. Run `cat /ezkl-exploration/conf/conf.yaml` to view the config file. The important settings to be aware of are as follows:

      - **model** - this is the name of the model to generating a proof for. The default is `mnist_gan`.  Other options include `mobilenet, mnist_gan, mnist_classifier, little_transformer` (FYI: directory `/ezkl-exploration/conf/model` contains a .yaml file for each model and within is a file path pointing to the model onnx file and an example input. You should not need to modify these.)
      - **worker_addresses** - this is a list of worker addresses that the dispatcher will use to connect to the workers and submit messages. By default we assume that a  single worker is listening on localhost::50052. If the worker is running on a different machine/port then update the `hostname:port` as required. If there are multiple workers running, add all their addresses to the list.  

   3. Start the dispatcher by running  

      

   

4. You will need at least two terminals. One of the terminals will run the dispatcher, and the other others will run a worker service to which the dispatcher can submit work. 

    
