Steps to Run:

1. Download docker image:

   `docker pull pwatters991/ezkl-workloads:1.0`

2.  Start a ZKP Worker Service:

   1. Start the docker container by running `docker run -it --rm pwatters991/ezkl-workloads:1.0` in the terminal. Upon startup, the docker container should clone  v2 branch of repo https://github.com/pw-02/ezkl-exploration.git

   2.  Run`python distrubuted_proving/worker.py --port 50053` to start a worker service which will listen on port 50053. Change the  port number if necessary.  

      

3. You will need at least two terminals. One of the terminals will run the dispatcher, and the other others will run a worker service to which the dispatcher can submit work. 

    
