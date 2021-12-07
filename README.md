# ge-ddrl-router
The source code for "GE-DDRL: Graph Embedding and DeepDistributional Reinforcement Learning for ReliableShortest Path: A Universal and Scale Free Solution".

## Table of Contents
- [GE-DDRL-Router](#ge-ddrl-router)
  - [Requirements](#requirements)
  - [How to use](#how_to_use)
  - [Template](#template)

### 1.Requirements
  numpy, cvxopt, scipy, networkx, gensim, pytorch and other common packages.
  
### 2.Framework of GE-DDRL-Router


### 2. How to use
  **step 1**: Create a simulative environment, you must choose one transaction data or make a data by yourself. For some transaction data without the sigma of link travel time, you can generate them through the funtions in func.py, you can find the method you want in func.py or you can make your own method to generate sigma of link travel time;

  **step 2**: Create a graph embedding module base on the environment that is created on **step 1**, some parameters are required to set;

  **step 3**: Create a Agent of DRL-Router based on the Xtate that is created on **step 2**;

  **step 4**: Configure the parameter of the Agent that is created on **step 3**, where K(number of samples), lr_rate(learning rate), v_min(range of distribution) and termination parameters are necessary;

  **step 5**: We need use dijkstra to pretrain the Agent, which is a warm start for DRL-Router. We suggest turning on the dynamic learning rate(dynamic_lr = 1) during pre-training and running 1000 episodes;

   **step 6**: Start the training of the Agent, we need to set the training parameters num_iterations, obj(define RSP problem) and parameter(different parameter for different RSP problem). When the training is finished, we got a Policy. The more training times, the more accurate the Policy results will be.

### 3. Template
  The following is an example for how to configure a DRL-Router：
  ```Python
  import DRL_C51
  import func

  Map_Name = "SiouxFalls"

  Map_id = {"SiouxFalls": 0,
            "Anaheim": 2,
            "Winnipeg": 3,
            "Barcelona": 4}

  Map_1 = func.Map()
  Map_1.extract_map(Map_id[Map_Name])
  Map_1.G = func.convert_map2graph(Map_1)

  X = DRL_C51.Xtates(Map_1, num_atoms=51)
  agent = DRL_C51.DRL_Agent(X, Map, 15)
  agent.update_V(-200, 0)
  agent.dynamic_lr = 1
  agent.lr_rate = 0.01
  agent.K = 5
  agent.train_dijkstra(1000)
  agent.dynamic_lr = 0
  agent.lr_rate = 0.1
  agent.train_IS(2000, parameter=40, obj="mean-std")
  print("-----------shortest path-------------")
  agent.find_shortest_path(1, True)
  print("--------------C51 path---------------")
  agent.find_path(1, 40, "mean-std", True)
  ```

