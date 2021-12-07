# GE-DDRL-ROUTER
The source code for "GE-DDRL: Graph Embedding and DeepDistributional Reinforcement Learning for ReliableShortest Path: A Universal and Scale Free Solution".

## Table of Contents
- [GE-DDRL-Router](#ge-ddrl-router)
  - [Requirements](#requirements)
  - [Framework of GE-DDRL-Router](#framework_of_GE-DDRL-Router)
  - [How to use](#how_to_use)
  - [Template](#template)

### 1.Requirements
  numpy, cvxopt, scipy, networkx, gensim, pytorch and other common packages.
  
### 2.Framework of GE-DDRL-Router
![image](https://github.com/YoZo-X/GE-DDRL/blob/master/img_files/figure1.png)

### 2. How to use
  **step 1**: Create a simulative environment(ENV), you must choose one transaction data or make a data by yourself. For some transaction data without the sigma of link travel time, you can generate them through the funtions in func.py, you can find the method you want in func.py or you can make your own method to generate sigma of link travel time;

  **step 2**: Create a graph embedding module(GE) base on the environment that is created on **step 1**, some parameters are required to set;

  **step 3**: Create a Agent based on the ENV and GE that are created on **step 1** and **step2**;

  **step 4**: Configure the parameter of the Agent that is created on **step 3**;
  
  **step 5**: (Optional)using dijkstra to pretrain the Agent(Imitation Learning), which is a warm start for DRL-Router.

   **step 6**: Start the training of the Agent, we need to set the training parameters num_iterations, When the training is finished, we got a Policy. The more training times, the better performance of results will be.

### 3. Running
  You can start your tranning in the main.py.

