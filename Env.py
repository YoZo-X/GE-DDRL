import func_new as func
import networkx as nx
import numpy as np

Map_id = {"SiouxFalls": 0,
          "Anaheim": 2,
          "Winnipeg": 3,
          "Barcelona": 4,
          "Massachusetts": 5,
          "Berlin-Friedrichshain": 6}


class Env:
    def __init__(self, map_name, termination, RSP_type="LET", RSP_parameter=-1):
        if map_name in Map_id:
            self.map = func.Map()
            self.map.extract_map(Map_id[map_name], r_0=1, r_s=termination, kai=0.4)
            self.map.G = func.convert_map2graph(self.map)
        else:
            self.map = map_name
            self.map.G = func.convert_map2graph(self.map)
        self.RSP_type = RSP_type
        self.RSP_initial_parameter = RSP_parameter
        self.RSP_parameter = self.RSP_initial_parameter
        self.curr_state = 1
        self.walk = [self.curr_state]
        self.termination = termination
        self.state_size = self.map.M.shape[0]
        self.action_size = self.map.M.shape[1]
        self.total_reward = 0
        self.initial_counter = {}
        for i in range(self.map.n_node):
            self.initial_counter[i] = 0
        self.state_counter = self.initial_counter
        self.LET_plan = [None] * self.map.n_node
        for i in range(self.map.n_node):
            if not self.LET_plan[i]:
                if i in nx.nodes(self.map.G) and nx.has_path(self.map.G, i, self.termination - 1):
                    self.LET_plan[i] = nx.dijkstra_path(self.map.G, i, self.termination - 1, "weight")
                    for j in range(1, len(self.LET_plan[i])):
                        if not self.LET_plan[self.LET_plan[i][j]]:
                            self.LET_plan[self.LET_plan[i][j]] = self.LET_plan[i][j:]
                else:
                    self.LET_plan[i] = None

    def reset(self, state=None):
        if state:
            self.curr_state = state
        else:
            self.curr_state = np.random.randint(1, self.state_size + 1)
            while not self.LET_plan[self.curr_state - 1] or self.curr_state == self.termination:
                self.curr_state = np.random.randint(1, self.state_size + 1)
        self.walk = [self.curr_state]
        self.total_reward = 0
        # if self.RSP_type == "SOTA":
        #     self.RSP_parameter = -func.dijkstra(self.map.G, self.curr_state - 1, self.termination - 1)[0]
        # else:
        self.RSP_parameter = self.RSP_initial_parameter
        self.state_counter = self.initial_counter
        self.state_counter[self.curr_state - 1] = 1
        return self.curr_state

    def render(self):
        print("state = {}; total_reward = {}".format(self.curr_state, self.total_reward))

    def get_actions(self, state):
        actions = list(self.map.G.edges(state-1))
        return actions

    def get_full_time(self, path, K):
        res = []
        edge_path = []
        for i in range(len(path) - 1):
            edge_idx = self.map.G.get_edge_data(path[i]-1, path[i + 1]-1)["index"]
            edge_path.append(edge_idx)
        res = np.random.normal(self.map.mu[edge_path], self.map.sigma[edge_path], [len(edge_path), K])
        res = np.sum(res, 0)
        return res

    def get_ave_time(self, path):
        total_reward = 0
        for i in range(len(path) - 1):
            edge_idx = self.map.G.get_edge_data(path[i], path[i + 1])["index"]
            total_reward += self.map.mu[edge_idx].item()
        return total_reward

    def step(self, action, reward_type="normal"):
        done = 0
        edge_idx = self.map.G.get_edge_data(action[0], action[1])["index"]
        next_state = action[1] + 1
        if reward_type == "normal":
            reward = -self.map.get_edge_normal_cost(edge_idx, 1)
            while reward > 0:
                reward = -self.map.get_edge_normal_cost(edge_idx, 1)
        elif reward_type == "log":
            reward = -self.map.get_edge_log_cost(edge_idx, 1)
        if next_state == self.termination:
            done = 1
        # if self.RSP_type == "SOTA":
        #     self.RSP_parameter -= reward
        self.total_reward += reward
        self.curr_state = next_state
        self.walk.append(self.curr_state)
        self.state_counter[self.curr_state - 1] += 1
        return next_state, reward, done

