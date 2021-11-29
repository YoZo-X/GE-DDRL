from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import networkx as nx
import numpy as np
import gensim
import pkg_resources
import func_new as func
import Env

loss_history = []

class MyCallback(CallbackAny2Vec):
    def __init__(self):
        global loss_history
        loss_history = []
        self.epoch = 0
        self.loss_to_be_subed = 0
    def on_batch_end(self, model):
        pass

    def on_epoch_end(self, model):
        # loss = model.get_latest_training_loss()
        # loss_now = loss - self.loss_to_be_subed
        # self.loss_to_be_subed = loss
        # loss_history.append([self.epoch, loss_now])
        # print('epi{}:{}'.format(self.epoch, loss_now))
        self.epoch += 1
        tmpSum = 0
        for i in range(5):
            prob = model.predict_output_word([i])
            for j in range(len(prob)):
                if prob[j][0] == i + 1:
                    tmpSum += np.log(prob[j][1])
                    break
        loss_history.append([self.epoch, tmpSum])
        print('epi{}:{}'.format(self.epoch, tmpSum))



class Path2Vec:
    def __init__(self, env, dimensions: int = 10, walk_length: int = 100, num_walks: int = 10000,
                 workers=8, epsilon: float = 0.01, policy_func="LET", epi=0.01, save_path=None, load_path=None,
                 is_gen =True, seed=808):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param epsilon: Return hyper parameter (default: 0.1)
        :param workers: Number of workers for parallel execution (default: 1)
        """

        self.env = env
        self.termination = env.termination
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.epsilon = epsilon
        self.num_walks = num_walks
        self.walks = []
        self.workers = workers
        np.random.seed(seed)

        if load_path:
            if is_gen:
                self.walks = self.generate_policy_walks(num_walks=self.num_walks, policy_func=policy_func, epi=epi)
            self.model = Word2Vec.load(load_path)
        else:
            self.walks = self.generate_policy_walks(num_walks=self.num_walks, policy_func=policy_func, epi=epi)

            # self.model = Word2Vec(sg=1, vector_size=self.dimensions, window=1, min_count=1, workers=self.workers, seed=808)
            # self.model.build_vocab(self.walks)
            # self.model.train(self.walks, epochs=100, total_examples=self.model.corpus_count,
            #                  compute_loss=True, callbacks=[callback])

            callback = MyCallback()
            # self.walks = [[0, 1, 2, 3, 4]] * 1000
            self.model = self.fit(window=1, min_count=1, epochs=30, batch_words=10000, compute_loss=False)#, callbacks=[callback])
            if save_path:
                self.model.save(save_path)
            # np.save("GE-DDRL_Result/ntest/embedding_loss_d={}.npy".format(self.dimensions), loss_history)


    def get_dijkstra_action(self, state):
        path = self.env.LET_plan[state - 1]
        action = [path[0], path[1]]
        return action

    def generate_policy_epsilon_walk(self, epsilon=0.01, policy_func="LET"):
        state_idx = self.env.reset()-1
        walk = [state_idx]
        if policy_func == "LET":
            while state_idx != self.termination - 1 and len(walk) < self.walk_length:
                actions = self.env.get_actions(state_idx+1)
                if np.random.rand() <= epsilon:
                    idx = np.random.randint(0, len(actions))
                    action = actions[idx]
                else:
                    action = self.get_dijkstra_action(state_idx + 1)

                while not self.env.LET_plan[action[1]]:
                    idx = np.random.randint(0, len(actions))
                    action = actions[idx]

                state_idx = action[1]
                walk.append(state_idx)
        else:
            while state_idx != self.termination - 1 and len(walk) < self.walk_length:
                actions = self.env.get_actions(state_idx+1)
                if np.random.rand() <= epsilon:
                    idx = np.random.randint(0, len(actions))
                    action = actions[idx]
                else:
                    action = policy_func(env=self.env, state=state_idx + 1)

                while not self.env.LET_plan[action[1]]:
                    idx = np.random.randint(0, len(actions))
                    action = actions[idx]

                state_idx = action[1]
                walk.append(state_idx)
        return walk

    def generate_policy_walks(self, num_walks, policy_func="LET", epi=0.01) -> list:
        """
        Generates the policy walks which will be used as the skip-gram input.
        :return: policy of walks. Each walk is a list of nodes.
        """

        walks = []
        count_walk = 0
        while True:
            walk = self.generate_policy_epsilon_walk(epi, policy_func)
            if len(walk) <= self.walk_length:
                walk.extend([self.termination - 1] * (self.walk_length - len(walk)))
            # walk = [str(i) for i in walk]
            walks.append(walk)
            count_walk += 1
            if count_walk >= num_walks:
                break
        return walks

    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameters for gensim.models.Word2Vec - do not supply 'size' / 'vector_size' it is
            taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        # Figure out gensim version, naming of output dimensions changed from size to vector_size in v4.0.0
        gensim_version = pkg_resources.get_distribution("gensim").version
        size = 'size' if gensim_version < '4.0.0' else 'vector_size'
        if size not in skip_gram_params:
            skip_gram_params[size] = self.dimensions

        if 'sg' not in skip_gram_params:
            skip_gram_params['sg'] = 1

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)

    def soft_update(self, tau=0.01, policy_func="LET", save_path=""):
        if len(self.walks) < self.num_walks:
            self.walks = self.generate_policy_walks(num_walks=self.num_walks, policy_func=policy_func)
        else:
            num_update = int(tau * self.num_walks)
            self.walks = self.generate_policy_walks(num_walks=num_update, policy_func=policy_func) + self.walks[num_update:]
        assert len(self.walks) == self.num_walks
        self.model.train(self.walks, epochs=30, total_examples=self.model.corpus_count)
        if save_path:
            self.model.save(save_path + "update_GE")

    def load(self, load_path):
        self.model = Word2Vec.load(load_path)

    def save(self, save_path):
        self.model.save(save_path)

    def get(self, state):
        return self.model.wv.get_vector(state - 1, norm=True).reshape(1, -1)

# node2vec = Path2Vec("SiouxFalls", 15, 10)
#
