import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random
from collections import deque, namedtuple


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


def weight_init_xavier(layers):
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)


class FPN(nn.Module):
    """Fraction proposal network"""

    def __init__(self, layer_size, seed, num_tau=32, device="cpu"):
        super(FPN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_tau = num_tau
        self.device = device
        self.ff = nn.Linear(layer_size, num_tau)
        self.softmax = nn.LogSoftmax(dim=1)
        weight_init_xavier([self.ff])

    def forward(self, x):
        """
        Calculates tau, tau_ and the entropy

        taus [shape of (batch_size, num_tau)]
        taus_ [shape of (batch_size, num_tau)]
        entropy [shape of (batch_size, 1)]
        """
        q = self.softmax(self.ff(x))
        q_probs = q.exp()
        taus = torch.cumsum(q_probs, dim=1)
        taus = torch.cat((torch.zeros((q.shape[0], 1)).to(self.device), taus), dim=1)
        taus_ = (taus[:, :-1] + taus[:, 1:]).detach() / 2.

        entropy = -(q * q_probs).sum(dim=-1, keepdim=True)
        assert entropy.shape == (q.shape[0], 1), "instead shape {}".format(entropy.shape)

        return taus, taus_, entropy


class QVN(nn.Module):
    """Quantile Value Network"""

    def __init__(self, input_size, output_size, layer_size, seed, N=32, device="cpu"):
        super(QVN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.N = N
        self.n_cos = N * 2
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi * i for i in range(1, self.n_cos + 1)]).view(1, 1, self.n_cos).to(
            device)  # Starting from 0 as in the paper
        self.device = device
        self.head = nn.Linear(self.input_size, layer_size)  # cound be a cnn
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, output_size)
        weight_init([self.head, self.ff_1])

    def calc_cos(self, taus):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        cos = torch.cos(taus.unsqueeze(-1) * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos

    def forward(self, input):
        """Calculate the state embeddings"""
        return torch.relu(self.head(input))

    def get_quantiles(self, input, taus, embedding=None):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size=1)]

        """
        if embedding == None:
            x = self.forward(input)
        else:
            x = embedding
        batch_size = x.shape[0]
        num_tau = taus.shape[1]
        cos = self.calc_cos(taus)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        return out.view(batch_size, num_tau, 1)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, gamma, n_step=1, device="cpu"):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action_vec", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

    def add(self, state, action_vec, reward, next_state, done):
        """Add a new experience to memory."""
        self.n_step_buffer.append((state, action_vec, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action_vec, reward, next_state, done = self.calc_multistep_return()
            e = self.experience(state, action_vec, reward, next_state, done)
            self.memory.append(e)

    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma ** idx * self.n_step_buffer[idx][2]

        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], \
               self.n_step_buffer[-1][4]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).int().to(self.device)
        actions_vec = torch.from_numpy(np.vstack([e.action_vec for e in experiences if e is not None])).float().to(
            self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).int().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        return (states, actions_vec, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 input_size,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 device,
                 embedding_model,
                 seed,
                 N=32):
        """Initialize an Agent object.

        Params
        ======
            input_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.input_size = input_size  # = n_node * d_vec
        self.action_size = 1
        self.seed = random.seed(seed)
        self.tseed = torch.manual_seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.n_step = n_step
        self.entropy_coeff = 0.001
        self.N = N
        self.action_step = 4
        self.last_action = None
        self.embedding_model = embedding_model

        # FQF-Network
        self.qnetwork_local = QVN(input_size=self.input_size, output_size=1, layer_size=layer_size, seed=seed,
                                  device=device, N=self.N).to(device)
        self.qnetwork_target = QVN(input_size=self.input_size, output_size=1, layer_size=layer_size, seed=seed,
                                   device=device, N=self.N).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # print(self.qnetwork_local)

        self.FPN = FPN(layer_size=layer_size, seed=seed, num_tau=self.N, device=device).to(device)
        self.frac_optimizer = optim.RMSprop(self.FPN.parameters(), lr=LR * 0.000001, alpha=0.95, eps=0.00001)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, device=self.device, seed=seed,
                                   gamma=self.GAMMA, n_step=n_step)

    def action_encoder(self, env, action):
        x = np.hstack((self.embedding_model.get(action[0] + 1), self.embedding_model.get(action[1] + 1),
                       self.embedding_model.get(env.termination)))
        x = torch.from_numpy(x).float()
        return x

    def step(self, env, state, action, reward, next_state, done):
        # Save experience in replay memory
        action_vec = self.action_encoder(env, action)
        self.memory.add(state, action_vec, reward, next_state, done)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            frac_loss, loss = self.learn(env, experiences)
            self.Q_updates += 1
            return frac_loss, loss
        return None, None

    def calc_fraction_loss(self, FZ_, FZ, taus):
        """calculate the loss for the fraction proposal network """

        gradients1 = FZ - FZ_[:, :-1]
        gradients2 = FZ - FZ_[:, 1:]
        flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
        flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
        gradients = (torch.where(flag_1, gradients1, - gradients1) + torch.where(flag_2, gradients2, -gradients2)).view(
            self.BATCH_SIZE, FZ.shape[1])
        assert not gradients.requires_grad
        loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
        return loss

    def calculate_huber_loss(self, td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        assert loss.shape == (td_errors.shape[0], self.N, self.N), "huber loss has wrong shape"
        return loss

    def get_optimal_action(self, env, state, is_plan=False):
        actions = env.get_actions(state)
        """Returns actions for given state as per current policy"""
        # Epsilon-greedy action selection
        self.qnetwork_local.eval()
        actions_values = None
        F_Zs_Target = []
        F_Zs = []
        Taus = []
        Taus_ = []
        for action in actions:
            if not env.LET_plan[action[1]]:
                action_value = torch.tensor([[-float("inf")]]).to("cuda:0")
                torch.tensor([[-float("inf")]]).to("cuda:0")
                F_Z_target = torch.tensor([-float("inf")] * self.N).to("cuda:0").view(-1, 1).view(1, self.N, 1)
                F_Z = torch.tensor([-float("inf")] * self.N).to("cuda:0").view(-1, 1).view(1, self.N, 1)
                taus_ = torch.tensor(np.arange(0, 1, 1/self.N)).to("cuda:0").view(-1, 1).view(1, self.N, 1)
                taus = torch.tensor(np.arange(0, 1+1/self.N, 1/self.N)).to("cuda:0").view(-1, 1).view(1, self.N+1, 1)
            else:
                with torch.no_grad():
                    action_vec = self.action_encoder(env, action)
                    features = self.qnetwork_local.forward(action_vec.to(self.device))
                    taus, taus_, entropy = self.FPN(features)
                    F_Z = self.qnetwork_local.get_quantiles(action_vec, taus_, features)
                    F_Z_sort, _ = torch.sort(F_Z.view(-1, 1), dim=0, descending=True)
                    F_Z_sort = F_Z_sort.view(action_vec.shape[0], taus_.shape[1], 1)
                    F_Z_target = self.qnetwork_target.get_quantiles(action_vec, taus_, features)
                    # F_Z_tau = self.qnetwork_local.get_quantiles(state, taus[:, 1:-1], features.detach())
                    if env.RSP_type == "dij":
                        action_value = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * F_Z_sort).sum(1)
                    elif env.RSP_type == "LET":
                        action_value = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * F_Z_sort).sum(1)
                    elif env.RSP_type == "SOTA":
                        SOTA_T = -env.get_ave_time(env.LET_plan[action[0]])
                        SOTA_T = SOTA_T * float(env.RSP_parameter)
                        index = (F_Z_sort <= SOTA_T).nonzero(as_tuple=False)
                        if index.__len__() == 0:
                            index = 0
                            dis = (SOTA_T - F_Z_sort[0][-1]) / (0 - F_Z_sort[0][- 1])
                            action_value = ((1 - dis) * taus_[0][0]).reshape(1, 1)
                        else:
                            index = index[0][1]
                            dis = (SOTA_T - F_Z_sort[0][index - 1]) / (F_Z_sort[0][index] - F_Z_sort[0][index - 1])
                            action_value = (dis * taus_[0][index] + (1 - dis) * taus_[0][index - 1]).reshape(1, 1)
                    elif env.RSP_type == "alpha":
                        index = (taus_ >= env.RSP_parameter).nonzero(as_tuple=False)
                        if index.__len__() == 0:
                            index = -1
                        else:
                            index = index[0][1]
                        dis = (env.RSP_parameter - taus_[0][index - 1]) / (taus_[0][index] - taus[0][index - 1])
                        action_value = (dis * F_Z_sort[0][index] + (1 - dis) * F_Z_sort[0][index - 1]).reshape(1, 1)
                    elif env.RSP_type == "mean-std":
                        mean = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * F_Z_sort).sum(1)
                        mean2 = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * F_Z_sort ** 2).sum(1)
                        variance = mean2 - mean ** 2
                        std = torch.sqrt(variance)
                        action_value = mean - env.RSP_parameter * std
            F_Zs_Target.append(F_Z_target)
            F_Zs.append(F_Z)
            Taus_.append(taus_)
            Taus.append(taus)

            if actions_values is None:
                actions_values = action_value
            else:
                actions_values = torch.cat((actions_values, action_value))
        assert actions_values.shape == (len(actions), 1)
        optimal_action_idx = np.argmax(actions_values.cpu().data.numpy())
        self.qnetwork_local.train()

        optimal_action = actions[optimal_action_idx]
        if optimal_action[1]+1 in env.walk and is_plan:
            tmp = env.LET_plan[state - 1]
            optimal_action = [tmp[0], tmp[1]]
        # print(actions, optimal_action)
        return optimal_action, action_value, F_Zs_Target[optimal_action_idx], F_Zs[optimal_action_idx], Taus[optimal_action_idx], \
               Taus_[optimal_action_idx]

    def act(self, env, state, eps=0.):
        """Returns actions for given state as per current policy"""
        # Epsilon-greedy action selection
        actions = env.get_actions(state)
        if random.random() > eps:  # select greedy action if random number is higher than epsilon or noisy network is used!
            if env.RSP_type == "dij":
                tmp = env.LET_plan[state - 1]
                optimal_action = [tmp[0], tmp[1]]
            else:
                optimal_action, _, _, _, _, _ = self.get_optimal_action(env, state)
        else:
            optimal_action_idx = np.random.choice(len(actions))
            optimal_action = actions[optimal_action_idx]
        while not env.LET_plan[optimal_action[1]]:
            optimal_action_idx = np.random.choice(len(actions))
            optimal_action = actions[optimal_action_idx]
        return optimal_action

    def learn(self, env, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions_vec, rewards, next_states, dones = experiences
        features = self.qnetwork_local.forward(actions_vec)
        taus, taus_, entropy = self.FPN(features.detach())

        # Get expected Q values from local model
        F_Z_expected = self.qnetwork_local.get_quantiles(actions_vec, taus_, features)
        Q_expected = F_Z_expected
        assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)

        # calc fractional loss
        with torch.no_grad():
            F_Z_tau = self.qnetwork_local.get_quantiles(actions_vec, taus[:, 1:-1], features.detach())
            FZ_tau = F_Z_tau

        frac_loss = self.calc_fraction_loss(Q_expected.detach(), FZ_tau, taus)
        entropy_loss = self.entropy_coeff * entropy.mean()
        frac_loss += entropy_loss

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            F_Z_next = None

            for i in range(len(next_states)):
                if not dones[i]:
                    _, _, F_Z_next_single, _, _, _ = self.get_optimal_action(env, next_states[i].cpu().item())
                else:
                    F_Z_next_single = torch.zeros(1, self.N, 1).to(self.device)
                if F_Z_next == None:
                    F_Z_next = F_Z_next_single
                else:
                    F_Z_next = torch.cat((F_Z_next, F_Z_next_single))

            Q_targets_next = F_Z_next.expand(self.BATCH_SIZE, self.N, 1).transpose(1, 2)
            Q_targets = rewards.unsqueeze(-1) + (
                    self.GAMMA ** self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
        huber_l = self.calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus_.unsqueeze(-1) - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1)
        loss = loss.mean()

        # Minimize the frac loss
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()

        # Minimize the huber loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return frac_loss.detach().cpu().numpy(), loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

    def load_weights(self, fpn_file, qvn_file):
        self.qnetwork_local.load_state_dict(torch.load(qvn_file))
        self.qnetwork_target.load_state_dict(torch.load(qvn_file))
        self.FPN.load_state_dict(torch.load(fpn_file))

    def full_path(self, env, origin, is_plan=False):
        env.reset(origin)
        state = origin
        done = 0
        while not done:
            action, _, _, _, _, _ = self.get_optimal_action(env, state, is_plan)
            state, reward, done = env.step(action)
            # print(len(env.walk))
            # assert len(env.walk) < 100
            assert len(env.walk) <= env.state_size
        return env.total_reward, env.walk

    def eval_runs(self, env, eps):
        """
        Makes an evaluation run with the current epsilon
        """
        reward_batch = []
        for _ in range(10):
            state = env.reset()
            while True:
                action = self.act(env=env, state=state, eps=eps)
                state, reward, done = env.step(action)
                if done:
                    break
            reward_batch.append(env.total_reward)

        return np.mean(reward_batch)

    def parameter_save(self, file_path="", pre_name=""):
        torch.save(self.qnetwork_local.state_dict(), file_path + pre_name + "QVN" + ".pth")
        torch.save(self.FPN.state_dict(), file_path + pre_name + "FPN" + ".pth")

    def all_train(self, env, episodes=1500, origin=None, eps_fixed=False, eps_episodes=500, min_eps=0.05, file_path=""):
        """
        Train FPN and QVN under current environment
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        output_history = []
        # frame = 0
        if eps_fixed:
            eps = 0
        else:
            eps = 1
        eps_start = 1
        i_episode = 1
        state = env.reset(state=origin)
        score = 0
        max_score = float('-inf')
        # max_score = self.eval_runs(env, 0.5)
        print("last_score = ", max_score)
        frame = 0
        # for frame in range(1, frames + 1):
        while True:
            frame += 1
            if i_episode >= episodes:
                break
            action = self.act(env=env, state=state, eps=eps)
            next_state, reward, done = env.step(action)
            # self.env.render()
            frac_loss, loss = self.step(env, state, action, reward, next_state, done)
            if loss:
                output_history.append((i_episode, frac_loss, loss, np.mean(scores_window)))
            state = next_state
            score += reward
            # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0
            # until the end of training
            if not eps_fixed:
                if i_episode < eps_episodes:
                    eps = max(eps_start - (i_episode * (1 / eps_episodes)), min_eps)
                else:
                    eps = max(min_eps - min_eps * ((i_episode - eps_episodes) / (episodes - eps_episodes)), 0.05)

            print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f} \tEpsilon: {:.2f}'
                  .format(i_episode, frame, np.mean(scores_window), eps), end="")
            if done:
                scores_window.append(score)
                scores.append(score)
                if i_episode % 100 == 0:
                    self.embedding_model.soft_update(policy_func=self.act, save_path=file_path)
                    print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f} \tEpsilon: {:.2f}'.format(i_episode, frame,
                                                                                        np.mean(scores_window), eps))
                i_episode += 1
                if eps <= 0.8 and np.mean(scores_window) >= max_score:
                    max_score = np.mean(scores_window)
                    self.parameter_save(file_path,
                                        "{}_{}".format(env.RSP_type, int(env.RSP_parameter * 100)) + "_optimal_")
                    self.embedding_model.save(
                        file_path + "{}_{}_".format(env.RSP_type, int(env.RSP_parameter * 100)) + "optimal_GE")
                state = env.reset(origin)
                score = 0

                self.parameter_save(file_path, "{}_{}_".format(env.RSP_type, int(env.RSP_parameter * 100)) + "final_")
                self.embedding_model.save(
                    file_path + "{}_{}_".format(env.RSP_type, int(env.RSP_parameter * 100)) + "final_GE")
        return output_history




