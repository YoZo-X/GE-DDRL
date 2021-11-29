from FQF import *
import Env
import GE
import argparse
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-env", type=str, default="SiouxFalls", help="Name of the Network, default = SiouxFalls")
    parser.add_argument("-rsp", type=str, default="LET", help="the type of rsp, default = LET")
    parser.add_argument("-rsp_par", type=float, default=1.0, help="the config of rsp, default = 1.0")
    parser.add_argument("-start", type=int, default=None, help="specify if determine the start node, default = None")
    parser.add_argument("-end", type=int, default=15, help="the termination of path planning, default = 15")
    parser.add_argument("-episodes", type=int, default=3000, help="Number of episodes to train, default = 3000")
    parser.add_argument("-seed", type=int, default=1, help="Random seed to replicate training runs, default = 1")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for updating the DQN, default = 32")
    parser.add_argument("-layer_size", type=int, default=512, help="Size of the hidden layer, default=128")
    parser.add_argument("-n_step", type=int, default=1, help="Multistep IQN, default = 1")
    parser.add_argument("-N", type=int, default=32, help="Number of quantiles, default = 32")
    parser.add_argument("-memory_size", type=int, default=int(1e5), help="Replay memory size, default = 1e5")
    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate, default = 1e-3")
    parser.add_argument("-gamma", type=float, default=1, help="Discount factor gamma, default = 0.99")
    parser.add_argument("-tau", type=float, default=1e-2, help="Soft update parameter tat, default = 1e-3")
    parser.add_argument("-eps_episodes", type=int, default=2500, help="Linear annealed frames for Epsilon, default = 2500")
    parser.add_argument("-min_eps", type=float, default=0.05, help="Final epsilon greedy value, default = 0.025")
    parser.add_argument("-emb_size", type=float, default=64, help="the dimensions of embedding, default = 64")
    parser.add_argument("-save_path", type=str, default="weights/", help="Specify the location of saving trained network")
    args = parser.parse_args()

    SEED = args.seed
    BUFFER_SIZE = args.memory_size
    BATCH_SIZE = args.batch_size
    LAYER_SIZE = args.layer_size
    GAMMA = args.gamma
    TAU = args.tau
    LR = args.lr
    MAP_NAME = args.env
    TERMINATION = args.end
    RSP = args.rsp
    RSP_CONFIG = args.rsp_par
    EMBEDDING_DIMENTION = args.emb_size
    INPUT_SIZE = EMBEDDING_DIMENTION * 3
    N_STEP = args.n_step

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    env = Env.Env(MAP_NAME, TERMINATION, RSP_type=RSP, RSP_parameter=RSP_CONFIG)
    embedding15 = GE.Path2Vec(env=env, dimensions=EMBEDDING_DIMENTION, epi=0.01, is_gen=True)
    agent = DQN_Agent(input_size=INPUT_SIZE,
                      layer_size=128,
                      n_step=N_STEP,
                      BATCH_SIZE=BATCH_SIZE,
                      BUFFER_SIZE=BUFFER_SIZE,
                      LR=LR,
                      TAU=TAU,
                      GAMMA=GAMMA,
                      device=device,
                      embedding_model=embedding15,
                      seed=SEED)
    eps_fixed = False
    t0 = time.time()
    final_average = agent.all_train(env=env, episodes=args.episodes, eps_fixed=args.eps_episodes, origin=args.start,
                                    eps_episodes=args.eps_episodes, min_eps=args.min_eps, file_path=args.save_path)
    t1 = time.time()
    print("Training time: {}min".format(round((t1 - t0) / 60, 2)))


