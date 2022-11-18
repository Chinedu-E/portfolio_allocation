import datetime
import argparse

import numpy as np
import random
import matplotlib.pyplot as plt


from models import DDPGAgent
from environment import StockEnv, Portfolio


def parse_args():
    parser = argparse.ArgumentParser("DDPG algorithm for portfolio allocation")
    # Environment and Portfolio
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    parser.add_argument("--initial-amount", type=int, default=10_000, help="Starting amount for the portfolio")
    parser.add_argument("--mode", type=str, default="train", help="training or testing")
    parser.add_argument("--companies", type=list[str], default=["GOOGL", "AAPL", "MSFT", "ZION", "COST", "IBM"],
                        help="Companies to be chosen in portfolio")
    parser.add_argument("--window", type=int, default=14, help="Lookback window (in days) to be given to agent")
    parser.add_argument("--train_split", type=float, default=0.8, help="proportion of data to be used for training")
    
    # Core Algorithm parameters
    parser.add_argument("--policy", type=str, default="LSTM", help="one of CNN or LSTM policy")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="replay buffer size")
    parser.add_argument("--actor-lr", type=float, default=3e-2, help="actor learning rate for Adam optimizer")
    parser.add_argument("--critic-lr", type=float, default=1e-2, help="critic learning rate for Adam optimizer")
    parser.add_argument("--tau", type=float, default=1e-4, help="which seed to use")
    parser.add_argument("--discount-rate", type=float, default=0.995,
                        help="gamma value for discounted rewards")
    parser.add_argument("--epsilon-min", type=float, default=0.0,
                        help="minimum epsilon value for exploration")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="starting epsilon value for exploration")
    parser.add_argument("--epsilon-decay", type=float, default=0.0,
                        help="epsilon decay value for exploration")
 
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="total number of episodes to run the environment for")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=10,
                        help="number of iterations between every optimization step")
    parser.add_argument("--start-epoch", type=int, default=10,
                        help="number of epochs to pass before training starts")
 
    # Bells and whistles
    parser.add_argument("--prioritized", type=bool, default=True,
                        help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized-alpha", type=float, default=0.5,
                        help="alpha parameter for prioritized replay buffer")
    parser.add_argument("--prioritized-beta0", type=float, default=0.4
                        , help="initial value of beta parameters for prioritized replay")
    parser.add_argument("--prioritized-eps", type=float, default=1e-6,
                        help="eps parameter for prioritized replay buffer")
    # Checkpointing
    parser.add_argument("--save-freq", type=int, default=1,
                        help="save model once every n epochs")
    parser.add_argument("--load-on-start", type=bool, default=True,
                        help="if true and model was previously saved then training will be resumed")

    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    init_weights = np.random.random(len(args.companies))
    paths = [f"weights/{args.policy}-{args.window}-actor.h5",
             f"weights/{args.policy}-{args.window}-actor_target.h5",
             f"weights/{args.policy}-{args.window}-critic.h5",
             f"weights/{args.policy}-{args.window}-critic_target.h5"]

    
    
    agent= DDPGAgent(window_length=args.window, n_stocks=len(args.companies), actor_lr=args.actor_lr,
                     critic_lr=args.critic_lr, tau=args.tau, discount_rate=args.discount_rate,
                     buffer_capacity=args.buffer_size, batch_size=args.batch_size, policy=args.policy)
    
    if args.load_on_start:
        try:
            agent.load_weights(paths)
        except:
            print("Could not find older models")
    all_rewards = []
    portfolios = []
    t = 0
    for epoch in range(args.num_epochs):
        portfolio = Portfolio(asset_names=args.companies, amount=args.initial_amount, weights=init_weights
                          ,window=args.window, train_split=args.train_split, mode=args.mode)
    
        env = StockEnv(portfolio, mode=args.mode)
        state = env.reset()
        done = False
        rewards = []
        while not done:
            if t % args.window == 0:
                action = agent.make_action(state, t)
            nstate, reward, done, info = env.step(action)
            print(info)
            if t % args.window == 0:
                agent.memorize(state, action, reward, done, nstate)
                
            if epoch+1 >= args.start_epoch and t%args.learning_freq == 0:
                agent.learn()
            
            state = nstate
            rewards.append(reward)
            t += 1
        if epoch+1 % args.save_freq:
            agent.save_weights(paths)
        all_rewards.append(rewards)
        portfolios.append(portfolio)
        
    for rewards in all_rewards:
        plt.plot(rewards)
        plt.show()
    for portfolio in portfolios:
        portfolio.plot()
    
if __name__ == "__main__":
    main()
            