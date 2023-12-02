from reinforce_cvetana import REINFORCE
import gym
import numpy as np
import sys
import time
import pandas as pd

all_rewards = []
all_steps = []
training_loss = []

def main():
    ## Initialize environment and agent
    env = gym.make('CartPole-v0')
    agent = REINFORCE(env)

    ## Hyperparameters
    target_reward = 195
    n_episodes = 5000

    start_time = time.time()

    ## Train agent
    for episode in range(1, n_episodes+1):
        ## Generate an episode
        states, actions, rewards = agent.generate_episode()
        ## Update agent policy based on generated episode
        loss = agent.update(states, actions, rewards)

        ## Keep track of loss for plotting purposes
        training_loss.append(loss)

        ## Keep track of episode length
        steps = len(states)
        total_reward = np.sum(rewards)
        all_rewards.append(total_reward)
        all_steps.append(steps)
        if episode % 10 == 0:
              sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(total_reward, decimals = 3),  np.round(np.mean(all_rewards[-100:]), decimals = 3), steps))

        ## Check if the environment is solved
        if np.round(np.mean(all_rewards[-100:]), decimals = 3) >= target_reward:
            sys.stdout.write('Environment solved in {} episodes!'.format(episode))
            ## Store weights to be used in testing environment
            agent.model.save_weights('./trained_weights.ckpt')
            break

    end_time = time.time()

    loss_df = pd.DataFrame(training_loss, columns=['Loss'])
    reward_df = pd.DataFrame(all_rewards, columns=['Reward'])
    loss_df.to_csv('losses.csv', index=False)
    reward_df.to_csv('rewards.csv', index=False)

    training_time = end_time - start_time

    print("Training time: {:.2f} seconds".format(training_time))

    with open('training_time.txt', 'w') as file:
      file.write(str(training_time))
    
    env.close()

if __name__ == '__main__':
    main()
