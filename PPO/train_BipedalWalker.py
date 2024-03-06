import pandas as pd
import torch
import random
import numpy as np
import gym
from torch import nn
from ppo import PPOAgent

# This function calculates the Generalized Advantage Estimation (GAE) for the agent. GAE is used in 
# reinforcement learning to reduce variance while keeping bias low
# Reference: https://towardsdatascience.com/generalized-advantage-estimation-in-reinforcement-learning-bf4a957f7975
def calculate_gae(agent, next_obs, rewards, next_done, dones, values):
    # We do not want to calculate the gradients here since we're only trying to run inference,
    # not update the weights
    with torch.no_grad():
        # Get the next value based on current state, and then reshape it for compatibility with other tensors
        next_value = agent.get_value(next_obs).reshape(1, -1)
        # Initializes a tensor for storing advantage estimates 
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        # Iterates over the steps in reverse order, typical for GAE since it's a backward-looking estimate
        for t in reversed(range(num_steps)):
            # If we're looking at the last timestamp, we use next_value and next_done
            # otherwise we use values and dones
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            # Calculate the temporal difference error i.e. delta - the difference between the sum 
            # of the reward and the discounted next value, and the current value estimate
            delta = rewards[t] + agent.gamma * nextvalues * nextnonterminal - values[t]
            # Update the advantage estimate for timestep t. It's a weighted sum of the current delta and the 
            # discounted previous GAE estimate. We use gae_lambda = .95 as per most resources online
            advantages[t] = lastgaelam = delta + agent.gamma * agent.gae_lambda * nextnonterminal * lastgaelam
        return advantages

# Create and configure the BipedalWalker environment
# I use gym wrappers to clip the action to ensure it is within the parameters
# and to transform and normalize the observation and the reward
# Reference: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
def make_env(gamma):
    def thunk():
        env = gym.make("BipedalWalker-v3")
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

if __name__ == "__main__":
    seed = 1
    num_minibatches = 32
    update_epochs = 10

    # The number of steps to iterate over, i.e. the size of the "buffer"
    num_steps = 2048
    total_timesteps = 1000000

    minibatch_size = int(num_steps // num_minibatches)
    num_iterations = total_timesteps // num_steps
    
    # Initialize everything to a seed to decrease potential randomness
    # and ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(.99)]
    )

    # Initialize agent
    agent = PPOAgent(envs, learning_rate = 3e-4)

    # Initialize tensors to store all needed variables
    obs = torch.zeros((num_steps, 1) + envs.single_observation_space.shape)
    actions = torch.zeros((num_steps, 1) + envs.single_action_space.shape)
    logprobs = torch.zeros((num_steps, 1))
    rewards = torch.zeros((num_steps, 1))
    dones = torch.zeros((num_steps, 1))
    values = torch.zeros((num_steps, 1))

    # Initialize the beginning state
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs)
    next_done = torch.zeros(1)
    episode_rewards = []
    losses = []

    # Interact with the environment num_iterations times and collect observations.
    # These observations will be used to train the agent in the following for loop
    for iteration in range(1, num_iterations + 1):
        # Get a list of observations
        for step in range(0, num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            # Again, we instruct torch to not collect gradients since we're only running inference.
            # We're not calculating any weight changes in this for loop
            with torch.no_grad():
                action, logprob, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Make the next step based on the action suggested from our model
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # Determine if the current episode has ended, either due to termination (end of episode) or
            # truncation (time limit reached)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).view(-1)
            next_obs, next_done = torch.Tensor(next_obs), torch.Tensor(next_done)
    
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_rewards.append(info['episode']['r'])
                        print(f"episodic_return={info['episode']['r']}")

        # Calculate the GAE 
        advantages = calculate_gae(agent, next_obs, rewards, next_done, dones, values)
        # Compute the returns by adding the advantages to the value estimates
        returns = advantages + values

        # Prepare batches of all variables
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update params based on minibatches.
        # We do not use no_grad here since we do want to update the weights at this point based
        # on all of the episode observations we collected previously
        b_inds = np.arange(num_steps)
        # Pass through the entire batch of data collected "update_epochs" times
        for epoch in range(update_epochs):
            # Shuffle indices to ensure random min batches
            np.random.shuffle(b_inds)

            # Iterate over the data in mini-batches
            for start in range(0, num_steps, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # For the current mini-batch, the agent's model computes the log probabilities and values of the actions taken
                _, newlogprob, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                # Calculate the logration based on the formula 
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                ## Actor Loss
                pg_loss1 = -b_advantages[mb_inds] * ratio # standard policy gradient loss
                pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - agent.clip_coef, 1 + agent.clip_coef) # ppo clipped loss
                
                # The final actor gradient loss is the maximum of these two losses, averaged over the mini-batch
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)

                ## Critic Loss
        
                # Calculate both clipped and unclipped versions of the critic loss
                # and choose the maximum between them. Average the final critic loss
                # over the mini batch
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2 # unclipped critic loss
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -agent.clip_coef,
                    agent.clip_coef,
                ) 
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2 # clipped critic loss
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Combination of the actor & critic loss. We set the critic loss to be half of its value
                # based on a hyperparameter suggested by CleanRL
                loss = pg_loss + v_loss * 0.5

                losses.append(loss.clone().detach().numpy())
                # Reset the gradients of the optimizer before the backward pass
                agent.optimizer.zero_grad()
                # Perform backprop to compute the gradients of the loss with respect to the model parameters
                loss.backward()
                # Apply gradient clipping to stabilize training
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                # Update the model parameters using the computed gradients
                agent.optimizer.step()

    episode_rewards = pd.DataFrame(episode_rewards)
    episode_rewards.to_csv('episode_rewards.csv')

    losses = pd.DataFrame(losses)
    losses.to_csv('losses.csv')

    # Store model weights
    torch.save(agent.actor_mean.state_dict(), 'bipedalwalker_actor_weights_cvetana.pth')
    torch.save(agent.critic.state_dict(), 'bipedalwalker_critic_weights_cvetana.pth')
    torch.save(agent.state_dict(), 'bipedalwalker_model_cvetana.pth')

    envs.close()
