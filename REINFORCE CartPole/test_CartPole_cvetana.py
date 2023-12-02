from reinforce_cvetana import REINFORCE
import gym
import time

## We have to set render mode to get the UI
env = gym.make('CartPole-v0', render_mode="human")
agent = REINFORCE(env)

## Pre-load trained model
agent.model.load_weights('./trained_weights.ckpt')

for episode in range(10):
    state = env.reset()[0]
    done = False
    total_reward = 0
    
    ## Generate episode
    while not done:
        env.render()
        action = agent.get_action(state)
        time.sleep(0.1)

        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f'Episode {episode + 1}: Total Reward = {total_reward}')

env.close()
