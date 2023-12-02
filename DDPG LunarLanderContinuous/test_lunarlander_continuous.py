import gym
import numpy as np
from ddpg_cvetana import DDPGAgent
import time

def main():
    ## We have to set render mode to get the UI
    env = gym.make('LunarLanderContinuous-v2', render_mode="human")

    agent = DDPGAgent(
        env,
        lr_actor=0.001,
        lr_critic=0.002,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100000,
        batch_size=64
    )

    ## We need to ensure model is built before we can load the pre-trained weights
    state = env.reset()[0]
    state = np.expand_dims(state, axis=0)  
    action = agent.actor(env.reset()[0])
    agent.critic(state, action)
    
    ## Load pre-trained weights
    agent.actor.load_weights('lunarlander_ddpg_actor_weights_cvetana.h5')
    agent.critic.load_weights('lunarlander_ddpg_critic_weights_cvetana.h5')

    for episode in range(10):
        state = env.reset()[0]
        done = False
        total_reward = 0
        
        ## Generate episode
        while not done:
            env.render()
            action = agent.choose_action(state)
            time.sleep(0.1)

            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        print(f'Episode {episode + 1}: Total Reward = {total_reward}')

    env.close()

if __name__ == "__main__":
    main()
