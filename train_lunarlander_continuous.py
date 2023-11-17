import gym
from ddpg_cvetana import DDPGAgent
import numpy as np
import time

def main():
    env = gym.make('LunarLanderContinuous-v2')

    # Initialize agent
    agent = DDPGAgent(
        env,
        lr_actor=0.001,
        lr_critic=0.002,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100000,
        batch_size=64
    )

    episode_rewards = []
    n_episodes = 1000

    start_time = time.time()

    for ep in range(n_episodes):
        state, _ = env.reset()
        agent.reset_noise()  
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            # Store in buffer so we can use for training
            agent.store_transition(state, action, reward, next_state, done)
            # Update policies (if applicable), based on previous experiences
            agent.learn()
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        
        print(f"Episode: {ep}, Total Reward: {total_reward}")

        if ep % 10 == 0:
            print(f'############## Mean Reward: {np.mean(episode_rewards[-100:])} ##############')

        if len(episode_rewards) > 100 and np.mean(episode_rewards[-100:]) > 200:
            print("Termination criterion met, stopping training")
            break

    end_time = time.time()

    training_time = end_time - start_time

    print("Training time: {:.2f} seconds".format(training_time))

    with open('training_time.txt', 'w') as file:
      file.write(str(training_time))

    agent.actor.save_weights('lunarlander_ddpg_actor_weights_cvetana.h5')
    agent.critic.save_weights('lunarlander_ddpg_critic_weights_cvetana.h5')

    agent.target_actor.save_weights('lunarlander_ddpg_target_actor_weights_cvetana.h5')
    agent.target_critic.save_weights('lunarlander_ddpg_target_critic_weights_cvetana.h5')

    np.save('lunarlander_training_rewards_cvetana.npy', episode_rewards)
    np.save('lunarlander_training_actor_loss_cvetana.npy', agent.actor_loss)
    np.save('lunarlander_training_critic_loss_cvetana.npy', agent.critic_loss)
    
if __name__ == "__main__":
    main()
