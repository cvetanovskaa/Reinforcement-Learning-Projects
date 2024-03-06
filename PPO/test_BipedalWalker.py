import torch
import numpy as np
import random
import gym

from ppo import PPOAgent

# Generate environment per training script setup, with human-readable render mode
def make_env(gamma):
    def thunk():
        env = gym.make("BipedalWalker-v3", render_mode = "human")
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
    
    # Initialize everything to the same seed as training to decrease potential randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(.99)]
    )

    # Initialize agent
    agent = PPOAgent(envs, learning_rate = 3e-4)
    agent.load_state_dict(torch.load('bipedalwalker_model_cvetana.pth')) # Load model weights
    agent.eval() # Set the agent to evaluation mode 
    
    # Reset the environment to start the evaluation and initialize a list to keep track of episodic returns
    obs, _ = envs.reset()
    episodic_returns = []

    # Run 10 episodes
    while len(episodic_returns) < 10:
        actions, _, _= agent.get_action_and_value(torch.Tensor(obs))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
