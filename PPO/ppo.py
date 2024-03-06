import torch
import numpy as np

from torch import nn
from torch.distributions.normal import Normal

# Initialize PPOAgent class
class PPOAgent(nn.Module):
    """PPO Agent"""

    def __init__(self, envs, learning_rate, gamma = .99, gae_lambda = .95, clip_coef = .2):
        super().__init__()

        # Initialize Critic network. We use 3 fully connected layers, with tanh activation.
        # The final layer has only one neuron output since the critic outputs a single value
        # for the state 
        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        
        # Since this is continuous space, the action sampling process involves creating a 
        # probability distribution over possible actions.
        # We use the Normal distribution as the one to sample from, and we learn the 
        # mean of the distributiont through our Actor network. We also keep track of a learnable
        # parameter for the log standard deviation. 

        # Initialize Actor network. We use 3 fully connected layers, with tanh activation.
        # The final layer's output size matches the size of the action space
        self.actor_mean = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, np.prod(envs.single_action_space.shape)),
        )
        # Initialized to zero and broadcasted to match the shape of the action space
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        # Initialize an Adam optimizer for training, with the given learning rate & a small epsilon value for numerical stability
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

        # Initalize variables passed in from agent initialization. Most values are from the paper
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef

    # Takes an observation (x) as input and passes it through the critic network to get the value estimate
    def get_value(self, x):
        return self.critic(x)

    # Compute & return the action to be taken (sampled from a normal distribution based on the current policy), 
    # the log probability of that action, and the value estimate for a given state, as per the 
    # agent's current policy and value function
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x) # Get the actor mean per the current policy
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd) # Get the learned standard deviation
        probs = Normal(action_mean, action_std) # Generate the Normal distribution with mu and sigma
        if action is None:
            action = probs.sample()
        # We sum the log probabilities to aggregate them across all dimensions of the action space
        return action, probs.log_prob(action).sum(1), self.critic(x)