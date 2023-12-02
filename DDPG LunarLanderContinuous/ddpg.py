import tensorflow as tf
import numpy as np
from utils import Actor, Critic, OrnsteinUhlenbeckActionNoise, ReplayBuffer
# Reference: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

class DDPGAgent:
    def __init__(self, env, lr_actor, lr_critic, gamma = .99,  tau = .001, buffer_capacity=1000, batch_size = 64):
        n_actions = env.action_space.shape[0]
        input_shape = env.observation_space.shape
        
        # For plotting purposes
        self.actor_loss = []
        self.critic_loss = []

        self.env = env 

        # Randomly initialize Critic network and Actor network, and the optimizers that will be used to update based on gradients
        self.actor = Actor(input_shape, n_actions)
        self.critic = Critic(input_shape)
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_critic)
        # Initialize OU noise as per
        # Although the paper sets the std_deviation at .2, I couldn't get the model to converge in less than 500 episodes with
        # that high std_deviation
        self.noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), std_deviation=0.05)

        # Initialize target Critic and Actor, used to stabilize training
        self.target_actor = Actor(input_shape, n_actions)
        self.target_critic = Critic(input_shape)
        # Copy random weights from original actor & critic into target to ensure they're all starting from the same point
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Initialize Replay Buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        # Set hyperparameters for learning and updates
        self.gamma = gamma
        self.tau = tau
        self.batch_size = 64

    # Add experience to the buffer so we can use it for learning later on
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    # Update the target networks, based on current network weights and hyperparameter tau
    # theta^q_prime = tau * theta^q + (1 - tau) * theta^q_prime
    # theta^mu_prime = tau * theta^mu + (1 - tau) * theta^mu_prime
    def update_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        new_weights = []
        target_actor_weights = self.target_actor.get_weights()
        for i, weight in enumerate(self.actor.get_weights()):
            # Calculate adjusted target_actor weights based on actor weights and tau
            new_weights.append(tau * weight + (1 - tau) * target_actor_weights[i])

        # Update target_actor weights based on calculated adjusted weights
        self.target_actor.set_weights(new_weights)

        new_weights = []
        target_critic_weights = self.target_critic.get_weights()
        for i, weight in enumerate(self.critic.get_weights()):
            # Calculate adjusted target_critic weights based on critic weights and tau
            new_weights.append(tau * weight + (1 - tau) * target_critic_weights[i])

        # Update target_critic weights based on calculated adjusted weights
        self.target_critic.set_weights(new_weights)
    
    # Reset the action exploration noise
    def reset_noise(self):
        self.noise.reset()

    # Choose an action based on the current policy with added noise for exploration
    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)

        # Get next action based on current actor policy & noise
        # a_t = mu(s_t|theta^mu) + N_t
        action = self.actor(state)[0].numpy()  
        noise = self.noise()
        action = action + noise

        # Ensure the chosen action is within the allowed bounds
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        return action

    # Update policies (if applicable), based on previous experiences. We only start updating once our buffer has had
    # enough experiences (at least batch_size)
    def learn(self):
        if self.buffer.size() < self.batch_size:
            # Do not proceed with learning if there aren't enough samples
            return 

        # Randomly sample a batch of previous experiences from the buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Update Critic
        with tf.GradientTape() as tape:
            # Compute targets
            target_actions = self.target_actor(next_states)
            target_q_values = tf.squeeze(self.target_critic(next_states, target_actions), 1)
            q_targets = rewards + self.gamma * target_q_values * (1 - dones)

            # Predict the current Q-values using the critic model
            critic_value = tf.squeeze(self.critic(states, actions), 1)

            # Calculate the MSE loss between the target Q-values and the predicted Q-values
            critic_loss = tf.keras.losses.MSE(q_targets, critic_value)

        # Compute the gradient of the loss with respect to the critic's trainable parameters
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        # Apply the gradients to the critic model's parameters
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        self.critic_loss.append(critic_loss.numpy())

        # Update Actor
        with tf.GradientTape() as tape:
            # Predict actions for the current states using the actor model
            new_actions = self.actor(states)
            # Compute the actor's loss as the negative mean of the critic's value predictions
            # This encourages the actor to produce actions that maximize the critic's Q-value predictions
            actor_loss = -tf.reduce_mean(self.critic(states, new_actions))

        # Compute the gradient of the actor's loss with respect to the actor's trainble parameters
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        # Apply the gradients to the actor model's parameters
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.actor_loss.append(actor_loss.numpy())

        # Soft update the target networks towards the primary networks
        self.update_networks(self.tau)