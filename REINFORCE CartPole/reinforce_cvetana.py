import tensorflow as tf
import numpy as np

class REINFORCE:
    """REINFORCE Agent"""

    def __init__(self, env, alpha = 1e-4, gamma = 0.99):
        """
        Initialize agent variables
        """
        self.env = env

        ## Initialize environment params
        self.n_action = self.env.action_space.n
        self.n_observation = self.env.observation_space.shape[0]

        ## Initialize hyperparameters
        self.gamma = gamma

        ## Initialize the policy randomly (Neural Net) => `Initialize theta arbitrarily`
        ## It will take in an observation (i.e. current state), and produce probabilities for each action based on that state
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.n_observation,)),  # Fully connected hidden layer with 256 units
            tf.keras.layers.Dropout(0.2),  # Dropout layer for regularization
            tf.keras.layers.BatchNormalization(),  # Batch normalization layer, normalizes inputs
            tf.keras.layers.Dense(256, activation='relu'),  # Another fully connected hidden layer with 256 units
            tf.keras.layers.Dropout(0.2),  # Again dropout layer for regularization
            tf.keras.layers.BatchNormalization(),  # Again batch normalization layer
            tf.keras.layers.Dense(self.n_action, activation=tf.keras.activations.softmax)  # Output layer with softmax activation to get probabilities that the output is one of `n_action` classes
        ])

        ## Initialize optimizer. Adam is the most commonly used optimizer nowadays
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def get_action(self, state):
        """
        Get action probabilities based on current state; Choose an action randomly based on probabilities
        """
        state = np.reshape(state, [1, self.n_observation]) # Ensure shape matches what the model is expecting
        probs = self.model.predict(state, verbose = 0).flatten()

        return np.random.choice(self.n_action, p=probs)

    def update(self, states, actions, rewards):
      """
      Update policy (i.e. DQN) based on a completed episode (Monte Carlo approach)
      """
    
      ## Use GradientTape for automatic gradient computation
      with tf.GradientTape() as tape:
          G = 0
          discounted_rewards = []

          ## Get discounted total reward: G <- sum(gamma * rewards)
          for reward in rewards[::-1]:  ## Loop over all rewards in a single episode (in a reverse order) => `Loop for each step of the episode 0, 1, ..., T-1`
              G = reward + self.gamma * G
              discounted_rewards.insert(0, G)
          discounted_rewards = np.array(discounted_rewards)

          ## Reward normalization for stability (Refer to http://karpathy.github.io/2016/05/31/rl/)
          discounted_rewards -= np.mean(discounted_rewards)
          discounted_rewards /= (np.std(discounted_rewards) + 1e-8)
          
          ## Calculate log probabilities using visited states, and taken actions `ln(At | St, theta)`
          ## Calculating these probabilities outside of the block does not work because GradientTape is 
          ## not able to follow and watch them correctly
          logits = self.model(states)
          indices = tf.range(0, tf.shape(logits)[0]) * tf.shape(logits)[1] + actions
          act_probs = tf.gather(tf.reshape(logits, [-1]), indices)
          log_probs = tf.math.log(act_probs)

          ## We are calculating the loss using the log probabilities and the discounted rewards, per the pseudocode
          ## GradientTape computes the gradients of the log probabilities in line #77
          ## Since the optimizer is designed to minimize the loss, we have to take the negative in order to maximize it
          loss = -tf.reduce_sum(log_probs * discounted_rewards)

      ## The resources held by a GradientTape are released as soon as GradientTape.gradient() method is called
      gradients = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
      
      ## Return training loss for plotting purposes
      return loss.numpy()

    def generate_episode(self):
        """
        Generate a MC episode following current policy
        """
        states = []
        actions = []
        rewards = []
        state = self.env.reset()[0]
        done = False
        while not done:
            ## Get action based on policy
            action = self.get_action(state)
            ## Move in that direction and observe next state, reward, and whether we are done
            next_state, reward, done, _, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        return np.vstack(states), np.array(actions), np.array(rewards)