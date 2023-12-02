import tensorflow as tf
import numpy as np
import random
from collections import deque

class Actor(tf.keras.Model):
    def __init__(self, input_shape, action_size):
        super(Actor, self).__init__()

        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        # Although the paper suggests 400 and 300 neurons, I was running into convergence issues with those number so I had to reduce them
        self.hidden1 = tf.keras.layers.Dense(128, activation="relu")
        self.hidden2 = tf.keras.layers.Dense(128, activation="relu")
        self.out_layer = tf.keras.layers.Dense(action_size, activation="tanh")

    def call(self, state):
        # Reshape state if needed, required to load the model during testing phase
        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)
        
        # Propagate state through model layers to predict next action
        x = self.input_layer(state)
        x = self.hidden1(x)
        x = self.hidden2(x)
        action = self.out_layer(x)

        return action

class Critic(tf.keras.Model):
    def __init__(self, input_shape):
        super(Critic, self).__init__()

        # Initialize layers; we use the input layer specifically for the state, and the action layer for the action
        # before concatenating them together to predict the value
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.hidden1 = tf.keras.layers.Dense(128, activation="relu")
        self.action_layer = tf.keras.layers.Dense(128)
        self.concat = tf.keras.layers.Concatenate()
        self.hidden2 = tf.keras.layers.Dense(128, activation="relu")
        # The final output is only one neuron since we need to predict single q value
        self.out_layer = tf.keras.layers.Dense(1)

    def call(self, state, action):
        s1 = self.input_layer(state)
        s1 = self.hidden1(s1)
        a1 = self.action_layer(action)
        x = self.concat([s1, a1])
        x = self.hidden2(x)
        q = self.out_layer(x)

        return q

class OrnsteinUhlenbeckActionNoise:
    # References: https://planetmath.org/ornsteinuhlenbeckprocess; https://en.wikipedia.org/wiki/Wiener_process
    # Theta value comes from the DDPG paper
    def __init__(self, mean, std_deviation, theta=0.15, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.x_initial = x_initial
        self.reset()

    # Reset the noise to its initial state
    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) *  2
            + self.std_dev * np.sqrt(1e-2) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x


class ReplayBuffer:
    # Initialize the replay buffer with a specified capacity
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)

        if self.buffer.maxlen > self.size():
            self.buffer.append(experience)
        else:
            # If the buffer is full, we need to remove entries from the beginning so we can replace them with new experiences
            self.buffer.popleft()
            self.buffer.append(experience)

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size):
        # Ensure we don't try to sample more experiences than possible
        sample_size = min(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, sample_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)