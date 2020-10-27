import os.path
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfc

OBSERVATIONS_SIZE = 6400

# Aim is to increase the log probability of winning actions , and to decrease the log probability of losing actions.
# Positive reward pushes the log probability of chosen action up; negative reward pushes the log probability of the chosen action down.

class Network:
    def __init__(self, hidden_layer_size, learning_rate, checkpoints_dir):
        self.learning_rate = learning_rate
        # set TensorFlow Session
        self.sess = tfc.InteractiveSession()
        # set observations
        self.observations = tfc.placeholder(tfc.float32, [None, OBSERVATIONS_SIZE])
        # set actions: +1 for up, -1 for down
        self.sampled_actions = tfc.placeholder(tfc.float32, [None, 1])
        self.advantage = tfc.placeholder(tfc.float32, [None, 1], name='advantage')
        # hidden layers features
        hidden_layers = tfc.layers.dense(self.observations, units=hidden_layer_size, activation=tfc.nn.relu, kernel_initializer=tfc.contrib.layers.xavier_initializer())
        self.up_probability = tfc.layers.dense( hidden_layers, units=1, activation=tfc.sigmoid, kernel_initializer=tfc.contrib.layers.xavier_initializer())
        self.loss = tfc.losses.log_loss(labels=self.sampled_actions, predictions=self.up_probability, weights=self.advantage)
        # set optimizer of network
        optimizer = tfc.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        tfc.global_variables_initializer().run()
        self.saver = tfc.train.Saver()
        # checkpoints
        self.checkpoint_file = os.path.join(checkpoints_dir,'policy_network.ckpt')

    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, observations):
        up_probability = self.sess.run(self.up_probability, feed_dict={self.observations: observations.reshape([1, -1])})
        return up_probability

    def train(self, state_action_reward_tuples):
        states, actions, rewards = zip(*state_action_reward_tuples)
        # stack together  states, actions, and rewards for this episode
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {self.observations: states, self.sampled_actions: actions, self.advantage: rewards}
        self.sess.run(self.train_op, feed_dict)