import itertools
import os
import time
import argparse
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import sys
import random
from collections import deque, namedtuple

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box
from gym import wrappers

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1))  # reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32') / 255.
        img = np.squeeze(img)
        return img

def make_env():
    env_spec = gym.spec('ppaquette/DoomHealthGathering-v0')
    env_spec.id = 'DoomHealthGathering-v0'
    env = env_spec.make()
    e = PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(env)),
                                 width=80, height=80, grayscale=True)
    return e
env = make_env()
#env = wrappers.Monitor(env, './experiment', force=True)

NOOP, FORWARD, TURN_R, TURN_L  = 0, 1, 2, 3
VALID_ACTIONS = [0, 1, 2, 3]


class Estimator():
    def __init__(self, scope="estimator"):
        self.scope = scope
        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):
        self.X_pl = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name="X")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = self.X_pl
        batch_size = tf.shape(self.X_pl)[0]

        conv1 = tf.contrib.layers.conv2d(
            X, 32, 5, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv1")
        pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        conv2 = tf.contrib.layers.conv2d(
            pool1, 32, 3, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv2")
        pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        conv3 = tf.contrib.layers.conv2d(
            pool2, 64, 2, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv3")
        pool3 = tf.nn.max_pool(conv3, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

	conv4 = tf.contrib.layers.conv2d(
            pool3, 64, 1, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv4")
        pool4 = tf.nn.max_pool(conv4, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        conv5 = tf.contrib.layers.conv2d(
            pool4, 128, 1, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv5")
        pool5 = tf.nn.max_pool(conv5, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        flattened = tf.contrib.layers.flatten(pool5)
        fc1 = tf.contrib.layers.fully_connected(flattened, 128)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS), activation_fn=None)

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, sess, s):
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        global_step, _, loss = sess.run(
            [tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    saver,
                    num_episodes,
                    replay_memory_size=10000,
                    replay_memory_init_size=1000,
                    update_target_estimator_every=500,
                    epsilon_decay_steps=60000,
                    discount_factor=0.99,
                    epsilon_start=0.9,
                    epsilon_end=0.001,
                    batch_size=32):

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = []

    checkpoint = tf.train.get_checkpoint_state("docs")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
	print("Could not find old network weights")
    replay_memory = []

    total_t = sess.run(tf.contrib.framework.get_global_step())
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))
    total_t=0
    for i_episode in range(num_episodes):

        state = env.reset()
        state = np.stack([state] * 4, axis=2)
        loss = None
        total_reward = 0
        actions_tracker = [0, 1, 2, 3]

        for t in itertools.count():

            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")
                saver.save(sess, 'docs/dqn-doom-basic', global_step=total_t)

            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            actions_tracker.append(action)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
	    #env.render()
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            replay_memory.append(Transition(state, action, reward, next_state, done))   
            total_reward += reward

            if total_t > replay_memory_init_size:
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                q_values_next = q_estimator.predict(sess, next_states_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = target_estimator.predict(sess, next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                    discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

                states_batch = np.array(states_batch)
                loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            state = next_state
            total_t += 1

            if done:
                print("Step {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss))
                print('reward %f, steps %d, eps %f' % (total_reward, t, epsilon))
                counts_action = np.bincount(actions_tracker) - 1
                print('noop %d, forward %d, turn_right %d, turn_left %d' % \
                    (counts_action[0], counts_action[1], counts_action[2], counts_action[3]))
                break

tf.reset_default_graph()

global_step = tf.Variable(0, name='global_step', trainable=False)
    
q_estimator = Estimator(scope="q")
target_estimator = Estimator(scope="target_q")

saver = tf.train.Saver()

if not os.path.exists('docs'):
    os.makedirs('docs')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    deep_q_learning(sess,
                    env,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    saver=saver,
		    num_episodes=500)
