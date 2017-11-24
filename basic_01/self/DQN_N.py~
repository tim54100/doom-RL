
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import os
import sys
import random
from collections import deque, namedtuple

np.random.seed(1)
tf.set_random_seed(1)


# In[5]:

class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        st_shape,
        learning_rate=0.01,
        reward_decay=0.9,
        epsilon_start=0.9,
        replace_target_iter=3000,
        memory_size=20000,
        batch_size=32,
        epsilon_decrease=True,
        output_graph=False,
    ):
        self.n_actions=n_actions
        self.width=int(st_shape[0])
        self.height=int(st_shape[1])
        self.grayscale=True if st_shape[2]=='1' else False
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_decrease =  epsilon_decrease
        self.epsilon = epsilon_start if epsilon_decrease else 0
	self.epsilon_start = epsilon_start 
	self.epsilon_end = 0.0001
	self.explore = 300000
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter

        # total_learning_step
        self.learn_step_counter = 0

        # initialize zero memory [state, action, reward, next_state, done]
        self.replay_memory = []
        self.Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

        self.q_estimator = Estimator(scope="q")
        self.target_estimator = Estimator(scope="target_q")
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        checkpoint = tf.train.get_checkpoint_state("saved_n")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        if not os.path.exists('saved_n'):
            os.makedirs('saved_n')
        if output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
    
    
    def make_policy(self, state, estimator):
        A =np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        q_values = estimator.predict(self.sess, np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)
	#print(best_action)
	if best_action>=2:
	    best_action=2
        A[best_action] += (1.0 - self.epsilon)
	#print(self.epsilon)
        if self.epsilon_decrease:
            self.epsilon -= ((self.epsilon_start-self.epsilon_end)/self.explore)
        return A
    def store_transition(self, state, action, reward, done, n_state):
        if len(self.replay_memory) == self.memory_size:
            self.replay_memory.pop(0)
        self.replay_memory.append(self.Transition(state, action, reward, n_state, done))
    def learn(self):
	#print("wtf")
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_estimator.replace_parms(self.sess, self.q_estimator)
            self.saver.save(self.sess, 'saved_n/saved', global_step=self.learn_step_counter)
        samples = random.sample(self.replay_memory, self.batch_size)
        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
        
        q_values = self.q_estimator.predict(self.sess, next_states_batch)
        best_actions = np.argmax(q_values, axis=1)
        q_target = self.target_estimator.predict(self.sess, next_states_batch)
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.gamma * q_target[np.arange(self.batch_size), best_actions]
        states_batch = np.array(states_batch)
        loss = self.q_estimator.update(self.sess, np.array(states_batch), np.array(action_batch), np.array(targets_batch))
        self.cost_his.append(loss)
        
        self.learn_step_counter += 1
        


# In[2]:

class Estimator():
    def __init__(self, scope="estimator",width=80,height=80):
        self.width=width
        self.height=height
        self.scope = scope
        with tf.variable_scope(scope, reuse = None):
            
            # c_names(collections_names) are the collections to store variables
            self.c_names= [self.scope+'net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.w_initializer=tf.random_normal_initializer(0.,0.3)
            self.b_initializer=tf.constant_initializer(0.1) #config if layers
            self._build_model()
    def _build_model(self):
        self.x_pl = tf.placeholder(tf.float32, [None, self.width,self.height ,4], name=self.scope+'x')
        self.q_target_pl = tf.placeholder(tf.float32, [None], name=self.scope+"Q_target")
        self.actions_pl = tf.placeholder(tf.int32, [None], name=self.scope+"actions")
        
        X=self.x_pl
        batch_size=tf.shape(self.x_pl)[0]
        

        conv1 = tf.contrib.layers.conv2d(
            X, 32, 5, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv1",
            variables_collections=self.c_names, weights_initializer=self.w_initializer,
            biases_initializer=self.b_initializer, reuse=None)
        pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

        conv2 = tf.contrib.layers.conv2d(
            pool1, 32, 3, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv2",
            variables_collections=self.c_names, weights_initializer=self.w_initializer,
            biases_initializer=self.b_initializer, reuse=None)
        pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

        conv3 = tf.contrib.layers.conv2d(
            pool2, 64, 2, 1, activation_fn=tf.nn.relu, padding="VALID", scope="conv3",
            variables_collections=self.c_names, weights_initializer=self.w_initializer,
            biases_initializer=self.b_initializer, reuse=None)
        pool3 = tf.nn.max_pool(conv3, [1, 3, 3, 1], [1, 2, 2, 1], padding="VALID")

        flattened = tf.contrib.layers.flatten(pool3)
        fc1 = tf.contrib.layers.fully_connected(flattened , 64, variables_collections=self.c_names)
        self.predictions = tf.contrib.layers.fully_connected(fc1, 4, activation_fn=None)

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        
        self.losses = tf.squared_difference(self.q_target_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)
        
        
        self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
        
    def predict(self, sess, state):
        return sess.run(self.predictions, { self.x_pl: state})
    def update(self, sess, state, action, q_target):
        feed_dict = {self.x_pl: state, self.q_target_pl: q_target, self.actions_pl: action }
        _, loss = sess.run(
            [self.train_op, self.loss],
            feed_dict)
        return loss
    def get_parms(self):
        return tf.get_collection(self.scope+'net_parms')
    def replace_parms(self, sess, estimator):
        t_parms = tf.get_collection(self.scope+'net_parms')
        e_parms = estimator.get_parms()
        sess.run([tf.assign(t, e) for t, e in zip(t_parms, e_parms)])
            


# In[ ]:




# In[ ]:




# In[95]:




# In[98]:




# In[ ]:



