
# coding: utf-8

# In[38]:

import tensorflow as tf
import numpy as np
from DQN_N import DeepQNetwork
import sys
sys.path.append("/home/eastgeno/workspace/gym")
sys.path.append("/home/eastgeno/anaconda2/lib/python2.7/site-packages")
import cv2
import gym
import ppaquette_gym_doom


# In[39]:

if __name__ == "__main__":
    #creat a env,the example is Breakout-v0
    
    env = gym.make('ppaquette/DoomBasic-v0')
    RL = DeepQNetwork(4,
            str(env.observation_space)[4:-1].split(','),
            learning_rate=0.01,
            reward_decay=0.9,
            epsilon_start=0.9,
            replace_target_iter=2000,
            memory_size=20000,
            )
    step = 0
    VALID_ACTIONS = [0, 1, 2, 3]
    for episode in range(3000):
        state = env.reset()
        
        state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        state = np.reshape(state, (80, 80))
        state = np.stack((state, state, state, state), axis=2)
        #print(state.shape)
        
        action_tracker=np.zeros(4)
        total_reward=0
        done = False

        while not done:
            #env.render()
            action_probs = RL.make_policy(state, RL.q_estimator)
            action = np.random.choice(np.arange(len(action_probs)))
	    #print(action_probs)
            action_tracker[action]+=1

	    #action_one_hot = np.zeros(4)
	    #action_one_hot[action] = 1
           # print("my action len:%d" % len(action_one_hot))
	   # print(env.allowed_actions)
            n_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            n_state = cv2.cvtColor(cv2.resize(n_state, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, n_state = cv2.threshold(n_state, 1, 255, cv2.THRESH_BINARY)
            n_state = np.reshape(n_state, (80, 80, 1))
            n_state = np.append(n_state, state[:, :, :3], axis=2)
            total_reward+=reward
            RL.store_transition(state, action, reward, done, n_state)
            
            if (step > 2000):
                RL.learn()
                
            state=n_state
            
            step+=1
        print('episode: %d/3000, total_reward: %d, epsilon: %f' % (episode, total_reward, RL.epsilon))
        print('action'+'action'.join(str(i)+': '+str(action_tracker[i])[:-2]+'  ' for i in range(len(action_tracker))))
    RL.plot_cost()


# In[27]:



# In[ ]:




# In[ ]:



