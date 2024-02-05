#Importing Dependencies for Mario
import tensorflow as tf
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from IPython. display import clear_output
from keras.models import save_model
from keras.models import load_model
import time

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace (env, RIGHT_ONLY)

total_reward = 0
done = True

class DQNAgent:
    def __init__(self, state_size, action_size):
        #Create variables for our agent
        self.state_space = state_size
        self.action_space = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.8
        #Gamma is discount rate
        #Exploration vs explotation
        #higher the epsilon rate the more random movements will be
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay_epsilon = 0.0001
        self.chosenAction = 0
        
        #Building Neural Networks for Agent and main and target networks
        self.main_network = self.build_network()
        self.target_network = self.build_network ()
        self.update_target_network() # This sets the weight of the main network to the target network
         
    def build_network(self):
        model = Sequential ()
        model.add(Conv2D (64,(4,4),strides=4, padding='same', input_shape=self.state_space))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4,4), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3,3), strides=1, padding='same'))
        model.add(Activation('relu')) 
        model. add(Flatten ())
        model.add (Dense (512, activation='relu'))
        model. add (Dense (256, activation='relu'))
        model.add (Dense (self. action_space, activation='linear'))
        model. compile (loss= 'mse', optimizer=Adam())
        return model
    
    #Updating the target network from the main networks
    def update_target_network (self):
        self.target_network.set_weights(self.main_network.get_weights())
        
    #Defining the action what the agent should take based on state.
    def act(self, state, onGround):
        if onGround < 83:
            print("On Ground")
            if random.uniform(0,1) < self.epsilon:
                self.chosenAction = np.random.randint (self.action_space)
                return self.chosenAction
            Q_value = self.main_network.predict(state)
            self.chosenAction = np.argmax(Q_value[0])
               # print(Q_value)
            return self.chosenAction
        else:
            print("Not on Ground")
            return self.chosenAction
    #Changing epsilon with the training as after training the model gets more accurate so trying out random actions 
    #more frequently will not benefit us 
    def update_epsilon(self, episode):
        self.epsilon = self.min_epsilon+(self.max_epsilon -self.min_epsilon)*np.exp(-self.decay_epsilon*episode)
        
    #train the network def train(self, batch_size):
    def train(self, batch_size):
        #minibatch from memory
        minibatch = random.sample (self.memory, batch_size)
        
        #Get variables from the batch so we can find q-value
        for state, action, reward, next_state, done in minibatch:
            target = self.main_network.predict(state)
            if done:
                target[0][action]=reward
            else:
                target[0][action] = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))
            
            self.main_network.fit (state, target, epochs=1, verbose=0)
            
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def load(self, name):
        self.main_network = load_model(name)
        self.target_network = load_model(name)
        
    def save(self, name):
        save_model(self.main_network, name)
        
#To get the number of actions possible 
action_space = env.action_space.n
#To get the state
state_space = (80, 88, 1)
from PIL import Image
#Here we are preprocessing the image into grey scale and in the size of 88 and 80 to make it computationally inexpensive 
def preprocess_state(state):
    image = Image.fromarray (state)
    image = image.resize((88, 80))
    print(image)
    image = image.convert('L')
    image = np. array (image)
    #image.show(image)
    return image


# env.observation_space
# dan = DONAgent (state_size)
num_episodes = 1000000
num_timesteps = 400000
batch_size = 64  
DEBUG_LENGTH = 300
dqn = DQNAgent (state_space, action_space)
        

print("STARTING TRAINING")

stuck_buffer = deque(maxlen=DEBUG_LENGTH)

for i in range (num_episodes):
    Return = 0
    done = False
    time_step = 0
    onGround = 79
    
    state = preprocess_state(env.reset())
    state = state.reshape(-1, 80, 88, 1)
    
    for t in range (num_timesteps) :
        env.render ()
        time_step += 1
        
        if t>1 and stuck_buffer.count(stuck_buffer[-1])>DEBUG_LENGTH-50:
            action = dqn.act(state, onGround=79)
        else:
            action = dqn.act (state, onGround)
        
        print("ACTION IS"+str(action))
        
        next_state, reward, done, info =env.step(action)
        
        #print(info['y_pos'])
        onGround = info['y_pos']
        stuck_buffer.append(info['x_pos'])
        
        next_state = preprocess_state(next_state)
        next_state = next_state.reshape (-1, 80, 88, 1)
        
        dqn.store_transition(state, action, reward, next_state, done)
        state = next_state
          
        Return += reward
        print("Episode is: {}\nTotal Time Step: {}\nCurrent Reward: {}\nInEpsilon is: {}".format(str(i), str(time_step), str(Return), str(dqn.epsilon)))
        clear_output (wait=True)
        
        if done:
            break
        
        if len (dqn.memory) > batch_size and i>5:
            dqn.train(batch_size) 
            
    dqn. update_epsilon (i)
    clear_output (wait=True)
    dqn.update_target_network()
    print("======== saving model ===========")
    dqn.save('marioRl.h5')
    print("======== saving model ==============")

env.close()

dqn.save('marioRl.h5')
dqn.load('marioRl.h5')
