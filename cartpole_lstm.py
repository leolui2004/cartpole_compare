import tensorflow as tf
tf.keras.backend.set_floatx('float32')

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
episode_limit = 30000
discount = 0.97
model_lr = 0.0001
score_avg_freq = 700
score_list = []

class ActorCriticModel:
    def __init__(self, ActionNumber):
        self.ActionNumber = ActionNumber
        self.model = self.lstm_model()
        self.opt = tf.keras.optimizers.Adam(model_lr)
    
    def lstm_model(self):
        input = tf.keras.layers.Input((2, 2))
        layer1 = tf.keras.layers.LSTM(32, return_sequences=True)(input)
        layer2 = tf.keras.layers.Dropout(0.2)(layer1)
        layer3 = tf.keras.layers.LSTM(64, return_sequences=True)(layer2)
        layer4 = tf.keras.layers.Dropout(0.2)(layer3)
        layer5 = tf.keras.layers.LSTM(128)(layer4)
        layer6 = tf.keras.layers.Dropout(0.2)(layer5)
        layer7 = tf.keras.layers.Dense(64, activation='linear')(layer6)
        layer8 = tf.keras.layers.Dense(32, activation='linear')(layer7)
        layer9 = tf.keras.layers.Dense(16, activation='linear')(layer8)
        logits = tf.keras.layers.Dense(self.ActionNumber)(layer9)
        value = tf.keras.layers.Dense(1)(layer9)
        return tf.keras.Model(inputs=[input], outputs=[logits, value])
    
    def predict(self, input):
        logits, _ = self.model.predict(input)
        return logits
    
    def compute_loss(self, done, state_, memory):
        if done:
            reward_sum = 0.
        else:
            reward_sum = self.model(tf.convert_to_tensor(state_, dtype=tf.float32))[-1][0]
        
        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            reward_sum = reward + discount * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        
        logits, values = self.model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))
        advantage = discounted_rewards - values
        value_loss = advantage ** 2
        
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
        
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions[0], logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss
    
    def train(self, done, state_, memory):
        state_ = tf.convert_to_tensor(state_, dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = self.compute_loss(done, state_, memory)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def action_choose(self, state):
        logits = self.model.predict(state)
        probs = np.exp(logits[0][0])/sum(np.exp(logits[0][0]))
        action = np.random.choice(self.ActionNumber, p=probs)
        return action

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

def encode(a):
    b = [[0,0],[0,0]]
    b[0] = a[0:2]
    b[1] = a[2:]
    return b

ACM = ActorCriticModel(2)
memory = Memory()

for episode in range(episode_limit):
    state_list, state_next_list, action_list = [], [], []
    score, score_memory, timestep = 0, 0, 0
    memory.clear()
    done = False
    observation = env.reset()
    state = encode(observation)
    
    while not done:
        timestep += 1
        action = ACM.action_choose(np.array(state)[np.newaxis, :])
        observation_next, reward, done, info = env.step(action)
        
        state_next = encode(observation_next)
        state_list.append(state)
        state_next_list.append(state_next)
        action_list.append(action)
        
        score += reward
        score_memory += reward
        state = state_next
        
        if done or timestep == 10:
            memory.store(np.array(state_list), np.array(action_list), score_memory)
            if score_memory > 8:
                ACM.train(done, np.array(np.array(state_next)[None, :]), memory)
            state_list, state_next_list, action_list = [], [], []
            score_memory, timestep = 0, 0
            memory.clear()
            if done:
                score_list.append(score)
                print('Episode {} Score {}'.format(episode + 1, score))

env.close()

score_avg_list = []
for i in range(1, episode_limit + 1):
    if i < score_avg_freq:
        score_avg_list.append(np.mean(score_list[:]))
    else:
        score_avg_list.append(np.mean(score_list[i - score_avg_freq:i]))
plt.plot(score_avg_list)
plt.show()