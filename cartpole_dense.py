import gym
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
episode_limit = 2000
timestep_limit = 2000
input_dim = 4
dense1_dim = 128
dense2_dim = 32
action_dim = 2
discount = 0.97
actor_lr = 0.0001
critic_lr = 0.0001
score_avg_freq = 50

action_space = [i for i in range(action_dim)] # [0,1]
score_list = [] # store scores for all episodes

input = Input(shape=(input_dim,)) # 4 dimensions from observation
delta = Input(shape=[1])
dense1 = Dense(dense1_dim, activation='relu')(input)
dense2 = Dense(dense2_dim, activation='relu')(dense1)
prob = Dense(action_dim, activation='softmax')(dense2)
value = Dense(1, activation='linear')(dense2)

def custom_loss(y_true, y_pred):
    out = K.clip(y_pred, 1e-8, 1-1e-8)
    log_lik = y_true * K.log(out)
    return K.sum(-log_lik * delta)

actor = Model(inputs=[input, delta], outputs=[prob])
actor.compile(optimizer=Adam(lr=actor_lr), loss=custom_loss)
critic = Model(inputs=[input], outputs=[value])
critic.compile(optimizer=Adam(lr=critic_lr), loss='mean_squared_error')
policy = Model(inputs=[input], outputs=[prob])

def action_choose(observation):
    state = observation[np.newaxis, :]
    probabiliy = policy.predict(state)[0]
    action = np.random.choice(action_space, p=probabiliy)
    return action, state

def learn(state, action, reward, state_next, done):
    state = state
    state_next = state_next
    critic_value = critic.predict(state)
    critic_value_next = critic.predict(state_next)
    
    target = reward + discount * critic_value_next * (1-int(done))
    delta =  target - critic_value
    actions = np.zeros([1, action_dim])
    actions[np.arange(1), action] = 1
    
    actor.fit([state, delta], actions, verbose=0)
    critic.fit(state, target, verbose=0)

for episode in range(episode_limit):
    score = 0
    observation = env.reset()
    
    for timestep in range(timestep_limit):
        #env.render()
        #action = env.action_space.sample() # random action
        action, state = action_choose(observation) # predicted action
        observation, reward, done, info = env.step(action)
        
        state_next = observation[np.newaxis, :] # reshape
        
        score += reward
        
        if done or timestep == timestep_limit - 1:
            score_list.append(score)
            print('Episode {} Score {}'.format(episode + 1, score))
            break

env.close()

score_avg_list = []
for i in range(1, episode_limit + 1):
    if i < score_avg_freq:
        score_avg_list.append(np.mean(score_list[:]))
    else:
        score_avg_list.append(np.mean(score_list[i - score_avg_freq:i]))
plt.plot(score_avg_list)
plt.show()