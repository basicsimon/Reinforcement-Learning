import gymnasium as gym
import numpy as np
import random
import time 
import matplotlib.pyplot as plt 

env = gym.make('Blackjack-v1', render_mode='human')

# Q-learning setting
alpha = 0.1        #learning rate
gamma = 0.9        # discount factor
epsilon = 1.0      # Epsilon /Greedy policy
epsilon_decay = 0.99
min_epsilon = 0.01

#  Q-table
Q_table = np.zeros((32, 11, 2, 2))  # states: actor: Hit or Stand

# Q-learning choice actor
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  
    else:
        return np.argmax(Q_table[state[0], state[1], int(state[2])])

# update Q-table
def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(Q_table[next_state[0], next_state[1], int(next_state[2])])
    Q_table[state[0], state[1], int(state[2]), action] += alpha * (
        reward + gamma * Q_table[next_state[0], next_state[1], int(next_state[2]), best_next_action] -
        Q_table[state[0], state[1], int(state[2]), action]
)

# Training the agent
rewards = []
episodes = 100  
animation_speed = 0.3

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    while True:
        env.render()
        time.sleep(animation_speed)
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    rewards.append(total_reward)
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if episode % 10 == 0:
        print(f"Episode {episode}, Epsilon: {epsilon:.2f}, Average Reward: {np.mean(rewards[-10:]):.2f}")
env.close() 

plt.plot(np.convolve(rewards, np.ones(10)/10, mode='valid'))
plt.title("Training Progress (Average Reward over 10 episodes)")
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.grid(True)
plt.show()

np.save("Q_table.npy", Q_table)
print("Training completed! Q-table saved.")