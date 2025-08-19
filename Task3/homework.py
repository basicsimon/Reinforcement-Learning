import gymnasium as gym
import numpy as np
import pickle

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8")

num_episodes = 50000
learning_rate = 0.1
discount_rate = 0.95
epsilon = 1.0
epsilon_decay = 0.8

q_table = np.ones((env.observation_space.n,env.action_space.n))

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_rate * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state

    epsilon = max(0.01, epsilon * epsilon_decay) 

with open('q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)

print("Training completed! Q-table saved.")

with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

test_env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", render_mode='human')

num_test_episodes = 10
success_times = 0

for episode in range(num_test_episodes):
    state, _ = test_env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        state = next_state

        if terminated and reward == 1:
            success_times += 1

print(f"Successful episodes in 8x8 map: {success_times / num_test_episodes:.2f}")

env.close()
test_env.close()

