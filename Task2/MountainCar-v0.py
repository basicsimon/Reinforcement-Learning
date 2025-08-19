import gymnasium as gym

env = gym.make("MountainCar-v0",  render_mode="rgb_array", goal_velocity=0.1)
state = env.reset() 

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)


for step in range(5): # Short loops that perform random actions
    action = env.action_space.sample()  # random actor
    next_state, reward, done, truncated, _ = env.step(action) 
    print(f"Step {step}: State={next_state}, Reward={reward}")
    if done or truncated:
        env.reset() 

env.close()
