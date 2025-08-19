import gymnasium as gym

def create_env():
    """
    Create a FrozenLake environment with a fixed configuration.
    Returns:
    env: The initialized environment
    """
    custom_map = [  # I find this code in gymnasium web
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
    return gym.make('FrozenLake-v1', desc=custom_map, is_slippery=True, render_mode='human')


def run_episode(env, steps=5):
    """
    Run a single experiment in the environment and perform a random action with a specified number of steps.
    Args:
    env: The Gym environment
    steps (int) -Maximum number of steps to run
    """

    state, info = env.reset()
    print(f"Initial State: {state}, Info: {info}")


    for step in range(steps):
        action = env.action_space.sample()  # Pick an action at random
        new_state, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}: Action={action}, NewState={new_state}, Reward={reward}, Terminated={terminated}, Truncated={truncated}")

        if terminated or truncated:           # Checking if it's over
            print(f"Episode ended at step {step + 1}")
            break
    print("Episode finished.\n")


def main():  #build env
    env = create_env()

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    num_episodes = 10 # 10 times run
    for i in range(num_episodes):
        print(f"Episode {i + 1}/{num_episodes}")
        run_episode(env, steps=20)

    env.close()


if __name__ == '__main__':
    main()








