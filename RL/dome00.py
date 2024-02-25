import gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

for _ in range(10):
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(env.step(action))

    if terminated or truncated:
        observation, info = env.reset()
env.close()