import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt


DISPLAY_REWARD_THRESHOLD = -2000
RENDER = True

# env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0') #取消限制
env.seed(1)  #普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子
env = env.unwrapped

print(env.action_space) #Discrete(2)
print(env.observation_space) #Box(4,)
print(env.observation_space.high)  #[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low)   #[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

#创建网络
RL = PolicyGradient(
    n_actions = env.action_space.n,  #2
    n_features = env.observation_space.shape[0], #4
    learning_rate = 0.02,
    reward_decay = 0.995
)

for i_episode in range(1000):
    observation = env.reset() #初始化环境，每次循环都初始化

    while True:
        if RENDER:
            env.render()

        action = RL.choose_action(observation)#actor网络根据当前环境产生一个动作

        observation_,reward,done,info = env.step(action)
        if done:
            reward=-1.0

        RL.store_transition(observation,action,reward) #存储单步游戏数据

        if done:  #存储一整场游戏，时间连续，没done的话就采集数据，done掉的话就学习
            ep_rs_sum = sum(RL.ep_rs)  #一场游戏的奖励和
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum  #第一场游戏running_reward不存在
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD:  #-2000
                RENDER = True
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if False:#i_episode == 30:
                plt.plot(vt)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
                pass

            break

        observation = observation_