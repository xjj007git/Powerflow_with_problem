import gymnasium as gym
import opfgym
from opfgym.envs.eco_dispatch import EcoDispatch
from opfgym.envs.voltage_control import VoltageControl

if 'EcoDispatch-v0' not in gym.envs.registry:
    gym.envs.registration.register(
        id='EcoDispatch-v0',  # 为环境指定一个唯一的 ID
        entry_point='opfgym.envs.eco_dispatch:EcoDispatch',  # 告诉 gym 如何找到这个环境类
    )
if 'VoltageControl-v0' not in gym.envs.registry:
    gym.envs.registration.register(
        id='VoltageControl-v0',
        entry_point='opfgym.envs.voltage_control:VoltageControl',
    )

env = gym.make("EcoDispatch-v0")
obs, info = env.reset()
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)