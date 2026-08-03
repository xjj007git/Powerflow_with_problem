# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
# %%
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import opfgym
from opfgym.envs.eco_dispatch import EcoDispatch
from opfgym.envs.voltage_control import VoltageControl
import pandapower as pp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
# %%
# print(__file__)
# print(type(__file__))

# %%
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "VoltageControl-v0"  #Hopper-v4
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e3                                  #一开始设置值是5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:#if (capture_video)和(idx==0)
            env = gym.make(env_id, render_mode="rgb_array", disable_env_checker=True)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, disable_env_checker=True)
        # env = DCVRewardWrapper(env, cmax=cmax)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



def calculate_generation_cost(net):
    """
    根据 net.poly_cost 计算总发电成本 (欧元/小时)
    支持二次多项式: cost = c2 * P^2 + c1 * P + c0
    这里只计算有功费用,应该用动态res表格
    """
    total_cost = 0.0
    if 'poly_cost' not in net or net.poly_cost.empty:
        return 10.0 * (np.maximum(0, net.res_gen['p_mw']).sum()+np.maximum(0, net.res_sgen['p_mw']).sum())  #maximum是比较数组的最大值函数
    cost_table = net.poly_cost[net.poly_cost['et'].isin(['gen', 'sgen','ext_grid'])].copy() #选出所有行，独立复制

    if cost_table.empty:
        return 10.0 * (np.maximum(0, net.res_gen['p_mw']).sum()+np.maximum(0, net.res_sgen['p_mw']).sum())

    for _, row in cost_table.iterrows():
        et = row['et']
        idx = row['element']
        c0 = row['cp0_eur']
        c1 = row['cp1_eur_per_mw']
        c2 = row['cp2_eur_per_mw2']

        if et == 'sgen':
            if idx < len(net.res_sgen):
                p = max(0, net.res_sgen.p_mw.iloc[idx])
                total_cost += c2 * p**2 + c1 * p + c0
        elif et == 'ext_grid':
            if idx < len(net.res_ext_grid):
                p = max(0, net.res_ext_grid.p_mw.iloc[idx])
                total_cost += c2 * p**2 + c1 * p + c0
        elif et == 'gen':
            if idx < len(net.res_gen):
                p = max(0, net.res_gen.p_mw.iloc[idx])
                total_cost += c2 * p**2 + c1 * p + c0
    return total_cost

def calculate_cmax(net):
    total_cost_max = 0.0    #用的静态表格
    # if 'poly_cost' not in net or net.poly_cost.empty:
    #     return 10.0 * (net.gen['max_p_mw'].sum()+net.sgen['max_p_mw'].sum())
    cost_table = net.poly_cost[net.poly_cost['et'].isin(['gen', 'sgen','ext_grid'])].copy() #选出所有行，独立复制

    # if cost_table.empty:
    #     return 10.0 * (net.gen['max_p_mw'].sum()+net.sgen['max_p_mw'].sum())

    for _, row in cost_table.iterrows():
        et = row['et']
        idx = row['element']
        c0 = row['cp0_eur']
        c1 = row['cp1_eur_per_mw']
        c2 = row['cp2_eur_per_mw2']

        if et == 'sgen':
            if 'max_p_mw' in net.sgen.columns:
                p_max = net.sgen['max_p_mw'].iloc[idx] if idx < len(net.sgen) else 0.0
            else:
                p_max = net.sgen['p_mw'].iloc[idx] if idx < len(net.sgen) else 0.0
                print(f"Warning: 'max_p_mw' not found for sgen at index {idx}, using current p_mw as max")
            total_cost_max += c2 * p_max**2 + c1 * p_max + c0

        elif et == 'ext_grid':
            if 'max_max_p_mw' in net.ext_grid.columns:
                p_max = net.ext_grid['max_max_p_mw'].iloc[idx] if idx < len(net.ext_grid) else 1000.0  
            else:
                p_max = 1000.0
                print(f"Warning: 'max_max_p_mw' not found for ext_grid at index {idx}, using default 1000 MW as max")
            total_cost_max += c2 * p_max**2 + c1 * p_max + c0

        elif et == 'gen':
            if 'max_p_mw' in net.gen.columns:
                p_max = net.gen['max_p_mw'].iloc[idx] if idx < len(net.gen) else 0.0
            else:
                p_max = net.gen['p_mw'].iloc[idx] if idx < len(net.gen) else 0.0
                print(f"Warning: 'max_p_mw' not found for gen at index {idx}, using current p_mw as max")
            total_cost_max += c2 * p_max**2 + c1 * p_max + c0
    return total_cost_max

def calculate_dcv(net, epsilon_v=1e-5, epsilon_q=1e-5, epsilon_s=1e-5):
    """
    根据论文公式 (6) 计算 DCV
    约束类型：
      1. 电压幅值 (pu)        : 上下限 0.9 ~ 1.1 pu (IEEE 标准常见值)
      2. 发电机无功出力 (MVar): 来自 gen['min_q_mvar'] / gen['max_q_mvar']
      3. 线路/变压器视在功率  : 使用 loading_percent (%)，上限 100%
    epsilon_v=1.0, epsilon_q=1.0, epsilon_s=1.0是权重,应该用res动态表格
    """
    # ----- 1. 电压约束 -----
    v_pu = net.res_bus.vm_pu
    v_min = 0.9
    v_max = 1.1
    v_range = v_max - v_min  # 0.2
    v_viol = np.maximum(0, v_pu - v_max) + np.maximum(0, v_min - v_pu)
    v_viol_norm = v_viol / v_range

    # ----- 2. 发电机无功约束 -----
    q_sgen_mvar = net.res_sgen.q_mvar.values if len(net.res_sgen) > 0 else []
    sgen_q_min = net.sgen['min_q_mvar'].values if 'min_q_mvar' in net.sgen.columns else np.zeros_like(q_sgen_mvar)
    sgen_q_max = net.sgen['max_q_mvar'].values if 'max_q_mvar' in net.sgen.columns else np.zeros_like(q_sgen_mvar)
    q_range_sgen = sgen_q_max - sgen_q_min
    
    with np.errstate(divide='ignore', invalid='ignore'):
        q_viol_sgen = np.maximum(0, q_sgen_mvar - sgen_q_max) + np.maximum(0, sgen_q_min - q_sgen_mvar)
        q_viol_norm_sgen = np.divide(q_viol_sgen, q_range_sgen, out=np.zeros_like(q_viol_sgen), where=q_range_sgen != 0)
    q_viol_norm_sgen = np.nan_to_num(q_viol_norm_sgen, nan=0.0)

    # ----- 3. 线路/变压器容量约束 -----
    loading = net.res_line.loading_percent  # 已计算好的负载率 (%)
    s_max = 100.0   # % (额定容量对应 100%)
    s_viol = np.maximum(0, loading - s_max)
    s_viol_norm = s_viol / s_max

    # ----- 加权合并 -----
    all_viol = np.hstack([v_viol_norm * epsilon_v,
                          q_viol_norm_sgen * epsilon_q,
                          s_viol_norm * epsilon_s])
    dcv = np.sqrt(np.mean(all_viol**2))
    return dcv

def compute_reward(cost, dcv, cmax, epsilon0=1e-4, epsilon1=0.1, R0=100.0, R1=20.0):
    """
    论文公式 (7)
    epsilon0=1e-4 是 DCV 的容忍度,epsilon1=0.1 是严重违规的阈值
    R0=1e-4 是基本奖励,R1=20.0 是成本奖励的权重
    """
    if dcv > epsilon1:
        # 严重不可行，给予零奖励（或一个很大的负惩罚，论文中是0）
        reward = 0.0
    else:
        cost_term = R1 * (cost / cmax)
        penalty = min(R0, dcv / epsilon0)
        reward = R0 + R1 - cost_term - penalty
    return reward

class DCVRewardWrapper(gym.Wrapper):
    def __init__(self, env, cmax=None):
        super().__init__(env)
        self.cmax = cmax
        self.epsilon0 = 1e-4
        self.epsilon1 = 1.0     #这里是epsilon的源头
        self.R0 = 100.0
        self.R1 = 20.0
        self._net = None

    def _get_net(self):
        """获取环境内部的 pandapower 网络"""
        # VoltageControl-v0 内部属性通常是 'net'
        #attr:attribute属性,hasattr(obj,name)检查obj中是否有name属性
        #opfgym 可能在 env.unwrapped 中存储网络对象，也就是说opfgym包装了net
        #raise是关键字，if,for等都是，AttributeError是异常类型
        try:
            # 使用官方推荐的 get_wrapper_attr 方法
            return self.env.get_wrapper_attr('net')
        except AttributeError:
            raise AttributeError("无法从环境中获取 'net' 属性，请检查环境是否正确初始化")

    def reset(self, **kwargs):                  #kwargs:keyword arguments关键字参数，允许传入任意数量的关键字参数，函数内部以字典形式访问
        obs, infos = self.env.reset(**kwargs)    #关键字参数就是参数，因为传参时有类似关键字的不可修改性
        self._net = self._get_net()
        if self.cmax is None:
            self.cmax = calculate_cmax(self._net)
        return obs, infos

    def step(self, action):
        obs, original_reward, terminated, truncated, infos = self.env.step(action)
        self._net = self._get_net()         # 更新网络引用（潮流已计算）
        cost = calculate_generation_cost(self._net)
        dcv = calculate_dcv(self._net)
        new_reward = compute_reward(cost, dcv, self.cmax,
                                    epsilon0=self.epsilon0,
                                    epsilon1=self.epsilon1,
                                    R0=self.R0,
                                    R1=self.R1)
        # 将中间量存入 infos 以便记录
        # dcv = 0.000001                                   #来测试是否包装成功
        infos['original_reward'] = original_reward       #infos是一个字典，由step返回
        infos['dcv'] = dcv
        infos['cost'] = cost
        return obs, new_reward, terminated, truncated, infos



if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")



    # #创建一个临时环境，计算 C_max
    # print("Precomputing C_max using a temporary environment...")
    # tmp_env = gym.make(args.env_id, disable_env_checker=True)
    # #临时 reset 以获得网络
    # tmp_obs, _ = tmp_env.reset()
    # # 获取网络对象
    # #net公开属性，_net保护属性，__net私有属性，在同一环境中会同时存在，优先使用公开属性

    # try:
    #     # 使用官方推荐的 get_wrapper_attr 方法
    #     tmp_net = tmp_env.get_wrapper_attr('net')
    # except AttributeError:
    #     raise AttributeError("无法从环境中获取 'net' 属性，请检查环境是否正确初始化")

    # pp.runpp(tmp_net)  # 确保网络参数完整，计算 C_max
    # cmax_val = calculate_cmax(tmp_net)
    # tmp_env.close()
    # print(f"Computed C_max = {cmax_val:.2f} $/h")



    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    #前面单开一个环境和这里的环境计算Cmax不一致
    print("Computing C_max using a temporary environment...")
    envs.envs[0].reset()  # 重置第一个环境以确保网络对象可用
    net = envs.envs[0].get_wrapper_attr('net')
    pp.runpp(net)
    cmax_val = calculate_cmax(net)
    print(f"Computed C_max = {cmax_val:.2f} $/h")
    envs.envs[0] = DCVRewardWrapper(envs.envs[0], cmax=cmax_val)
    envs.envs[0] = gym.wrappers.RecordEpisodeStatistics(envs.envs[0])



    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}, dcv={info['dcv']}, cost={info['cost']}, original_reward={info['original_reward']}") #可以添加DCV和cost的打印
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:#更新价值critic网络
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support     #更新策略actor网络
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
