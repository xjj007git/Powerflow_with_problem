import warnings

import custom_envs
import numpy as np
import time
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

from sac_continuous_action import Args
from sac_continuous_action import Actor
from sac_continuous_action import SoftQNetwork1
from sac_continuous_action import SoftQNetwork2

from gymnasium.vector import SyncVectorEnv

import predictive_model
from collections import namedtuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
import matlab.engine
warnings.filterwarnings("ignore", category=FutureWarning)

LOG_STD_MAX = 2
LOG_STD_MIN = -5



if __name__ == "__main__":
    #先把m文件路径找到，并且运行一下case30，读取mpc
    eng = matlab.engine.start_matlab()
    path = eng.eval("which('case30.m')")
    # print(path)  
    mpc = eng.case30()
    eng.quit()

    def make_env():
        env = custom_envs.PowerSystemEnv(mpc, path)
        env = TimeLimit(env, max_episode_steps=288)
        env = RecordEpisodeStatistics(env)
        return env

    args = tyro.cli(Args)
    Mg = int(0.7 * args.batch_size)
    Mr = args.batch_size - Mg

    writer = SummaryWriter("runs/30bus_10")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = SyncVectorEnv([make_env])

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork1(envs).to(device)
    qf2 = SoftQNetwork2(envs).to(device)
    qf1_target = SoftQNetwork1(envs).to(device)
    qf2_target = SoftQNetwork2(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())#state_dict()返回一个包含整个模块状态的字典对象，通常包含参数和持久缓冲区（如running averages）的名称映射到相应的张量。
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)#定义优化器，优化器的参数是qf1和qf2的参数列表，学习率为args.q_lr，调用q_optimiezer.step()会更新qf1和qf2的参数。
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
    #只提供容器的作用
    rb_virtual_box = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    Batch = namedtuple('ReplayBufferSamples', ['observations', 'next_observations', 'actions', 'rewards', 'dones'])

    predictive_model = predictive_model.PredictiveModel(envs.envs[0], device=device, rb_virtual_box=rb_virtual_box)

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

        #这里写rb_virtual,在前期随机探索后每一回合进行虚拟数据产生,但是只产生200个
        if global_step % 288 == 0 and global_step > args.learning_starts:
            #更新虚拟数据的标准差
            #内部已经修改内部全部变量sigma_global,这里可以提供简单一点的打印方式
           sigma = predictive_model.update_deviation_via_sample(rb, sample_size=5000)
           predictive_model.generate_virtual_data(rb, actor=actor)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if global_step > args.learning_starts:
            if 'dcv' in infos:
                writer.add_scalar("charts/reward-step", rewards[0], global_step)
                writer.add_scalar("charts/dcv-step", infos["dcv"][0], global_step)
                # print("每步dcv")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}, dcv={info['dcv']}, cost={info['cost']}")
                    # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
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
        if global_step > args.learning_starts:
            # data = rb.sample(args.batch_size)

            #虚拟经验池和真实经验池混合采样
            data_real = rb.sample(Mr)
            try:
                data_virtual = predictive_model.rb_virtual.sample(Mg)
                # print("虚拟经验池采样成功")
            except ValueError:
                # 虚拟缓冲区为空或样本不足，用真实数据替代
                data_virtual = rb.sample(Mg)

            #虚拟经验采样和真实经验采样,对于rb而言，只能每一个tenrsor用cat拼接
            cat_obs = torch.cat([data_real.observations, data_virtual.observations])
            cat_acts = torch.cat([data_real.actions, data_virtual.actions])
            cat_next_obs = torch.cat([data_real.next_observations, data_virtual.next_observations])
            cat_dones = torch.cat([data_real.dones, data_virtual.dones])
            cat_rews = torch.cat([data_real.rewards, data_virtual.rewards])
            data = Batch(observations=cat_obs, actions=cat_acts, next_observations=cat_next_obs, dones=cat_dones, rewards=cat_rews) #按照sample的顺序拼接

            # print(f"data的维度: obs={data.observations.shape}, actions={data.actions.shape}, next_obs={data.next_observations.shape}, rewards={data.rewards.shape}, dones={data.dones.shape}")

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)#这里应该是t+1时刻的Q估计值，和target没啥关系
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)#计算t+1时刻的Q值和t时刻Q值，然后计算半均方差损失函数
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()#清零
            qf_loss.backward()#计算梯度
            q_optimizer.step()#更新参数

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
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
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)#。item()方法将单元素张量转换为Python数值
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