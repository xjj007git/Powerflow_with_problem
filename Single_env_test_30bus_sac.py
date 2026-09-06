from custom_envs import PowerSystemEnv
from custom_envs import CustomTimeLimit
from custom_envs import CustomRecordEpisodeStatistics
from custom_envs import add_para_to_tensorboard
from custom_envs import add_chart_to_tensorboard
from custom_envs import check_conservation_of_energy
import matlab.engine
import numpy as np
import time
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from cleanrl_utils.buffers import ReplayBuffer
from Single_sac import Args
from Single_sac import Actor
from Single_sac import SoftQNetwork1
from Single_sac import SoftQNetwork2
from gymnasium.vector import SyncVectorEnv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.add_dll_directory(r"D:\Users\xjj\anaconda3\envs\hesac\Library\bin")
import math
import pandas as pd

LOG_STD_MAX = 2
LOG_STD_MIN = -5



if __name__ == "__main__":
    #先把m文件路径找到，并且运行一下case30，读取mpc
    eng = matlab.engine.start_matlab()
    path = eng.eval("which('case30.m')")
    mpc = eng.case30()
    eng.quit()

    def make_env():
        env = PowerSystemEnv(mpc, path)
        env = CustomTimeLimit(env, max_episode_steps=288)
        env = CustomRecordEpisodeStatistics(env, deque_size=100)
        return env

    args = tyro.cli(Args)
    #先固定训练步数
    # args.total_timesteps = 50000

    writer = SummaryWriter("runs/30bus_1")
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
    env = make_env()

    max_action = env.action_space.high

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork1(env).to(device)
    qf2 = SoftQNetwork2(env).to(device)
    qf1_target = SoftQNetwork1(env).to(device)
    qf2_target = SoftQNetwork2(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())#state_dict()返回一个包含整个模块状态的字典对象，通常包含参数和持久缓冲区（如running averages）的名称映射到相应的张量。
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)#定义优化器，优化器的参数是qf1和qf2的参数列表，学习率为args.q_lr，调用q_optimiezer.step()会更新qf1和qf2的参数。
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    gross_load_pq_list = []  # 用于记录每步的gross_load_pq
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = np.array(env.action_space.sample())
            # print(f"global_step={global_step}, random action={action}")   #采集一个随机动作，方便我固定动作做测试
        else:
            action, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            action = action.detach().cpu().numpy()

        #固定动作测试环境单元
        # action = np.array([2.0991030e+01,1.9663343e+01,4.6175299e+00,2.1679873e+00,3.9365467e+01,
        #             9.5006186e-01,1.0572938e+00,1.0163674e+00,9.8251331e-01,1.0506465e+00,
        #             1.0710058e+00,3.2419793e-02,6.0476184e-01,9.7409123e-01,8.1701368e-02,
        #             6.2478316e-01,2.9376015e-01,1.2806474e-01,4.4767186e-01,2.0577361e-01,
        #             3.9837193e-01,8.7941164e-01,1.4867796e-01,8.0984271e-01,8.3373344e-01,
        #             4.5655915e-01,8.0827528e-01,2.2652337e-01,6.3903052e-01,4.3548229e-01,6.6719705e-01])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)

        #检查打印一下发电机出力
        # served_gen_p = action[:5]
        # print(f"global_step={global_step}, served_gen_p={served_gen_p}")        

        #给tensorboard添加一些变量
        # add_para_to_tensorboard(writer, global_step, info)
        add_chart_to_tensorboard(writer, global_step, info)

        #检查能量守恒
        # check_conservation_of_energy(info, writer, global_step)

        #这里是测试gross_load_pq是否会持续削减的
        # gross_load_pq_list.append(gross_load_pq)
        # if global_step == args.total_timesteps - 1:
        #     #存为excel文件
        #     cols = len(gross_load_pq)
        #     rows = args.total_timesteps
        #     data_2d = pd.DataFrame(gross_load_pq_list).values.reshape(rows, cols)
        #     df = pd.DataFrame(data_2d)
        #     df.to_excel("./data/gross_load_pq_with_shed.xlsx", index=False, header=False)

        if global_step > args.learning_starts:
            if 'dcv' in info:
                writer.add_scalar("charts/reward-step", reward, global_step)
                writer.add_scalar("charts/dcv-step", info["dcv"], global_step)
                # print("每步dcv")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in info:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}, dcv={info['dcv']}, cost={info['cost']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # print(truncation)
        if truncation:
            real_next_obs = info['final_observation']

        rb.add(obs, real_next_obs, action, reward, termination, info)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
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

    env.close()
    writer.close()