import numpy as np
import torch
from pandapower.converter.matpower import from_mpc
import my_HELMpy
import pandapower as pp
import custom_envs

class PredictiveModel:
    def __init__(self, env, device, rb_virtual_box, Nr=200):
        self.env = env
        self.device = device
        self.rb_virtual = rb_virtual_box
        # self.y_dim = 40
        self.y_dim = env.unwrapped.grid['num_loads'] + env.unwrapped.grid['num_loads'] + env.unwrapped.grid['num_wind']
        self.Nr = Nr
        self.sigma_global = None
        self.Nh = 1
        self.Cmax = env.unwrapped.Cmax

        self.grid = env.unwrapped.grid
        self.num_non_slack = self.grid['num_non_slack']
        self.num_loads = self.grid['num_loads']
        self.num_wind = self.grid['num_wind']
        self.num_gen = self.grid['num_gen']
        self.base_pd = self.grid['base_pd']
        self.base_qd = self.grid['base_qd']
        self.load_buses = self.grid['load_buses']
        self.gen_buses = self.grid['gen_buses']
        self.slack_idx = self.grid['slack_idx']
        self.non_slack_indices = self.grid['non_slack_indices']
        self.path = self.grid['path']

        self.y_low = np.concatenate([
            np.zeros(self.num_loads),               # p_d
            np.zeros(self.num_loads),               # q_d
            -50 * np.ones(self.num_wind)            # p_w（风电最小0，但这里设-50保守）
        ])
        self.y_high = np.concatenate([
            1.2 * self.base_pd,
            1.2 * self.base_qd,
            50 * np.ones(self.num_wind)
        ])

        #构造HELM求解器
        self.net = from_mpc(self.path, f_hz=50)
        pp.runpp(self.net)
        self.net_HE = my_HELMpy.quick_HELM(self.net)   #此处应该更改为快速求解器

        self.slack_p = self.net.res_ext_grid['p_mw'].values[0] if len(self.net.res_ext_grid) > 0 else 0.0
        self.slack_q = self.net.res_ext_grid['q_mvar'].values[0] if len(self.net.res_ext_grid) > 0 else 0.0

    def update_deviation_via_sample(self, rb, sample_size=5000):
        #打算直接从rb海量数据中抽样取5000个然后计算样本标准差,因为我觉得要是全计算后面要计算十多万数据太多了，会很耗时间
        # sample
        batch = rb.sample(sample_size)
        
        # 取出 obs 和 next_obs
        obs = batch.observations     #刚刚取出来是tensor类型
        next_obs = batch.next_observations
        
        # 转为 numpy
        if hasattr(obs, 'cpu'):
            obs = obs.cpu().numpy()
            next_obs = next_obs.cpu().numpy()
        
        # 提取随机变量y
        y_now = obs[:, :self.y_dim]
        y_next = next_obs[:, :self.y_dim]
        diff = y_next - y_now
        
        # 计算标准差，加小常数防止除零
        sigma = np.std(diff, axis=0) + 1e-6  #沿着列方向求标准差
        self.sigma_global = sigma
        return sigma

    def generate_virtual_data(self, rb, actor): #不含更新标准差的部分
        if self.sigma_global is None:
            raise RuntimeError("请先调用 update_noise_model 计算噪声标准差！")
        
        #从rb中采样200个
        batch = rb.sample(self.Nr)
        obs_tensor = batch.observations
        if hasattr(obs_tensor, 'cpu'):
            obs = obs_tensor.cpu().numpy()

        #用当前actor获取动作
        actions, _, _ = actor.get_action(torch.Tensor(obs_tensor).to(self.device))  #actions的shape会跟随obs
        actions = actions.detach().cpu().numpy()

        #预测一下状态的y
        y_now = obs[:, :self.y_dim]
        y_next = y_now + self.sigma_global * np.random.normal(0, 1, size = y_now.shape)
        #截断防止越界
        y_next = np.clip(y_next, self.y_low, self.y_high)

        #加上pu-就是s_next了,但是从actions中取出来的不包含松弛节点，我们需要获取松弛节点的pu,这个pu不参与计算，不是很重要,随便赋值了
        p_u_non_slack = actions[:, :self.num_non_slack]
        slack_p = np.full((self.Nr, 1), self.slack_p)
        p_u = np.concatenate([p_u_non_slack, slack_p], axis=1)
        next_obs = np.concatenate([y_next, p_u], axis=1).astype(np.float64)

        #计算reward
        rewards, dones = self.compute_virtual_reward(obs, actions, next_obs)

        # print(f"虚拟数据生成: obs.shape={obs.shape}, actions.shape={actions.shape}, next_obs.shape={next_obs.shape}")

        for i in range(self.Nr):
            self.rb_virtual.add(obs[i], next_obs[i], actions[i], rewards[i], dones[i], {}) #按照add的顺序添加

    def compute_virtual_reward(self, obs, actions, next_obs):
        """
        输入：
            obs: 当前状态 (Nr, obs_dim)
            actions: 当前策略生成的动作 (Nr, act_dim)  
            next_obs: 预测的下一状态 (Nr, obs_dim)
        输出：
            rewards: (Nr,) 
            dones: (Nr,) 
        """
        Nr = obs.shape[0]
        rewards = np.zeros(Nr)
        dones = np.zeros(Nr, dtype=bool)

        state = {
            'loads': np.column_stack([self.net.res_load['p_mw'].values, self.net.res_load['q_mvar'].values]),
            'winds': np.zeros((self.num_wind, 2)),  # 无风电场则为空
            'gen': np.column_stack([self.net.res_gen['p_mw'].values, self.net.res_gen['q_mvar'].values]), #不包括松弛节点
            'slack': np.array([self.slack_p, self.slack_q]), 
            'bus_voltage': np.column_stack([self.net.res_bus['vm_pu'].values, self.net.res_bus['va_degree'].values])
        }

        for i in range(Nr):         #i从0到199
            action = actions[i]  
            
            # 解析动作
            p_set = action[:self.num_non_slack]
            v_set = action[self.num_non_slack: self.num_non_slack+self.num_gen]
            load_shed = action[self.num_non_slack+self.num_gen: self.num_non_slack+self.num_gen+self.num_loads]
            wind_curt = action[-self.num_wind:]
            #将actions变为control,从array变成了dict
            control = {
                'p_set': p_set,      # 非平衡机有功目标
                'v_set': v_set,          # 发电机电压目标
                'load_shed': load_shed,       # 负荷削减比例
                'wind_curt': wind_curt        # 弃风比例
            }

            #解析状态,下面要计算HE了,计算奖励算next_obs,所以用next_obs传入HE
            next_obs_single = next_obs[i]

            pd = next_obs_single[:self.num_loads]
            qd = next_obs_single[self.num_loads: self.num_loads+self.num_loads]
            pw = next_obs_single[self.num_loads*2: self.num_loads*2+self.num_wind]
            pu = next_obs_single[-self.num_gen:]

            #先把每一个load,wind节点的Pd,Pw给计算好,为之后计算发电机净出力铺垫,因为HELM计算出来的Bus净出力
            actual_load_p = pd * (np.ones(self.num_loads) - control['load_shed'])
            actual_load_q = qd * (np.ones(self.num_loads) - control['load_shed'])
            
            #Vsp
            Vsp = np.ones(self.net_HE.nb)
            Vsp[np.concatenate([self.net_HE.ref,self.net_HE.net.gen.bus])] = control['v_set']

            S_total = np.zeros(self.net_HE.nb, dtype=np.complex128)
            #P_pv
            S_total[self.net_HE.net.gen.bus] = control['p_set']

            #S_pq,先忽略风电
            S_total[self.net_HE.net.load.bus] = S_total[self.net_HE.net.load.bus] - actual_load_p - 1j * actual_load_q

            S_total = S_total/self.net_HE.net._ppc["baseMVA"]
            
            #提取数据
            P_pv = np.real(S_total[self.net_HE.pv])
            S_pq = S_total[self.net_HE.pq]

            #HELM潮流计算
            V_HE,S_HE,terms = self.net_HE.run_DHE(S_pq, P_pv, Vsp)

            S_HE_mw = S_HE * self.net_HE.net._ppc["baseMVA"]
            load_on_gen_buses = np.intersect1d(self.load_buses, self.gen_buses)
            P_load_on_gen = np.zeros(self.num_gen)
            Q_load_on_gen = np.zeros(self.num_gen)
            for bus in load_on_gen_buses:
                gen_idx = np.where(self.gen_buses == bus)[0][0]
                load_idx = np.where(self.load_buses == bus)[0][0]
                P_load_on_gen[gen_idx] = actual_load_p[load_idx]
                Q_load_on_gen[gen_idx] = actual_load_q[load_idx]
            S_gen = S_HE_mw[self.gen_buses] + (P_load_on_gen + 1j * Q_load_on_gen)
            gen_p_mw = np.real(S_gen)
            gen_q_mva = np.imag(S_gen)

            next_state = state.copy()

            next_state['loads'][:, 0] = next_obs_single[:self.num_loads]
            next_state['loads'][:, 1] = next_obs_single[self.num_loads: self.num_loads+self.num_loads]
            
            next_state['winds'][:, 0] = next_obs_single[self.num_loads+self.num_loads:self.num_loads+self.num_loads+self.num_wind]
            
            #gen是由S_HE_mw传出的S_bus计算过来的S_gen得出
            next_state['gen'][:, 0] = np.real(S_gen[self.non_slack_indices])
            next_state['gen'][:, 1] = np.imag(S_gen[self.non_slack_indices])

            slack_p = np.real(S_gen[self.slack_idx])
            slack_q = np.imag(S_gen[self.slack_idx])
            next_state['slack'] = np.array([slack_p, slack_q])
            
            V_HE_magnitude = np.abs(V_HE)
            V_HE_angle = np.angle(V_HE) 
            next_state['bus_voltage'][:, 0] = V_HE_magnitude
            next_state['bus_voltage'][:, 1] = V_HE_angle

            C = self.env.unwrapped.calculate_generation_cost(next_state)
            DCV = self.env.unwrapped.calculate_dcv(next_state)
            reward = self.env.unwrapped.compute_reward(C, DCV, self.Cmax, epsilon0=1e-4, epsilon1=0.1, R0=100.0, R1=20.0)
            
            if reward == 0:
                done = False
            else:
                done = True

            rewards[i] = reward
            dones[i] = done

        return rewards, dones