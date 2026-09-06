from __future__ import annotations
import matlab.engine
import gymnasium as gym
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
import pandapower.networks as pn
import pandapower as pp
from scipy.sparse import csr_matrix
import scipy
import time
from pandapower.converter.matpower import from_mpc
import HELMpy
import my_HELMpy
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

from copy import deepcopy
from typing import TYPE_CHECKING
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class PowerSystemEnv(gym.Env):
    def __init__(self, mpc, path):
        super().__init__()

        #从mpc读取数据
        bus = mpc['bus']
        gen = mpc['gen']
        gencost = mpc['gencost']
        bus_np = np.array(bus)
        gen_np = np.array(gen)
        gencost_np = np.array(gencost)

        num_bus = len(bus_np)

        #找出负载节点load,依据仅仅是Pd>0
        load_buses = np.where(bus_np[:,2] > 0)[0]   # 索引 0-based,20个Load节点
        num_loads = len(load_buses)
        # 提取基准有功和无功（单位 MW/MVar）
        base_pd = bus_np[load_buses, 2]   # 列索引 2 (0-based) 是 Pd
        base_qd = bus_np[load_buses, 3]   # 列索引 3 是 Qd
        #提取所有节点的Vmax、Vmin
        V_max = bus_np[:, 11]
        V_min = bus_np[:, 12]

        # 发电机信息
        ngen = gen_np.shape[0]
        gen_buses = gen_np[:, 0].astype(int) - 1   # 转为 0-based 母线索引
        pg_max = gen_np[:, 8]    # Pmax (列索引 8)
        pg_min = gen_np[:, 9]   # Pmin (列索引 9)
        qg_max = gen_np[:, 3]
        qg_min = gen_np[:, 4]
        vg_default = gen_np[:, 5]   # 默认电压设定值 (列索引 5)

        # 找出平衡机（参考母线，bus type = 3）
        slack_bus = np.where(bus_np[:,1] == 3)[0][0] #此处不适用于双slack节点,这里主要是在Bus中母线号
        slack_gen_idx = np.where(gen_buses == slack_bus)[0][0]   # 平衡机在 gen 中的索引
        non_slack_indices = [i for i in range(ngen) if i != slack_gen_idx] #其他在gen中索引
        num_non_slack = len(non_slack_indices)

        #风电场
        wind_buses = np.array([])
        num_wind = 0

        grid_data = {
            'num_bus': num_bus,
            'num_loads': num_loads,
            'num_wind': num_wind,
            'num_gen': ngen,    #这里是6，包括松弛节点
            'num_non_slack': num_non_slack, #这里给一个非松弛节点数

            'V_max': V_max,
            'V_min': V_min,

            'base_pd': base_pd, #这是负载基准功率
            'base_qd': base_qd,

            'gen_Pmax': pg_max,
            'gen_Qmin': qg_min,
            'gen_Qmax': qg_max,

            'gen_buses': gen_buses, #所有电机包括松弛节点在Bus中索引
            'load_buses': load_buses, #所有load节点在Bus中索引

            'non_slack_indices': non_slack_indices, #非松弛节点在gen中索引
            'slack_idx': slack_gen_idx, #这是松弛节点在gen中索引

            'gen_cost': gencost_np,
            'path': path,

            #传入动作空间边界
            'act_pg_max': pg_max[non_slack_indices],
            'act_pg_min': pg_min[non_slack_indices],
            'act_v_max': V_max[gen_buses],
            'act_v_min': V_min[gen_buses]
        }

        self.grid = grid_data
        
        #构造HELM求解器
        net = from_mpc(self.grid['path'], f_hz=50)
        self.net_HE = HELMpy.HELM(net)
        baseMVA = self.net_HE.net._ppc["baseMVA"]
        # self.my_net_HE = my_HELMpy.HELM(net)

        # ---- 状态空间 ----
        self.obs_low = np.concatenate([
            0.5 * self.grid['base_pd'],                     # p_d 下限
            0.5 * self.grid['base_qd'],                     # q_d 下限  见论文第四部分
            0.6 * 50 * np.ones(self.grid['num_wind']),      # p_w 下限
            np.zeros(self.grid['num_gen'])                  # p_u_prev
        ])
        self.obs_high = np.concatenate([
            1.2 * self.grid['base_pd'],                     # p_d 上限
            1.2 * self.grid['base_qd'],                     # q_d 上限
            50 * np.ones(self.grid['num_wind']),            # p_w 上限
            self.grid['gen_Pmax']                           # p_u_prev 上限
        ])

        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.obs_dim = self.observation_space.shape[0]

        #把y边界定义放init
        self.y_low = np.concatenate([
            0.5 * self.grid['base_pd'],                     # p_d 下限
            0.5 * self.grid['base_qd'],                     # q_d 下限
            0.6 * 50 * np.ones(self.grid['num_wind']),      # p_w 下限
        ])
        self.y_high = np.concatenate([
            1.2 * self.grid['base_pd'],                     # p_d 上限
            1.2 * self.grid['base_qd'],                     # q_d 上限
            50 * np.ones(self.grid['num_wind'])            # p_w 上限假设50MWA
        ])

        # ---- 动作空间 ----
        # 非平衡机有功上限
        non_slack_Pmax = self.grid['gen_Pmax'][self.grid['non_slack_indices']]
        self.act_low = np.concatenate([
            self.grid['act_pg_min'],
            self.grid['act_v_min'],                         #这里就是标幺值
            np.zeros(self.grid['num_loads']),               # 负荷削减
            np.zeros(self.grid['num_wind'])                 # 弃风
        ])
        self.act_high = np.concatenate([
            self.grid['act_pg_max'],
            self.grid['act_v_max'],
            0.3 * np.ones(self.grid['num_loads']),          #负荷最大削减比降低到30%
            np.ones(self.grid['num_wind'])
        ])
        self.action_space = gym.spaces.Box(low=self.act_low, high=self.act_high, dtype=np.float32)
        self.act_dim = self.action_space.shape[0]

        #成本参数
        self.load_shed_cost = 15.0      #10
        self.wind_curt_cost = 500.0
        self.Cmax = self.calculate_cmax()

        #内部超参数
        self.epsilon0 = 8e-4    #5e-4   4e-4
        self.epsilon1 = 0.1
        self.R0 = 100.0
        self.R1 = 20

        # 内部状态：当前系统状态
        self.current_state = None
        self.terminated = False
        self.truncated = False

    def compute_reward(self, cost, dcv, cmax, epsilon0, epsilon1, R0, R1):
        #打印测试数值
        # print(f"epsilon0是:{epsilon0}")
        if dcv > epsilon1:
            # 严重不可行，给予零奖励（或一个很大的负惩罚，论文中是0）
            reward = 0.0
            # self.terminated = True
            # self.terminated = False
        else:
            cost_term = R1 * (cost / cmax)
            penalty = min(R0, dcv / epsilon0)
            reward = R0 + R1 - cost_term - penalty
            self.terminated = False
        return reward
    
    def calculate_dcv(self, state):
        """
        根据论文公式 (6) 计算 DCV
        约束类型：
        1. 所有电压幅值 (pu)        
        2. 发电机无功出力 (MVar)
        3. 线路/变压器视在功率  
        """
        violations = [] 
        epsilon = 0.3  # 0.3    

        # ----- 所有节点的电压约束 -----
        vm = state['bus_voltage'][:, 0]
        v_min = self.grid['V_min']
        v_max = self.grid['V_max']
        v_range = v_max - v_min  #也是一个30长度数组
        v_viol = np.maximum(0, vm - v_max) + np.maximum(0, v_min - vm)
        v_viol_norm = v_viol / v_range
        violations.append(epsilon * v_viol_norm)

        # 发电机无功约束（含 slack 和 gen）
        #提取state中的测量无功，这里讲slack和non_slack分开了
        q_non_slack = state['gen'][:, 1]
        q_slack = state['slack'][1]

        #先将non_slack的Q边界提取出来
        q_non_slack_min = self.grid['gen_Qmin'][self.grid['non_slack_indices']]
        q_non_slack_max = self.grid['gen_Qmax'][self.grid['non_slack_indices']] 

        #再将slack的Q边界提取出来
        q_slack_min = self.grid['gen_Qmin'][self.grid['slack_idx']]
        q_slack_max = self.grid['gen_Qmax'][self.grid['slack_idx']] 

        #计算range
        q_range_non_slack = q_non_slack_max - q_non_slack_min
        q_range_slack = q_slack_max - q_slack_min
        q_range = np.concatenate([q_range_non_slack, [q_range_slack]])

        #计算violence
        q_viol_non_slack = np.maximum(0, q_non_slack - q_non_slack_max) + np.maximum(0, q_non_slack_min - q_non_slack)
        q_viol_slack = np.maximum(0, q_slack - q_slack_max) + np.maximum(0, q_slack_min - q_slack)

        q_viol = np.concatenate([q_viol_non_slack, [q_viol_slack]])
        q_viol_norm = q_viol / q_range

        violations.append(epsilon * q_viol_norm)

        # 线路容量约束,state没有存储

        # 合并
        all_viol = np.concatenate(violations)
        dcv = np.sqrt(np.mean(all_viol**2))
        return dcv
    
    def test_calculate_dcv(self, state):
        """
        根据论文公式 (6) 计算 DCV
        约束类型：
        1. 所有电压幅值 (pu)        
        2. 发电机无功出力 (MVar)
        3. 线路/变压器视在功率  
        """
        violations = [] 
        epsilon = 0.3  # 0.3    

        # ----- 所有节点的电压约束 -----
        vm = state['bus_voltage'][:, 0]
        v_min = self.grid['V_min']
        v_max = self.grid['V_max']
        v_range = v_max - v_min  #也是一个30长度数组
        v_viol = np.maximum(0, vm - v_max) + np.maximum(0, v_min - vm)
        v_viol_norm = v_viol / v_range
        violations.append(epsilon * v_viol_norm)

        # 发电机无功约束（含 slack 和 gen）
        #提取state中的测量无功，这里讲slack和non_slack分开了
        q_non_slack = state['gen'][:, 1]
        q_slack = state['slack'][1]

        #先将non_slack的Q边界提取出来
        q_non_slack_min = self.grid['gen_Qmin'][self.grid['non_slack_indices']]
        q_non_slack_max = self.grid['gen_Qmax'][self.grid['non_slack_indices']] 

        #再将slack的Q边界提取出来
        q_slack_min = self.grid['gen_Qmin'][self.grid['slack_idx']]
        q_slack_max = self.grid['gen_Qmax'][self.grid['slack_idx']] 

        #计算range
        q_range_non_slack = q_non_slack_max - q_non_slack_min
        q_range_slack = q_slack_max - q_slack_min
        q_range = np.concatenate([q_range_non_slack, [q_range_slack]])

        #计算violence
        q_viol_non_slack = np.maximum(0, q_non_slack - q_non_slack_max) + np.maximum(0, q_non_slack_min - q_non_slack)
        q_viol_slack = np.maximum(0, q_slack - q_slack_max) + np.maximum(0, q_slack_min - q_slack)

        q_viol = np.concatenate([[q_viol_slack], q_viol_non_slack])
        q_viol_norm = q_viol / q_range

        violations.append(epsilon * q_viol_norm)

        # 线路容量约束,state没有存储

        #查出越限机组的电机索引
        over_limit_indices = np.where(q_viol_non_slack >0.01)[0]   #不包含松弛节点,这里修改判断标准0——>0.01,因为比较的时候不知道为什么会产生一些偏差
        over_limit_indices_include_slack = np.where(q_viol > 0.01)[0]
        in_boundary = np.where(q_viol_non_slack ==0)[0]  #这里直接在non_slack中选择,不考虑将ref改变成PQ节点

        #返回需要打印的数据
        info = {
            #六台电机边界
            'q_non_slack_min': q_non_slack_min,
            'q_non_slack_max': q_non_slack_max,
            'q_slack_min': q_slack_min,
            'q_slack_max': q_slack_max,

            #真实数据
            'q_non_slack': q_non_slack,
            'q_slack': q_slack,

            #这是越限机组的索引
            'over_limit_indices': over_limit_indices,   #不包含松弛节点
            'over_limit_indices_include_slack': over_limit_indices_include_slack,

            #这是v和q的违反度，已经归一化了
            'v_viol_norm': v_viol_norm,
            'q_viol_norm': q_viol_norm,

            #传出去给下一轮全新迭代action的索引
            'in_boundary': in_boundary
        }

        # 合并
        all_viol = np.concatenate(violations)
        dcv = np.sqrt(np.mean(all_viol**2))
        return dcv, info
    
    def calculate_cmax(self): 
        total_cost_max = 0.0   
        for i in range(self.grid['num_gen']):   # 遍历所有发电机（包括松弛节点）
            c0 = self.grid['gen_cost'][i, 6]   # 注意顺序：c0, c1, c2
            c1 = self.grid['gen_cost'][i, 5]
            c2 = self.grid['gen_cost'][i, 4]
            p_max = self.grid['gen_Pmax'][i]
            total_cost_max += c2 * p_max**2 + c1 * p_max + c0

        #还需要添加最大弃载成本
        total_cost_max += np.sum(self.load_shed_cost * self.grid['base_pd'])  # 假设最大弃载为基准负荷

        return total_cost_max
    
    def calculate_generation_cost(self, state, load_shed):
        total_cost = 0.0
        C_gen = 0.0
        C_load_shed = 0.0

        #发电机有功成本
        pg_non_slack = state['gen'][:, 0]
        pg_slack = state['slack'][0]

        for i in range(self.grid['num_non_slack']):
            idx = self.grid['non_slack_indices'][i]
            c0 = self.grid['gen_cost'][idx, 6]
            c1 = self.grid['gen_cost'][idx, 5]
            c2 = self.grid['gen_cost'][idx, 4]

            total_cost += c2 * pg_non_slack[i]**2 + c1 * pg_non_slack[i] + c0

        c0_s = self.grid['gen_cost'][self.grid['slack_idx'], 6]
        c1_s = self.grid['gen_cost'][self.grid['slack_idx'], 5]
        c2_s = self.grid['gen_cost'][self.grid['slack_idx'], 4]
        total_cost += c2_s * pg_slack**2 + c1_s * pg_slack + c0_s

        C_gen = total_cost

        #弃载成本
        gross_load_p = state['loads'][:, 0].copy()
        total_cost += np.sum(self.load_shed_cost * gross_load_p * load_shed)

        C_load_shed = np.sum(self.load_shed_cost * gross_load_p * load_shed)

        return total_cost, C_gen, C_load_shed


    def _compute_base_powerflow(self):
        #做一次潮流计算获取一个可能存在的初始状态，以供初始化
        net = from_mpc(self.grid['path'], f_hz=50)
        pp.runpp(net)

        slack_p = net.res_ext_grid['p_mw'].values[0] if len(net.res_ext_grid) > 0 else 0.0
        slack_q = net.res_ext_grid['q_mvar'].values[0] if len(net.res_ext_grid) > 0 else 0.0

        state = {
            'loads': np.column_stack([net.res_load['p_mw'].values, net.res_load['q_mvar'].values]),
            'winds': np.zeros((self.grid['num_wind'], 2)),  # 无风电场则为空
            'gen': np.column_stack([net.res_gen['p_mw'].values, net.res_gen['q_mvar'].values]), #不包括松弛节点
            'slack': np.array([slack_p, slack_q]), 
            'bus_voltage': np.column_stack([net.res_bus['vm_pu'].values, net.res_bus['va_degree'].values])
        }
        # print(f"第一次潮流的load情况:{state['loads']}")
        return state
    
    def _get_obs(self, state):
        p_d = state['loads'][:, 0]
        q_d = state['loads'][:, 1]
        if self.grid['num_wind'] > 0:
            p_w = state['winds'][:, 0]
        else:
            p_w = np.zeros(self.grid['num_wind'])
        p_u_prev_slack = np.array([state['slack'][0]])
        p_u_prev_non_slack = state['gen'][:, 0]
        # p_u_prev = np.concatenate([p_u_prev_non_slack, p_u_prev_slack]) #这里会按照顺序拼接
        p_u_prev = np.concatenate([p_u_prev_slack, p_u_prev_non_slack]) #感觉不影响计算，因为obs不直接参与计算

        obs = np.concatenate([p_d, q_d, p_w, p_u_prev]).astype(np.float32)
        # obs_low = self.obs_low
        # obs_high = self.obs_high
        # eps = 1e-8
        # obs_norm = 2 * (obs - obs_low) / (obs_high - obs_low + eps) - 1
        # obs_norm = np.clip(obs_norm, -1, 1) #防越界处理
        # print(f"obs:{obs}")
        # return obs_norm       #这里试图将obs映射到-1到1,不知道想干什么
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) #仅仅对seed进行了设置

        # control = {
        #     'p_gen_set': np.zeros(self.grid['num_non_slack']),
        #     'v_set': np.ones(self.grid['num_gen']),
        #     'load_shed': np.zeros(self.grid['num_loads']),
        #     'wind_curt': np.zeros(self.grid['num_wind'])
        # }
        # init_loads = np.column_stack([self.grid['base_pd'], self.grid['base_qd']])
        # init_winds = np.zeros((self.num_wind, 2))
        # temp_state = {
        #     'loads': init_loads,
        #     'winds': init_winds,
        #     'gen': np.zeros((self.grid['num_gen'], 2)),
        #     'bus_voltage': np.ones((self.grid['num_bus'], 2))
        # }

        state = self._compute_base_powerflow()

        self.current_state = state
        self.terminated = False
        self.truncated = False

        observation = self._get_obs(state)
        # print(f"reset的obs:{observation}")
        info = {"is_operable": True}
        return observation, info

    def step(self, action):
        """
        参数:
            action: numpy array, shape (34,), 包含:
                - 前5个: 非平衡发电机有功设定 (MW)
                - 接着6个: 所有发电机电压设定 (pu)
                - 接着20个: 负荷削减比例 [0,1]
                - 最后3个: 弃风比例 [0,1]
        返回:
            obs, reward, terminated, truncated, info
        """
        # 解析动作
        p_set = action[:self.grid['num_non_slack']]
        v_set = action[self.grid['num_non_slack']:self.grid['num_non_slack']+self.grid['num_gen']]
        load_shed = action[self.grid['num_non_slack']+self.grid['num_gen']:self.grid['num_non_slack']+self.grid['num_gen']+self.grid['num_loads']]
        wind_curt = action[-self.grid['num_wind']:]

        #将actions变为control,从array变成了dict
        control = {
            'p_set': p_set,      # 非平衡机有功目标
            'v_set': v_set,          # 发电机电压目标
            'load_shed': load_shed,       # 负荷削减比例
            'wind_curt': wind_curt        # 弃风比例
        }

        #先把每一个load,wind节点的Pd,Pw给计算好,为之后计算发电机净出力铺垫,因为HELM计算出来的Bus净出力
        #同时,obs中的Pd,Qd 应该是在负载削减前的, 否则会越来越小
        gross_load_p = self.current_state['loads'][:, 0].copy()
        gross_load_q = self.current_state['loads'][:, 1].copy()

        #此处记录返回的gross_load_pq,在step之外组合写入excel查看gross_load_pq不会衰减
        #为什么在step外组合?因为step每次操作excel的库与mat之类冲突,会搞炸环境,我选择保守操作
        gross_load_pq = np.concatenate([gross_load_p, gross_load_q])

        #零削减
        # served_load_p = gross_load_p.copy()
        # served_load_q = gross_load_q.copy()

        #非零削减
        served_load_p = self.current_state['loads'][:, 0] * (np.ones(self.grid['num_loads']) - control['load_shed'])
        served_load_q = self.current_state['loads'][:, 1] * (np.ones(self.grid['num_loads']) - control['load_shed'])
        served_load_pq = np.concatenate([served_load_p, served_load_q])

        #用来测试Q越界,用来存下当前状态的负载p,q,用来对HELM层反复计算
        self.re_HE_state = self.current_state.copy()
        self.re_HE_load_p = served_load_p
        self.re_HE_load_q = served_load_q

        #获取HELM求解器
        net_HE = self.net_HE

        #利用control和self.current_state修改V_sp
        Vsp = np.ones(net_HE.nb)    #Vsp就是选的幅值,下面的写法会让除了PV的其他节点的V一直固定在一开始的值
        # Vsp = self.current_state['bus_voltage'][:, 0] #这里提取了current_state所有节点电压，顺序是bus从1-30
        Vsp[np.concatenate([net_HE.ref,net_HE.net.gen.bus])] = control['v_set'] #这里认为control(就是action)的v_set是slack在前，non_slack在后

        #利用control和self.current_state修改P_pv,之后再提取P_pv，Pv节点就不包含松弛节点
        S_total = np.zeros(net_HE.nb, dtype=np.complex128)
        S_total[net_HE.net.gen.bus] = control['p_set'] #有名值，虽然current_state中有值，但是control优先

        #利用control和self.current_state修改S_pq
        # S_total[net_HE.net.load.bus] = S_total[net_HE.net.load.bus] - self.current_state['loads'][:, 0] * (np.ones(self.grid['num_loads']) - control['load_shed']) - 1j * self.current_state['loads'][:, 1] * (np.ones(self.grid['num_loads']) - control['load_shed'])
        S_total[net_HE.net.load.bus] = S_total[net_HE.net.load.bus] - served_load_p - 1j * served_load_q    #负载可以在gen上面对节点叠加,不影响
        if len(net_HE.net.sgen.bus) > 0:
            S_total[net_HE.net.sgen.bus] = S_total[net_HE.net.sgen.bus] + net_HE.net.sgen.p_mw
        S_total = S_total/net_HE.net._ppc["baseMVA"]

        #提取数据
        P_pv = np.real(S_total[net_HE.pv])
        S_pq = S_total[net_HE.pq]

        #HELM潮流计算
        V_HE,S_HE,terms = net_HE.run_DHE(S_pq, P_pv, Vsp)

        #提取p_u,但是这里提取的其实不是pu,而是bus节点净出力Pbus = Pg + Pw - Pd
        # gen_p_mw = np.zeros(self.grid['num_gen'])   #此处包含松弛节点
        # gen_p_mw = np.real(S_HE_mw[self.grid['gen_buses']])
        S_HE_mw = S_HE * net_HE.net._ppc["baseMVA"]
        #下面要找出哪些负荷在发电机母线上面
        load_on_gen_buses = np.intersect1d(self.grid['load_buses'], self.grid['gen_buses'])
        #先建立放需要传给Pbus的Pd容器，形状和Pbus一致，方便相加
        P_load_on_gen = np.zeros(self.grid['num_gen'])
        Q_load_on_gen = np.zeros(self.grid['num_gen'])
        #遍历load_on_gen_buses，将Pd,Qd传进去
        for bus in load_on_gen_buses:
            gen_idx = np.where(self.grid['gen_buses'] == bus)[0][0]
            load_idx = np.where(self.grid['load_buses'] == bus)[0][0]
            P_load_on_gen[gen_idx] = served_load_p[load_idx]
            Q_load_on_gen[gen_idx] = served_load_q[load_idx]
        S_gen = S_HE_mw[self.grid['gen_buses']] + (P_load_on_gen + 1j * Q_load_on_gen)
        gen_p_mw = np.real(S_gen)
        gen_q_mva = np.imag(S_gen)

        #计算成本应该在状态转移修改pd之前计算完毕
        # 计算成本C
        C, C_gen, C_load_shed = self.calculate_generation_cost(self.current_state, control['load_shed'])

        #状态转移函数
        #y的随机分布,这里pd,qd也应该改成负荷削减之后的
        p_d_curr = gross_load_p.copy()
        q_d_curr = gross_load_q.copy()
        if self.grid['num_wind'] > 0:
            p_w_curr = self.current_state['winds'][:, 0] * (np.ones(self.grid['num_wind']) - control['wind_curt'])
        else:
            p_w_curr = np.array([])
        y = np.concatenate([p_d_curr, q_d_curr, p_w_curr])
        sigma = 0.01 * abs(y) + 1e-6
        y_sample = np.random.normal(loc = y,scale = sigma,size = y.shape)
        # y_next = np.minimum(np.maximum(y_sample, self.y_low),self.y_high)
        y_next = np.clip(y_sample, self.y_low, self.y_high)
        next_obs = np.concatenate([y_next, gen_p_mw]).astype(np.float64) #包含了松弛节点

        #依据next_obs和HELM输出修改next_state
        next_state = self.current_state.copy()
        
        # #切片要小心，可能会影响gen的顺序
        next_state['loads'][:, 0] = next_obs[:self.grid['num_loads']]
        next_state['loads'][:, 1] = next_obs[self.grid['num_loads']:self.grid['num_loads']+self.grid['num_loads']]
        
        next_state['winds'][:, 0] = next_obs[self.grid['num_loads']+self.grid['num_loads']:self.grid['num_loads']+self.grid['num_loads']+self.grid['num_wind']]
        
        #gen是由S_HE_mw传出的S_bus计算过来的S_gen得出
        next_state['gen'][:, 0] = np.real(S_gen[self.grid['non_slack_indices']])
        next_state['gen'][:, 1] = np.imag(S_gen[self.grid['non_slack_indices']])

        slack_p = np.real(S_gen[self.grid['slack_idx']])
        slack_q = np.imag(S_gen[self.grid['slack_idx']])
        next_state['slack'] = np.array([slack_p, slack_q])
        
        V_HE_magnitude = np.abs(V_HE)
        V_HE_angle = np.angle(V_HE) 
        next_state['bus_voltage'][:, 0] = V_HE_magnitude
        next_state['bus_voltage'][:, 1] = V_HE_angle

        # 更新状态
        self.current_state = next_state

        # 计算DCV
        DCV = self.calculate_dcv(self.current_state)

        # 奖励和终止判断
        reward = self.compute_reward(C, DCV, self.Cmax, self.epsilon0, self.epsilon1, self.R0, self.R1)

        # 提取观测值，也就是观测的状态空间，对应论文S，这里要返回的其实是下一个状态
        obs = self._get_obs(self.current_state)

        terminated = self.terminated
        truncated = self.truncated   # 留给 TimeLimit 包装器

        # 信息字典
        info = {
            'cost': C, 
            'dcv': DCV,
            #添加参数
            'served_load_pq': served_load_pq,
            'gross_load_pq': gross_load_pq,
            #验证能守
            'S_HE_mw': S_HE_mw,
            'S_gen': S_gen,
            #奖励分解
            'C_gen': C_gen,
            'C_load_shed': C_load_shed
        }

        # 注意：gymnasium 需要返回 (obs, reward, terminated, truncated, info)
        # 这里 truncation 通常由外部包装器处理，返回 False
        return obs, reward, terminated, truncated, info

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec
class CustomTimeLimit(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will issue a `truncated` signal if a maximum number of timesteps is exceeded.

    If a truncation is not defined inside the environment itself, this is the only place that the truncation signal is issued.
    Critically, this is different from the `terminated` signal that originates from the underlying environment as part of the MDP.

    Example:
       >>> import gymnasium as gym
       >>> from gymnasium.wrappers import TimeLimit
       >>> env = gym.make("CartPole-v1")
       >>> env = TimeLimit(env, max_episode_steps=1000)
    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int,
    ):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``None``, ``env.spec.max_episode_steps`` is used)
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, max_episode_steps=max_episode_steps
        )
        gym.Wrapper.__init__(self, env)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        observation, reward, terminated, truncated, info = self.env.step(action)    #这里存储的都是要的信息
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
            observation, _ = self.reset()  # Reset the environment for the next episode
        if truncated:
            info['final_observation'] = observation
            info['final_info'] = info.copy()

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to include the `max_episode_steps=self._max_episode_steps`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = deepcopy(env_spec)
            env_spec.max_episode_steps = self._max_episode_steps

        self._cached_spec = env_spec
        return env_spec

from typing import Optional
from collections import deque
class CustomRecordEpisodeStatistics(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos
        )

def add_chart_to_tensorboard(writer, global_step, info):
    C_gen = info['C_gen']
    C_load_shed = info['C_load_shed']
    DCV = info['dcv']

    R0 = 100.0
    # epsilon0 = 1e-4
    epsilon0 = 8e-4     #5e-4
    DCV_penalty = min(R0, DCV / epsilon0)

    writer.add_scalar("charts/C_gen", C_gen, global_step)
    writer.add_scalar("charts/C_load_shed", C_load_shed, global_step)
    writer.add_scalar("charts/DCV_penalty", DCV_penalty, global_step)

def add_para_to_tensorboard(writer, global_step, info):
    gross_load_pq = info['gross_load_pq']
    served_load_pq = info['served_load_pq']

    gross_load_p = gross_load_pq[:len(gross_load_pq)//2]
    gross_load_q = gross_load_pq[len(gross_load_pq)//2:]
    gross_load_p_sum = np.sum(gross_load_p)
    gross_load_q_sum = np.sum(gross_load_q)
    gross_load_p_mean = np.mean(gross_load_p)
    gross_load_q_mean = np.mean(gross_load_q)
    gross_load_p_min = np.min(gross_load_p)
    gross_load_q_min = np.min(gross_load_q)
    writer.add_scalar("para/gross_load_p_sum-step", gross_load_p_sum, global_step)
    writer.add_scalar("para/gross_load_q_sum-step", gross_load_q_sum, global_step)
    writer.add_scalar("para/gross_load_p_mean-step", gross_load_p_mean, global_step)
    writer.add_scalar("para/gross_load_q_mean-step", gross_load_q_mean, global_step)
    writer.add_scalar("para/gross_load_p_min-step", gross_load_p_min, global_step)
    writer.add_scalar("para/gross_load_q_min-step", gross_load_q_min, global_step)

    served_load_p = served_load_pq[:len(served_load_pq)//2]
    served_load_q = served_load_pq[len(served_load_pq)//2:]
    served_load_p_sum = np.sum(served_load_p)
    served_load_q_sum = np.sum(served_load_q)
    served_load_p_mean = np.mean(served_load_p)
    served_load_q_mean = np.mean(served_load_q)
    served_load_p_min = np.min(served_load_p)
    served_load_q_min = np.min(served_load_q)
    writer.add_scalar("para/served_load_p_sum-step", served_load_p_sum, global_step)
    writer.add_scalar("para/served_load_q_sum-step", served_load_q_sum, global_step)
    writer.add_scalar("para/served_load_p_mean-step", served_load_p_mean, global_step)
    writer.add_scalar("para/served_load_q_mean-step", served_load_q_mean, global_step)
    writer.add_scalar("para/served_load_p_min-step", served_load_p_min, global_step)
    writer.add_scalar("para/served_load_q_min-step", served_load_q_min, global_step)

def check_conservation_of_energy(info, writer, global_step):
    served_load_pq = info['served_load_pq']
    S_gen = info['S_gen']   #为什么用info的S_gen,因为还要算上松弛节点的出力
    S_HE_mw = info['S_HE_mw']  #所有节点注入功率之和等于网损

    #分成有功与无功分别验证
    gen_p = np.real(S_gen)
    gen_q = np.imag(S_gen)

    served_load_p = served_load_pq[:len(served_load_pq)//2]
    served_load_q = served_load_pq[len(served_load_pq)//2:]

    loss_p = np.real(S_HE_mw)
    loss_q = np.imag(S_HE_mw)

    #开始计算综合
    total_gen_p = np.sum(gen_p)
    total_gen_q = np.sum(gen_q)

    total_loss_p = np.sum(loss_p)
    total_loss_q = np.sum(loss_q)

    total_served_load_p = np.sum(served_load_p)
    total_served_load_q = np.sum(served_load_q)

    #检验有功,无功残差
    residual_p = total_gen_p - total_served_load_p - total_loss_p
    residual_q = total_gen_q - total_served_load_q - total_loss_q

    # print(f"有功残差: {residual_p:.6f} MW")
    # print(f"无功残差: {residual_q:.6f} MVar")

    writer.add_scalar("para/residual_p-step", residual_p, global_step)
    writer.add_scalar("para/residual_q-step", residual_q, global_step)



if __name__ == "__main__":

    #先把m文件路径找到，并且运行一下case30，读取mpc
    eng = matlab.engine.start_matlab()
    path = eng.eval("which('case30.m')")
    print(path)  
    mpc = eng.case30()
    eng.quit()

    env = PowerSystemEnv(mpc, path)
    env = TimeLimit(env, max_episode_steps=288)
    env = RecordEpisodeStatistics(env)
    print("环境创建成功！")