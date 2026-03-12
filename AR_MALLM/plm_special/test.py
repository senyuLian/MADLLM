import torch
from rw_config import get_config
from AR_env.User import update_time, ObT
import numpy as np  
from plm_special.utils.utils import set_random_seed
from AR_env.AR_env_rl import Env

class test_on_env:
    def __init__(self):
        self.config = get_config('config')
        self.t_env = 0
        self.t_ep = 0
        self.t_ep_max = self.config['RL_config']['max_seq_length']
        self.penalty = self.config['AR_env_config']['penalty']

    def reset(self):
        self.t_ep = 0
        self.env = Env(self.config)

    def evaluate_on_AR_Env(self, args, model, target_return, process_reward_fn=None):
        if process_reward_fn is None:
            process_reward_fn = lambda x: x
        self.reset()

        with torch.no_grad():
            timesteps = [0] * self.env.max_user_num
            episodes_return, episodes_len = 0, 0
            
            set_random_seed(args.seed)
            actions = np.zeros((1, self.env.max_user_num))  # 形状为(1, n_agents)
      
        while True:
            if self.t_ep >= self.t_ep_max:  # 超过最大步数则终止
                break

            print(self.t_ep,timesteps)
            self.env.start()
            for i in range(self.env.max_user_num):
                if self.env.Users[i].active_flag and self.t_ep < self.t_ep_max:
                    self.env.user_active_flag[self.t_ep][i] = 1
            #############
            if self.env.sys_time % update_time == 0 and any(user.active_flag == 1 for user in self.env.Users):                
                if self.t_ep > 0: # 跳过初始状态
                    reward = self.env.get_current_reward(self.t_ep-1)   #所有活跃用户该时刻内的qos均值-公平性惩罚值
                    reward = process_reward_fn(reward)
                    target_return -= reward
                    episodes_return += reward
                    episodes_len += 1
            #############

            if (self.env.sys_time - update_time) in self.env.stage_time or (self.env.sys_time in self.env.stage_time): #有新用户到达的时刻
                obs = self.env.get_obs()
                ##############
                for user_id in range(self.env.max_user_num):
                    user = self.env.Users[user_id]                   
                    # 跳过新到达用户（已经处理过）
                    if user_id in self.env.newest_arrive_user:
                        continue                     
                    # 跳过不活跃用户
                    if not user.active_flag or user.stop_flag:
                        continue                    
                    # 为活跃用户生成动作
                    user_obs = obs[user_id].float().unsqueeze(0).unsqueeze(0) #使用系统时间，不需要自己传时间参数
                    pre_r = self.env.get_tp_pre_add_r(self.t_ep, user_id)
                    user_action = model.sample(user_id, pre_r, user_obs, target_return, timesteps[user_id])
                    timesteps[user_id] += 1
                    actions[0][user_id] = user_action
                ##########################

                for user_id in self.env.newest_arrive_user: #处理新到达的用户
                    actions[0][user_id] = self.config['AR_env_config']['init_actions'][user_id]
                    if (self.env.sys_time - update_time) in self.env.stage_time:
                        self.env.newest_arrive_user.remove(user_id)

                for user in self.env.Users: ###更新用户对象
                    if user.stop_flag is False and user.active_flag is True:
                        user.update_parameters(self.env.sys_time, self.env.user_alg_name, action=int(actions[0][user.id]), t_ep=self.t_ep)
                        self.env.queries_num += len(user.cur_query_list)
                self.t_ep += 1


            elif self.env.sys_time % update_time == 0 and not self.env.stopflag: #更新节点

                obs = self.env.get_obs()
                #########################
                for user_id in range(self.env.max_user_num):
                    user = self.env.Users[user_id]                   
                    # 跳过新到达用户（已经处理过）
                    if user_id in self.env.newest_arrive_user:
                        continue                     
                    # 跳过不活跃用户
                    if not user.active_flag or user.stop_flag:
                        continue                    
                    # 为活跃用户生成动作
                    user_obs = obs[user_id].float().unsqueeze(0).unsqueeze(0) #使用系统时间，不需要自己传时间参数
                    pre_r = self.env.get_tp_pre_add_r(self.t_ep, user_id)
                    user_action = model.sample(user_id, pre_r, user_obs, target_return, timesteps[user_id])
                    timesteps[user_id] += 1
                    actions[0][user_id] = user_action
                ##########################

                for user in self.env.Users:
                    if user.stop_flag is False and user.active_flag is True:
                        user.update_parameters(self.env.sys_time, self.env.user_alg_name, action=actions[0][user.id], t_ep=self.t_ep)
                        if self.env.sys_time == user.gene_time + self.env.user_active_time - update_time:
                            user.stop_time = user.cur_query_list[-1].arrivalTime
                            user.last_query = user.cur_query_list[-1]
                        self.env.queries_num += len(user.cur_query_list)
                self.t_ep += 1

            self.env.run()
            if self.env.stopflag:
                break
            
            if self.env.realstopflag is True:
                break
            self.env.sys_time += 1
        
        r = self.env.get_all_reward()
        sum_reward = np.sum(r) #一个episode中，不同时间步的总reward之和
        print(sum_reward)

        ctime_lst = []
        for i in range(len(self.env.User_Qos_memory)):
            start_time = self.env.User_Qos_memory[i][3]
            end_time = self.env.User_Qos_memory[i][4]
            ctime_lst.append((end_time-start_time) * ObT)

        user_t = []
        user_y = []

        for i in range(0, self.env.max_user_num):
            user_t.append([])
            user_y.append([])

        for i in range(0, len(self.env.User_Qos_memory)):
            for j in range(0, self.env.max_user_num):
                if self.env.User_Qos_memory[i][0] == j:
                    user_t[j].append(self.env.User_Qos_memory[i][3])
                    user_y[j].append(self.env.User_Qos_memory[i][1])

        user_sum = np.zeros(self.env.max_user_num)
        for i in range(self.env.max_user_num):
            user_sum[i] = np.sum(user_y[i])

        sum = np.sum(user_sum) #所有用户所有请求的qos之和
        std = np.std(user_sum)

        # the following part is for stage evaluation
        stage_time = self.config['AR_env_config']['user_arrive_time']
        user_active_time = self.config['AR_env_config']['user_active_time'] / ObT  # 每位用户活跃的时间, 单位ObT
        real_stage_time = np.append(np.array(stage_time), np.array(stage_time) + np.ones(self.env.max_user_num) * user_active_time)
        stages = []
        for i in range(len(real_stage_time) - 3):
            stages.append([real_stage_time[i + 1], real_stage_time[i + 2]])

        stages_index = [[0, 2], [0, 3], [0, 4], [0, 5],
                        [1, 5], [2, 5], [3, 5]]

        user_sum4stages = np.zeros([len(stages), self.env.max_user_num])
        for i in range(self.env.max_user_num):
            for j in range(len(user_t[i])):
                for k in range(len(stages)):
                    if stages[k][0] <= user_t[i][j] < stages[k][1]:
                        user_sum4stages[k][i] += user_y[i][j]
        sum4stages = np.zeros(len(stages))
        std4stages = np.zeros(len(stages))
        for k in range(len(stages)):
            sum4stages[k] = np.sum(user_sum4stages[k])
            std4stages[k] = np.std(user_sum4stages[k][stages_index[k][0]:stages_index[k][1]])
        all_std = np.sum(std4stages)

        if std == all_std:
            print('yes')
        

        return  sum, std, all_std, sum_reward, ctime_lst


