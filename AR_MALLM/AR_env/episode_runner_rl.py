import os
import json
from AR_env_rl import Env
from User import update_time
from episode_batch import Episode_Batch
from agents_controller import BasicMAC
import numpy as np
from User import ObT
import torch as th


class EpisodeRunner:
    def __init__(self, config):
        self.config = config
        self.t_env = 0
        self.t_ep = 0
        self.t_ep_max = self.config['RL_config']['max_seq_length']

        # 初始化数据收集相关变量
        self.training_data = []  # 存储训练过程中的所有数据
        self.current_episode_data = []  # 临时存储当前episode的数据
        self.current_episode = 0  # 当前episode编号

    def setup(self, mac: BasicMAC):
        self.mac = mac

    def reset(self):
        self.t_ep = 0
        self.batch = Episode_Batch()
        self.env = Env(self.config)

        # 清空当前episode数据
        self.current_episode_data = []

    def save_training_data(self, filename='training_data.json'):
        """保存训练数据到文件"""
        if not os.path.exists(self.config['Ex_config']['res_path']):
            os.makedirs(self.config['Ex_config']['res_path'])
        
        filepath = os.path.join(self.config['Ex_config']['res_path'], filename)
        with open(filepath, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        
        print(f"Training data saved to {filepath}")



    def run(self, test_mode):
        self.reset()

         
        while True:
            if self.t_ep >= self.t_ep_max:  # 超过最大步数则终止
                break
            self.env.start()
            for i in range(self.env.max_user_num):
                if self.env.Users[i].active_flag and self.t_ep < self.t_ep_max:
                    self.env.user_active_flag[self.t_ep][i] = 1

            if (self.env.sys_time - update_time) in self.env.stage_time or (self.env.sys_time in self.env.stage_time):

                obs = self.env.get_obs()
                state = self.env.get_state()
                avail_actions = self.env.get_avail_actions()
                self.batch.update_batch('state', state, self.t_ep)
                self.batch.update_batch('obs', obs, self.t_ep)
                self.batch.update_batch('avail_actions', avail_actions, self.t_ep)
                actions = self.mac.select_actions(self.batch, t_ep=self.t_ep, t_env=self.t_env, test_mode=test_mode)

                ############################
                # 收集数据 - 在动作选择后
                obs_numpy = obs.numpy() if hasattr(obs, 'numpy') else obs
                actions_numpy = actions.numpy() if hasattr(actions, 'numpy') else actions
                
                for user_id in range(self.env.max_user_num):
                    if self.env.Users[user_id].active_flag:
                        data = {
                            'episode': self.current_episode,
                            'step': self.t_ep,
                            'agent_id': user_id,
                            'obs': obs_numpy[user_id].tolist(),
                            'agent_action': int(actions_numpy[0][user_id]),
                            'agent_active': True,
                            'agent_done': False,
                            'timestamp': self.env.sys_time,
                            'agent_reward': 0.0,  # 稍后更新
                            'reward': 0.0  # 稍后更新
                        }
                        self.current_episode_data.append(data)
                
                ##################

                for user_id in self.env.newest_arrive_user:
                    actions[0][user_id] = self.config['AR_env_config']['init_actions'][user_id]
                    if (self.env.sys_time - update_time) in self.env.stage_time:
                        self.env.newest_arrive_user.remove(user_id)
                self.batch.update_batch('actions', actions, self.t_ep)
                for user in self.env.Users:
                    if user.stop_flag is False and user.active_flag is True:
                        user.update_parameters(self.env.sys_time, self.env.user_alg_name, action=int(actions[0][user.id]), t_ep=self.t_ep)
                        self.env.queries_num += len(user.cur_query_list)
                self.t_ep += 1


            elif self.env.sys_time % update_time == 0 and not self.env.stopflag:

                obs = self.env.get_obs()
                state = self.env.get_state()
                avail_actions = self.env.get_avail_actions()
                self.batch.update_batch('state', state, self.t_ep)
                self.batch.update_batch('obs', obs, self.t_ep)
                self.batch.update_batch('avail_actions', avail_actions, self.t_ep)
                actions = self.mac.select_actions(self.batch, t_ep=self.t_ep, t_env=self.t_env, test_mode=test_mode)

                #############################
                # 收集数据 - 在动作选择后
                obs_numpy = obs.numpy() if hasattr(obs, 'numpy') else obs
                actions_numpy = actions.numpy() if hasattr(actions, 'numpy') else actions
                
                for user_id in range(self.env.max_user_num):
                    if self.env.Users[user_id].active_flag:
                        data = {
                            'episode': self.current_episode,
                            'step': self.t_ep,
                            'agent_id': user_id,
                            'obs': obs_numpy[user_id].tolist(),
                            'agent_action': int(actions_numpy[0][user_id]),
                            'agent_active': True,
                            'agent_done': False,
                            'timestamp': self.env.sys_time,
                            'agent_reward': 0.0,  # 稍后更新
                            'reward': 0.0  # 稍后更新
                        }
                        self.current_episode_data.append(data)
                ###########################

                self.batch.update_batch('actions', actions, self.t_ep)
                for user in self.env.Users:
                    if user.stop_flag is False and user.active_flag is True:
                        user.update_parameters(self.env.sys_time, self.env.user_alg_name, action=actions[0][user.id], t_ep=self.t_ep)
                        if self.env.sys_time == user.gene_time + self.env.user_active_time - update_time:
                            user.stop_time = user.cur_query_list[-1].arrivalTime
                            user.last_query = user.cur_query_list[-1]
                        self.env.queries_num += len(user.cur_query_list)
                self.t_ep += 1


            if self.env.stopflag:
                self.batch.update_batch('terminated', 1, self.t_ep-1)
            self.env.run()
            if self.env.realstopflag is True:


                last_state = self.env.get_state()
                last_obs = self.env.get_obs()
                last_avail_actions = self.env.get_avail_actions()
                self.batch.update_batch('state', last_state, self.t_ep)
                self.batch.update_batch('obs', last_obs, self.t_ep)
                self.batch.update_batch('avail_actions', last_avail_actions, self.t_ep)
                last_actions = self.mac.select_actions(self.batch, t_ep=self.t_ep, t_env=self.t_env, test_mode=test_mode)
                self.batch.update_batch('actions', last_actions, self.t_ep)
                break
            self.env.sys_time += 1
        
        r = self.env.get_all_reward()
        sum_reward = np.sum(r)
        for i in range(self.t_ep_max):
            self.batch.update_batch('reward', r[i], i)
            r_independent = self.env.get_tp_reward(i)

            ####################
            # 更新收集的数据中的奖励信息
            for data in self.current_episode_data:
                if data['step'] == i:
                    agent_id = data['agent_id']
                    data['agent_reward'] = float(r_independent[agent_id])
                    data['reward'] = float(r[i])
                    # 更新done标志
                    user = self.env.Users[agent_id]
                    user_stop_time = user.gene_time + self.env.user_active_time - update_time
                    data['agent_done'] = (
                        i == self.t_ep_max - 1 #or
                        # self.env.sys_time >= user_stop_time or
                        # not user.active_flag
                    )
            ###################

        if not test_mode:
            self.t_env += self.t_ep

        ############
        # 将当前episode数据添加到总数据中
        self.training_data.extend(self.current_episode_data)
        # 定期保存数据
        if self.current_episode % 1 == 0:
            self.save_training_data(f'training_data_ep{self.current_episode}.json')
        #############


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

        sum = np.sum(user_sum)
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

        return self.batch, sum, std, all_std, sum_reward, ctime_lst



