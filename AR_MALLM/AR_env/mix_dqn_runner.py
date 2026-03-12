# one DQN agent vs four QoE-based agents
from AR_env_rl import Env
from rw_config import get_config
import numpy as np
from User import update_time
import torch.optim as optim
from DQN_Agent import ReplayBuffer, Qnet, train
import os
import copy
import torch as th
from assist_func import save_variable
import json,os  


class DQNRunner:
    def __init__(self):
        self.config = get_config('config')
        self.t_ep = 0
        self.t_env = 0
        self.t_ep_max = self.config['RL_config']['max_seq_length']
        self.max_episode = self.config['Ex_config']['max_episode_num']

        # self.AgentNum = self.config['env_config']['n_agent']
        self.AgentNum = 1  # 只有一个DQN Agent

        self.AgentList = []
        self.Q_targetList = []
        self.OptimizerList = []
        self.lr = self.config['dqn_config']['lr']
        self.AgentDevice = self.config['dqn_config']['device']
        self.n_state = self.config['agent_config']['n_obs']
        self.n_action = self.config['agent_config']['n_action']
        self.env = None
        self.MemoryList = []

        self.training_data = []  # 存储训练过程中的所有数据
        self.current_episode_data = []  # 临时存储当前episode的数据

        self.loss_history = []

    def save_training_data(self, filename='training_data.json'):
        """保存训练数据到文件"""
        import json
        if not os.path.exists(self.config['Ex_config']['res_path']):
            os.makedirs(self.config['Ex_config']['res_path'])
        
        filepath = os.path.join(self.config['Ex_config']['res_path'], filename)
        with open(filepath, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        
        print(f"Training data saved to {filepath}")

    def setup(self):
        for _ in range(self.AgentNum):
            self.MemoryList.append(ReplayBuffer(self.config))
            self.AgentList.append(Qnet(n_state=self.n_state, n_action=self.n_action-1).to(self.AgentDevice))  # Independent Agent 不需要free动作
            self.Q_targetList.append(Qnet(n_state=self.n_state, n_action=self.n_action-1).to(self.AgentDevice))  # Target Network
        for i in range(self.AgentNum):
            self.OptimizerList.append(optim.Adam(self.AgentList[i].parameters(), lr=self.lr))
            self.Q_targetList[i].load_state_dict(self.AgentList[i].state_dict())  # Target Network Copy

    def reset(self):
        self.t_ep = 0
        self.env = Env(self.config)

    def train(self):
        interval = 0
        s_lst = []
        a_lst = []
        r_lst = []
        d_lst = []

        for i in range(self.AgentNum):
            s_lst.append([])
            a_lst.append([])
            r_lst.append([])
            d_lst.append([])

        for m in range(self.max_episode):
            epsilon = max(0.01, 0.2 - 0.19 * (m / 5000))  # Linear annealing from 20% to 5%
            self.reset()

            while True:
                self.env.start()
                for i in range(self.env.max_user_num):
                    if self.env.Users[i].active_flag and self.t_ep < self.t_ep_max:
                        self.env.user_active_flag[self.t_ep][i] = 1
                if (self.env.sys_time - update_time) in self.env.stage_time or (self.env.sys_time in self.env.stage_time):
                    states = self.env.get_obs().numpy()
                    avail_actions = self.env.get_avail_actions().numpy()
                    # actions = np.array([])
                    actions = np.zeros(self.env.max_user_num)

                    


                    # for j in range(self.env.max_user_num):
                    for j in range(self.AgentNum):  # Agent 数量为 1
                        if avail_actions[j][0] == 1.0:
                            # actions = np.append(actions, -1)  # 非active用户，不执行有效动作
                            actions[j] = -1
                        else:
                            action = self.AgentList[j].sample_action(th.from_numpy(states[j]).float().to(self.AgentDevice), epsilon)
                            # actions = np.append(actions, action)
                            actions[j] = action

                    

                    # for j in range(self.env.max_user_num):
                    for j in range(self.AgentNum):  # Agent 数量为 1
                        if self.env.Users[j].active_flag is True and j not in self.env.newest_arrive_user and actions[j] != -1:

                            data = {
                            'episode': m,
                            'step': self.t_ep,
                            'dqn_obs': states[j].tolist(),  # DQN agent的观测
                            'dqn_action': int(actions[j]),   # DQN agent的动作
                            'dqn_done': False,              # 暂时设为False，后面会更新
                            'timestamp': self.env.sys_time
                            }
                            self.current_episode_data.append(data)

                            s_lst[j].append(states[j])
                            a_lst[j].append(actions[j])
                            actions[j] += 1
                            if self.t_ep != self.t_ep_max:
                                d_lst[j].append(1)
                            else:
                                d_lst[j].append(0)
                            if len(s_lst[j]) >= 2 and self.t_ep >= 2:
                                # r_lst[j].append(self.env.get_tp_reward(self.t_ep - 2)[j]) ######### !!!!!!!单个dqn智能体的奖励
                                r_lst[j].append(self.env.get_all_reward_at_step(self.t_ep - 1))   # 所有用户的总奖励
                                state = s_lst[j].pop(0)
                                action = a_lst[j].pop(0)
                                reward = r_lst[j].pop(0)
                                done = d_lst[j].pop(0)
                                self.MemoryList[j].put([state, int(action), reward, s_lst[j][0], done])
                            if self.MemoryList[j].size() > 500:
                                interval += 1
                                train(self.AgentList[j], self.Q_targetList[j], self.MemoryList[j], self.OptimizerList[j], runner=self)
                                if interval % 10 == 0:
                                    for w in range(self.AgentNum):
                                        self.Q_targetList[w].load_state_dict(self.AgentList[w].state_dict())
                    for user_id in copy.deepcopy(self.env.newest_arrive_user):
                        actions[user_id] = self.config['AR_env_config']['init_actions'][user_id]
                        if (self.env.sys_time - update_time) in self.env.stage_time:
                            self.env.newest_arrive_user.remove(user_id)
                    for user in self.env.Users:
                        if user.stop_flag is False and user.active_flag is True:
                            if user.id == 0:
                                user.update_parameters(self.env.sys_time, self.env.user_alg_name,
                                                       action=actions[user.id], t_ep=self.t_ep)
                            else:
                                user.update_parameters(self.env.sys_time, 'qoe_based',
                                                       action=actions[user.id], t_ep=self.t_ep)
                            self.env.queries_num += len(user.cur_query_list)
                    self.t_ep += 1

                elif self.env.sys_time % update_time == 0 and not self.env.stopflag:
                    states = self.env.get_obs().numpy()
                    avail_actions = self.env.get_avail_actions().numpy()
                    # actions = np.array([])
                    actions = np.zeros(self.env.max_user_num)

                    # for j in range(self.env.max_user_num):
                    for j in range(self.AgentNum):  # Agent 数量为 1
                        if avail_actions[j][0] == 1.0:
                            actions[j] = -1  # 非active用户，不执行有效动作
                        else:
                            action = self.AgentList[j].sample_action(th.from_numpy(states[j]).float().to(self.AgentDevice), epsilon)
                            # actions = np.append(actions, action)
                            actions[j] = action

                    # for j in range(self.env.max_user_num):
                    for j in range(self.AgentNum):  # Agent 数量为 1
                        if self.env.Users[j].active_flag is True and j not in self.env.newest_arrive_user and actions[j] != -1:

                            data = {
                            'episode': m,
                            'step': self.t_ep,
                            'dqn_obs': states[j].tolist(),  # DQN agent的观测
                            'dqn_action': int(actions[j]),   # DQN agent的动作
                            'dqn_done': False,              # 暂时设为False，后面会更新
                            'timestamp': self.env.sys_time
                            }
                            self.current_episode_data.append(data)

                            s_lst[j].append(states[j])
                            a_lst[j].append(actions[j])
                            actions[j] += 1
                            if self.t_ep != self.t_ep_max:
                                d_lst[j].append(1)
                            else:
                                d_lst[j].append(0)
                            if len(s_lst[j]) >= 2 and self.t_ep >= 2:
                                r_lst[j].append(self.env.get_all_reward_at_step(self.t_ep - 1))
                                # r_lst[j].append(self.env.get_tp_reward(self.t_ep - 2)[j])
                                state = s_lst[j].pop(0)
                                action = a_lst[j].pop(0)
                                reward = r_lst[j].pop(0)
                                done = d_lst[j].pop(0)
                                self.MemoryList[j].put([state, int(action), reward, s_lst[j][0], done])
                            if self.MemoryList[j].size() > 500:
                                interval += 1
                                train(self.AgentList[j], self.Q_targetList[j], self.MemoryList[j], self.OptimizerList[j], runner=self)
                                if interval % 10 == 0:
                                    for w in range(self.AgentNum):
                                        self.Q_targetList[w].load_state_dict(self.AgentList[w].state_dict())
                    for user in self.env.Users:
                        if user.stop_flag is False and user.active_flag is True:
                            if user.id == 0:
                                user.update_parameters(self.env.sys_time, self.env.user_alg_name,
                                                       action=actions[user.id], t_ep=self.t_ep)
                            else:
                                user.update_parameters(self.env.sys_time, 'qoe_based',
                                                       action=actions[user.id], t_ep=self.t_ep)
                            if self.env.sys_time == user.gene_time + self.env.user_active_time - update_time:
                                user.stop_time = user.cur_query_list[-1].arrivalTime
                                user.last_query = user.cur_query_list[-1]
                            self.env.queries_num += len(user.cur_query_list)
                    self.t_ep += 1
                self.env.run()
                if self.env.realstopflag is True:
                    break
                self.env.sys_time += 1

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

            for data in self.current_episode_data:
                data['total_qoe'] = float(np.sum(user_sum))
                data['qoe_std'] = float(np.std(user_sum))
                # 更新done标志
                data['dqn_done'] = (data['step'] == self.t_ep_max - 1)  # 如果是最后一步则为done

            self.training_data.extend(self.current_episode_data)
            self.current_episode_data = []  # 重置当前episode数据

            sum = np.sum(user_sum)
            # std = np.std(user_sum)

            # print('episode ' + str(m) + 'dqnqos: ' + str(user_sum[0]))
            print('episode ' + str(m) + 'sum: ' + str(sum))
            # print('episode ' + str(m) + 'std: ' + str(std))

            if not os.path.exists(self.config['Ex_config']['model_path']):
                os.mkdir(self.config['Ex_config']['model_path'])
            if not os.path.exists(self.config['Ex_config']['res_path']):
                os.mkdir(self.config['Ex_config']['res_path'])

            if m % 20 == 0:
                for j in range(self.AgentNum):
                    th.save(self.AgentList[j].state_dict(), self.config['Ex_config']['model_path'] + 'agent-' + str(m) + '-' + str(j) + '.pkl')

            # 每100个episode保存一次数据
            if m % 100 == 0 or m == self.max_episode - 1:
                self.save_training_data(f'training_data_ep{m}.json')

        
        with open(os.path.join(self.config['Ex_config']['res_path'], 'loss.json'), 'w') as f:
            json.dump(self.loss_history, f)

    def eval(self):
        if not os.path.exists(self.config['Ex_config']['res_path']):
            os.mkdir(self.config['Ex_config']['res_path'])
        np.random.seed(0)
        th.manual_seed(self.config['Ex_config']['random_seed'])
        # max_user_num = self.config['AR_env_config']['max_user_num']

        a_sum_list = []
        sum_list = []
        std_list = []
        for m in range(0, self.config['Ex_config']['test_num']):
            self.reset()
            trained_agents_list = []
            for j in range(self.AgentNum):
                file_name = self.config['Ex_config']['model_path'] + 'agent-' + str(m * self.config['Ex_config']['test_step']) + '-' + str(j) + '.pkl'
                if os.path.isfile(file_name):
                    trained_agents_list.append(th.load(file_name))  # 加载模型

            for j in range(self.AgentNum):
                runner.AgentList[j].load_state_dict(trained_agents_list[j])  # 模型赋值

            while True:
                self.env.start()
                for i in range(self.env.max_user_num):
                    if self.env.Users[i].active_flag and self.t_ep < self.t_ep_max:
                        self.env.user_active_flag[self.t_ep][i] = 1
                if (self.env.sys_time - update_time) in self.env.stage_time or (self.env.sys_time in self.env.stage_time):
                    states = self.env.get_obs().numpy()
                    avail_actions = self.env.get_avail_actions().numpy()
                    # actions = np.array([])
                    actions = np.zeros(self.env.max_user_num)

                    # for j in range(self.env.max_user_num):
                    for j in range(self.AgentNum):
                        if avail_actions[j][0] == 1.0:
                            actions[j] = -1  # 非active用户，不执行有效动作
                        else:
                            action = self.AgentList[j].sample_action(th.from_numpy(states[j]).float().to(self.AgentDevice), 0)  # 测试过程不使用greedy探索
                            actions[j] = action
                    # for j in range(self.env.max_user_num):
                    for j in range(self.AgentNum):
                        if self.env.Users[j].active_flag is True and j not in self.env.newest_arrive_user and actions[j] != -1:
                            actions[j] += 1
                    for user_id in copy.deepcopy(self.env.newest_arrive_user):
                        actions[user_id] = self.config['AR_env_config']['init_actions'][user_id]
                        if (self.env.sys_time - update_time) in self.env.stage_time:
                            self.env.newest_arrive_user.remove(user_id)
                    for user in self.env.Users:
                        if user.stop_flag is False and user.active_flag is True:
                            if user.id == 0:
                                user.update_parameters(self.env.sys_time, self.env.user_alg_name,
                                                       action=actions[user.id], t_ep=self.t_ep)
                            else:
                                user.update_parameters(self.env.sys_time, 'qoe_based',
                                                       action=actions[user.id], t_ep=self.t_ep)
                            self.env.queries_num += len(user.cur_query_list)
                    self.t_ep += 1
                elif self.env.sys_time % update_time == 0 and not self.env.stopflag:
                    states = self.env.get_obs().numpy()
                    avail_actions = self.env.get_avail_actions().numpy()
                    # actions = np.array([])
                    actions = np.zeros(self.env.max_user_num)

                    # for j in range(self.env.max_user_num):
                    for j in range(self.AgentNum):
                        if avail_actions[j][0] == 1.0:
                            actions = np.append(actions, -1)  # 非active用户，不执行有效动作
                        else:
                            action = self.AgentList[j].sample_action(th.from_numpy(states[j]).float().to(self.AgentDevice), 0)
                            actions[j] = action
                    # for j in range(self.env.max_user_num):
                    for j in range(self.AgentNum):
                        if self.env.Users[j].active_flag is True and j not in self.env.newest_arrive_user and actions[j] != -1:
                            actions[j] += 1
                    for user in self.env.Users:
                        if user.stop_flag is False and user.active_flag is True:
                            if user.id == 0:
                                user.update_parameters(self.env.sys_time, self.env.user_alg_name, action=actions[user.id], t_ep=self.t_ep)
                            else:
                                user.update_parameters(self.env.sys_time, 'qoe_based', action=actions[user.id], t_ep=self.t_ep)
                            if self.env.sys_time == user.gene_time + self.env.user_active_time - update_time:
                                user.stop_time = user.cur_query_list[-1].arrivalTime
                                user.last_query = user.cur_query_list[-1]
                            self.env.queries_num += len(user.cur_query_list)
                    self.t_ep += 1
                self.env.run() 
                if self.env.realstopflag is True:
                    break
                self.env.sys_time += 1
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

            Sum = np.sum(user_sum)
            Std = np.std(user_sum)

            sum_list.append(Sum)
            std_list.append(Std)
            a_sum_list.append(user_sum[0])
            with open(self.config['Ex_config']['res_path'] + 'qos_list', 'a') as qos_file:
                qos_file.write(str(Sum) + '\n')
            with open(self.config['Ex_config']['res_path'] + 'std_list', 'a') as std_file:
                std_file.write(str(Std) + '\n')
            with open(self.config['Ex_config']['res_path'] + 'a_qos_list', 'a') as a_qos_file:
                a_qos_file.write(str(user_sum[0]) + '\n')

    def test(self):
        if not os.path.exists(self.config['Ex_config']['res_path']):
            os.mkdir(self.config['Ex_config']['res_path'])
        np.random.seed(0)
        th.manual_seed(self.config['Ex_config']['random_seed'])

        self.reset()
        trained_agents_list = []
        for j in range(self.AgentNum):
            file_name = self.config['Ex_config']['model_path'] + 'agent-' + str(3300) + '-' + str(j) + '.pkl'
            if os.path.isfile(file_name):
                trained_agents_list.append(th.load(file_name, map_location='cpu'))  # 加载模型

        for j in range(self.AgentNum):
            runner.AgentList[j].load_state_dict(trained_agents_list[j])  # 模型赋值
        while True:
            self.env.start()
            for i in range(self.env.max_user_num):
                if self.env.Users[i].active_flag and self.t_ep < self.t_ep_max:
                    self.env.user_active_flag[self.t_ep][i] = 1
            if (self.env.sys_time - update_time) in self.env.stage_time or (self.env.sys_time in self.env.stage_time):
                states = self.env.get_obs().numpy()
                avail_actions = self.env.get_avail_actions().numpy()
                # actions = np.array([])
                actions = np.zeros(self.env.max_user_num)

                # for j in range(self.env.max_user_num):
                for j in range(self.AgentNum):
                    if avail_actions[j][0] == 1.0:
                        actions[j] = -1  # 非active用户，不执行有效动作
                    else:
                        action = self.AgentList[j].sample_action(th.from_numpy(states[j]).float().to(self.AgentDevice), 0)  # 测试过程不使用greedy探索
                        actions[j] = action
                # for j in range(self.env.max_user_num):
                for j in range(self.AgentNum):
                    if self.env.Users[j].active_flag is True and j not in self.env.newest_arrive_user and actions[j] != -1:
                        actions[j] += 1
                for user_id in copy.deepcopy(self.env.newest_arrive_user):
                    actions[user_id] = self.config['AR_env_config']['init_actions'][user_id]
                    if (self.env.sys_time - update_time) in self.env.stage_time:
                        self.env.newest_arrive_user.remove(user_id)
                for user in self.env.Users:
                    if user.stop_flag is False and user.active_flag is True:
                        if user.id == 0:
                            user.update_parameters(self.env.sys_time, self.env.user_alg_name,
                                                   action=actions[user.id], t_ep=self.t_ep)
                        else:
                            user.update_parameters(self.env.sys_time, 'qoe_based',
                                                   action=actions[user.id], t_ep=self.t_ep)
                        self.env.queries_num += len(user.cur_query_list)
                self.t_ep += 1
            elif self.env.sys_time % update_time == 0 and not self.env.stopflag:
                states = self.env.get_obs().numpy()
                avail_actions = self.env.get_avail_actions().numpy()
                # actions = np.array([])
                actions = np.zeros(self.env.max_user_num)

                # for j in range(self.env.max_user_num):
                for j in range(self.AgentNum):
                    if avail_actions[j][0] == 1.0:
                        actions = np.append(actions, -1)  # 非active用户，不执行有效动作
                    else:
                        action = self.AgentList[j].sample_action(th.from_numpy(states[j]).float().to(self.AgentDevice), 0)
                        actions[j] = action
                # for j in range(self.env.max_user_num):
                for j in range(self.AgentNum):
                    if self.env.Users[j].active_flag is True and j not in self.env.newest_arrive_user and actions[j] != -1:
                        actions[j] += 1
                for user in self.env.Users:
                    if user.stop_flag is False and user.active_flag is True:
                        if user.id == 0:
                            user.update_parameters(self.env.sys_time, self.env.user_alg_name, action=actions[user.id], t_ep=self.t_ep)
                        else:
                            user.update_parameters(self.env.sys_time, 'qoe_based', action=actions[user.id], t_ep=self.t_ep)
                        if self.env.sys_time == user.gene_time + self.env.user_active_time - update_time:
                            user.stop_time = user.cur_query_list[-1].arrivalTime
                            user.last_query = user.cur_query_list[-1]
                        self.env.queries_num += len(user.cur_query_list)
                self.t_ep += 1
            self.env.run()
            if self.env.realstopflag is True:
                break
            self.env.sys_time += 1
        save_variable(self.env.Batches, config['Ex_config']['res_path'] + 'batch')
        save_variable(self.env.User_Qos_memory, config['Ex_config']['res_path'] + 'user_qos')
        for i in range(self.env.max_user_num):
            # save_variable(batch['obs'].numpy()[0][:, 0], config['Ex_config']['res_path'] + 'obs_' + str(i))
            save_variable(self.env.Users[i].frame_rate_memory, config['Ex_config']['res_path'] + 'frame_' + str(i))
            save_variable(self.env.Users[i].dense_rate_memory, config['Ex_config']['res_path'] + 'dense_' + str(i))


if __name__ == '__main__':
    config = get_config('config')
    np.random.seed(0)
    th.manual_seed(config['Ex_config']['random_seed'])
    runner = DQNRunner()
    runner.setup()
    runner.train()
    #runner.eval()
    # runner.test()
