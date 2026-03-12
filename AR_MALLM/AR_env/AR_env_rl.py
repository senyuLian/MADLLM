from .User import User, ObT
from .BaseStation import BaseStation, Batch
import numpy as np
import torch as th
import math
from .assist_func import one_hot
import copy


class Env(object):
    def __init__(self, config):
        self.realstopflag = False
        self.stopflag = False

        self.max_t_ep = config['RL_config']['max_seq_length']

        self.Users = []  # 用户列表
        self.User_num = 0  # 活跃用户数量
        self.max_user_num = config['AR_env_config']['max_user_num']  # 最大用户数量
        self.user_active_flag = np.zeros([self.max_t_ep, self.max_user_num])

        self.BaseStation = BaseStation()
        self.bs_alg_name = config['AR_env_config']['BS_alg']
        self.query_arrival_time_list = []  # 记录所有请求到达时间
        self.sys_time = 0  # 系统时间，单位ObT

        self.n_action = config['agent_config']['n_action']
        self.n_obs = config['agent_config']['n_obs']

        self.Qos_memory = []  # 保存请求的用户以及QoS
        self.User_Qos_memory = []  # 保存用户角度的请求
        self.queries_num = 0  # 系统在时间段内处理的请求数量
        self.Batches = []  # 记录系统的batch选择

        self.newest_arrive_user = []  # 记录最新到达用户的id
        self.user_arrive_time = config['AR_env_config']['user_arrive_time']  # 3位用户到达的时间
        self.stage_time = self.user_arrive_time  # 有用户到达的stage开始的时间
        self.user_active_time = config['AR_env_config']['user_active_time'] / ObT  # 每位用户活跃的时间, 单位ObT
        self.real_stage_time = np.append(np.array(self.stage_time), np.array(self.stage_time) + np.ones(self.max_user_num) * self.user_active_time)
        self.real_stage_t_ep = (self.real_stage_time * ObT).astype(np.int32)

        self.penalty = config['AR_env_config']['penalty']
        self.throughput = config['AR_env_config']['throughput']
        self.gpu_process_up = config['AR_env_config']['gpu_process_up']

        self.max_band = config['AR_env_config']['max_band']
        self.min_band = config['AR_env_config']['min_band']
        self.users_band_traces = []
        self.user_alg_name = config['AR_env_config']['user_alg']
        self.normalize_r_flag = config['RL_config']['normalize_r_flag']

        if 'fix' in config['AR_env_config']['user_band_trace']:
            for i in range(self.max_user_num):
                band = np.array([config['AR_env_config']['fixed_band'][i]])
                self.users_band_traces.append(band.repeat(config['AR_env_config']['user_active_time']))
        else:
            self.users_band_traces_id = np.loadtxt(config['AR_env_config']['user_band_trace_path']
                                                   + config['AR_env_config']['user_band_trace'] + '/ids')[config['AR_env_config']['user_band_trace_group_id']]
            for i in range(self.max_user_num):
                self.users_band_traces.append(np.loadtxt(config['AR_env_config']['user_band_trace_path']
                                                         + config['AR_env_config']['user_band_trace'] + '/band_trace_' + str(int(self.users_band_traces_id[i]))) * 1000)
        for i in range(self.max_user_num):
            self.Users.append(User(i, band_trace=self.users_band_traces[i]))

        

    def start(self):
        # 加入新用户
        if self.User_num < self.max_user_num:
            while self.sys_time == self.user_arrive_time[self.User_num]:
                self.Users[self.User_num].activate(self.sys_time)
                self.newest_arrive_user.append(self.User_num)
                self.User_num += 1
                if self.User_num == self.max_user_num:
                    break

        # 将每个用户各自缓存中已经到达基站的请求放入基站
        for user in self.Users:
            while len(user.cur_query_list):
                if user.cur_query_list[0].arrivalTime == self.sys_time:
                    self.BaseStation.Histogram.add_in_gram(user.cur_query_list.pop(0))
                    self.query_arrival_time_list.append([self.sys_time, user.id])
                else:
                    break

        # 将已超时的请求从直方图中取出
        if len(self.BaseStation.Histogram.query_bins) > 0:
            bin_ID_now = int(self.sys_time / self.BaseStation.Histogram.deltaT)
            for i in range(0, bin_ID_now - 5):
                while len(self.BaseStation.Histogram.query_bins[i]) > 0:
                    query_temp = self.BaseStation.Histogram.query_bins[i].pop(0)
                    query_temp.end_flag = True
                    query_temp.end_time = query_temp.geneTime + query_temp.hardddl
                    query_temp.QoS = 0
                    query_temp.User_QoS = 0
                    self.Qos_memory.append([query_temp.user_id, query_temp.QoS, query_temp.id, query_temp.geneTime, query_temp.end_time])
                    self.User_Qos_memory.append([query_temp.user_id, query_temp.QoS, query_temp.id, query_temp.geneTime, query_temp.end_time, query_temp.t_ep])

        # 更新用户的状态
        for i in range(len(self.Users)):
            if self.sys_time == self.Users[i].stop_time and self.Users[i].active_flag is True:
                self.Users[i].stop_flag = True
            if self.Users[i].stop_flag is True and self.Users[i].real_stop_flag is False: #and self.Users[i].last_query.end_flag is True:
                self.Users[i].real_stop_flag = True
                self.Users[i].deactivate()
                self.User_num -= 1

        # 更新环境截止状态
        self.stopflag = True
        if len(self.Users) > 0:
            for user in self.Users:
                if user.stop_flag is False:
                    self.stopflag = False

        # 更新环境真实截止状态
        self.realstopflag = True
        for user in self.Users:
            if user.real_stop_flag is False:
                self.realstopflag = False

    def run(self):
        step_flag = -1
        # 更新所有gpu的状态
        for gpu in self.BaseStation.gpu_cluster:
            gpu.update(self.sys_time)
        # 如果有空闲的gpu，并且该gpu还未进行决策，则在系统进行一次step操作
        for gpu in self.BaseStation.gpu_cluster:
            if gpu.busy_flag is False and gpu.batch_flag is False:
                step_flag = gpu.id

        for gpu in self.BaseStation.gpu_cluster:
            if gpu.batch_flag is True:
                if self.BaseStation.Histogram.compute() >= gpu.batch.real_size:

                    user_queries_process = self.batch_divide(gpu)

                    gpu.busy_flag = True
                    gpu.now_endTime = self.sys_time + int(gpu.batch.real_size /
                                                          gpu.batch.throughput / ObT)
                    gpu.batch_flag = False
                    batch_index = 0

                    while True:
                        for bin in self.BaseStation.Histogram.query_bins:
                            if len(bin) > 0:
                                for query in copy.copy(bin):
                                    for i in range(0, self.max_user_num):
                                        if query.user_id == i and user_queries_process[i] > 0:
                                            gpu.batch.batch_query.append(query)
                                            bin.remove(query)
                                            user_queries_process[i] -= 1
                                            batch_index += 1
                                    if batch_index == gpu.batch.real_size:
                                        break
                                if batch_index == gpu.batch.real_size:
                                    break
                            else:
                                continue
                        if batch_index == gpu.batch.real_size:
                            break
                    for query in gpu.batch.batch_query:
                        query.end_time = gpu.now_endTime
                        query.model = gpu.batch.model_type
                        query.end_flag = True
                        query.qos_compute()
                        query.user_qos_compute()
                        self.Qos_memory.append([query.user_id, query.QoS, query.id, query.geneTime, query.end_time])
                        self.User_Qos_memory.append([query.user_id, query.User_QoS, query.id, query.geneTime, query.end_time, query.t_ep])

                if 0 < self.BaseStation.Histogram.compute() < gpu.batch.real_size and self.stopflag is True:
                    gpu.busy_flag = True
                    gpu.batch.size_log = int(np.log2(self.BaseStation.Histogram.compute())) + 1
                    gpu.batch.real_size = pow(2, gpu.batch.size_log)
                    gpu.batch.throughput = gpu.batch.throughputList[gpu.batch.gpu_type][gpu.batch.model_type][
                        gpu.batch.size_log]
                    gpu.now_endTime = self.sys_time + int(gpu.batch.real_size /
                                                          gpu.batch.throughput / ObT)
                    gpu.batch_flag = False
                    batch_index = 0
                    remain_quries = self.BaseStation.Histogram.compute()

                    while True:
                        for bin in self.BaseStation.Histogram.query_bins:
                            if len(bin) > 0:
                                while len(bin) > 0:
                                    batch_index += 1
                                    gpu.batch.batch_query.append(bin.pop(0))
                                    if batch_index == remain_quries:
                                        break
                                if batch_index == remain_quries:
                                    break
                            else:
                                continue
                        if batch_index == remain_quries:
                            break
                    for query in gpu.batch.batch_query:
                        query.end_time = gpu.now_endTime
                        query.model = gpu.batch.model_type
                        query.end_flag = True
                        query.qos_compute()
                        query.user_qos_compute()
                        self.Qos_memory.append([query.user_id, query.QoS, query.id, query.geneTime, query.end_time])
                        self.User_Qos_memory.append([query.user_id, query.User_QoS, query.id, query.geneTime, query.end_time, query.t_ep])
        if step_flag >= 0:
            if self.bs_alg_name == 'new_heur_inter':
                self.choose_action_NewHeuristic(step_flag, 0.4, 0.8)

    def choose_action_NewHeuristic(self, gpu_id, a=0.5, k=0.5):
        param = []
        nt = self.BaseStation.Histogram.compute()  # 计算当前直方图中的请求数量
        rt = self.estimate_arrival_rate(2000)

        B_tiny = 0
        B_full = 0
        G = 1
        throughput = np.array(self.throughput)
        throughput = throughput * self.gpu_process_up
        while True:
            if B_tiny <= 6:
                x = (a * (pow(2, B_tiny) / throughput[G][0][B_tiny]) * rt + (1 - a) * nt) * k * 1.0
                if pow(2, B_tiny) >= x:
                    break
                else:
                    B_tiny += 1
            else:
                B_tiny = 6
                break
        while True:
            if B_full <= 6:
                x = (a * (pow(2, B_full) / throughput[G][1][B_full]) * rt + (1 - a) * nt) * k * 1.0
                if pow(2, B_full) >= x:
                    break
                else:
                    B_full += 1
            else:
                B_full = 6
                break
        if nt > 0:
            for bin in self.BaseStation.Histogram.query_bins:
                if len(bin) > 0:
                    gene_time = bin[0].geneTime
                    break
            end_time_full = self.sys_time + int(pow(2, B_full) / throughput[G][1][B_full] / ObT)
            end_time_tiny = self.sys_time + int(pow(2, B_tiny) / throughput[G][0][B_tiny] / ObT)
            if end_time_full - gene_time < 1000:
                QoS_full = 1
            elif end_time_full - gene_time > 2000:
                QoS_full = 0
            else:
                QoS_full = 1 - (end_time_full - gene_time - 1000) / (2000 - 1000)

            if end_time_tiny - gene_time < 1000:
                QoS_tiny = 0.6
            elif end_time_tiny - gene_time > 2000:
                QoS_tiny = 0
            else:
                QoS_tiny = (1 - (end_time_tiny - gene_time - 1000) / (2000 - 1000)) * 0.6

            if QoS_full > QoS_tiny:
                param.append(B_full)
                param.append(1)
            else:
                param.append(B_tiny)
                param.append(0)
        else:
            param.append(B_full)
            param.append(1)
        self.BaseStation.gpu_cluster[gpu_id].batch_flag = True
        self.BaseStation.gpu_cluster[gpu_id].batch = Batch(size_log=param[0], model=param[1], gpu_type=1, throughputlist=self.throughput, gpu_process_up=self.gpu_process_up)
        self.Batches.append(param + [self.sys_time])

    def estimate_arrival_rate(self, delta_t):
        query_num = 0
        if 0 < self.sys_time < delta_t:
            query_num = len(self.query_arrival_time_list)
        elif self.sys_time >= delta_t:
            while True:
                if self.query_arrival_time_list[-1-query_num][0] > self.sys_time - delta_t:
                    query_num += 1
                else:
                    break
                if query_num == len(self.query_arrival_time_list):
                    break
        return query_num / delta_t * 1000

    def get_state(self):
        user_qos = np.zeros(self.max_user_num)
        for qos in self.User_Qos_memory:
            if np.max([time for time in self.real_stage_time if time <= self.sys_time]) <= qos[3] <= self.sys_time:
                user_qos[qos[0]] += qos[1]
        for i in range(len(user_qos)):
            user_qos[i] /= self.user_active_time * 60 * ObT
        user_esti_band = []  # estimated user bandwidth
        for user in self.Users:
            user_esti_band.append((user.estimate_band() - self.min_band) / (self.max_band - self.min_band)) if user.estimate_band() > 0 else user_esti_band.append(0)  # normalize
        user_esti_band = np.array(user_esti_band)

        users_obs = self.get_obs().numpy().flatten()

        state = np.append(user_qos, user_esti_band)
        state = np.append(state, users_obs)
        state = th.from_numpy(state)
        return state

    def get_obs(self):
        user_obs = np.zeros([self.max_user_num, self.n_obs])
        for i in range(self.max_user_num):
            user_obs[i][0] = self.Users[i].get_obs(self.sys_time, delta_t=1000)[0] / 2000
            user_obs[i][1] = (self.Users[i].get_obs(self.sys_time, delta_t=1000)[1] - 2.4) / (240 - 2.4) if (self.Users[i].get_obs(self.sys_time, delta_t=1000))[1] != 0 else 0
            user_obs[i][2] = self.Users[i].get_obs(self.sys_time, delta_t=1000)[2] / 2000
            user_obs[i][3] = (self.Users[i].get_obs(self.sys_time, delta_t=1000)[3] - 2.4) / (240 - 2.4) if (self.Users[i].get_obs(self.sys_time, delta_t=1000))[3] != 0 else 0
            user_obs[i][4] = self.Users[i].get_obs(self.sys_time, delta_t=1000)[4] / 60
            user_obs[i][5] = self.Users[i].get_obs(self.sys_time, delta_t=1000)[5] / 60
            action = self.Users[i].get_obs(self.sys_time, delta_t=1000)[6]
            action = th.from_numpy(np.array(action))
            action_one_hot = one_hot(self.n_action, action).numpy()
            user_obs[i][6: 6 + self.n_action] = action_one_hot
            user_obs[i][6 + self.n_action] = (self.Users[i].get_obs(self.sys_time, delta_t=1000)[7] - self.min_band) / (self.max_band - self.min_band) if (self.Users[i].get_obs(self.sys_time, delta_t=1000))[7] != 0 else 0
            user_obs[i][6 + self.n_action + 1] = (self.Users[i].get_obs(self.sys_time, delta_t=1000)[8] - self.min_band) / (self.max_band - self.min_band) if (self.Users[i].get_obs(self.sys_time, delta_t=1000))[8] != 0 else 0
        user_obs = th.from_numpy(user_obs)
        return user_obs ############



    def get_avail_actions(self):
        user_avail_actions = np.zeros([self.max_user_num, self.n_action])
        for i in range(self.max_user_num):
            user_avail_actions[i] = np.append(np.ones(1), np.zeros(self.n_action-1))
        for i in range(self.max_user_num):
            if self.Users[i].stop_flag is False and self.Users[i].active_flag is True:
                user_avail_actions[i] = np.append(np.zeros(1), np.ones(self.n_action-1))
        user_avail_actions = th.from_numpy(user_avail_actions)
        return user_avail_actions

    def get_all_reward(self):
        users_t_qos = np.zeros([self.max_t_ep, self.max_user_num])
        for qos_info in self.User_Qos_memory:
            if qos_info[5] >= 0:
                users_t_qos[qos_info[5]][qos_info[0]] += qos_info[1]
        r = np.zeros(self.max_t_ep)
        for i in range(self.max_t_ep):
            r1 = np.sum(users_t_qos[i]) if not self.normalize_r_flag else np.sum(users_t_qos[i]) / np.sum(self.user_active_flag[i])
            r2 = np.std(users_t_qos[i]) if len(users_t_qos[i]) > 1 else 0 #############此地方做了修改
            r[i] = r1 - self.penalty * r2
        return r
    
    def get_all_reward_at_step(self, t_ep):
        """
        返回某个时间步 t_ep 的全局 reward(根据 get_all_reward() 计算)
        """
        all_reward = self.get_all_reward()  # 获取整个 episode 的 reward 序列
        return all_reward[t_ep]

    # 获取指定 step 产生请求的每个用户reward
    # 独立训练的 RL 无法得到别的用户的请求信息, 只追求自身qos更大
    def get_tp_reward(self, t_ep):
        users_t_qos = np.zeros([self.max_t_ep, self.max_user_num])
        for qos_info in self.User_Qos_memory:
            if qos_info[5] >= 0:
                users_t_qos[qos_info[5]][qos_info[0]] += qos_info[1]
        r = np.zeros(self.max_user_num)
        for i in range(self.max_user_num):
            r[i] = users_t_qos[t_ep][i]
        return r

    def get_current_reward(self, t_ep):
        """
        实时计算当前时间步的reward
        """
        # 获取当前时间步各用户的QoS
        users_current_qos = np.zeros(self.max_user_num)
        
        # 只计算在当前时间步结束的请求
        for qos_info in self.User_Qos_memory:
            if qos_info[5] == t_ep:  # 只取当前时间步的QoS记录
                users_current_qos[qos_info[0]] += qos_info[1]
        
        # 计算公平性reward
        r1 = np.sum(users_current_qos) if not self.normalize_r_flag else np.sum(users_current_qos) / np.sum(self.user_active_flag[t_ep])    
        r2 = np.std(users_current_qos) if len(users_current_qos) > 1 else 0
        
        current_reward = r1 - self.penalty * r2
        return current_reward

    # pre_r
    def get_tp_pre_add_r(self, t_ep, id):
        # 获取当前时间步各用户的QoS
        users_current_qos = np.zeros(self.max_user_num)
        r_pre = np.zeros(self.max_user_num)
        
        # 只计算在当前时间步结束的请求
        for qos_info in self.User_Qos_memory:
            if qos_info[5] == t_ep:  # 只取当前时间步的QoS记录
                users_current_qos[qos_info[0]] += qos_info[1]
        r = np.zeros(self.max_user_num)
        for i in range(self.max_user_num):
            r[i] = users_current_qos[i]
        
        for i in range(self.max_user_num):
            r2 = np.std(r[0:i+1]) if len(r[0:i+1]) > 1 else 0
            r_pre[i] = sum(r[0:i+1])/(i+1) - self.penalty * r2  

        return r_pre[id]


    def batch_divide(self, gpu):
        user_queries_num = self.BaseStation.Histogram.compute_for_users(self.max_user_num)

        # batch分配此处使用轮取
        sorted_user = np.argsort(user_queries_num)
        batch_remain_num = gpu.batch.real_size
        user_queries_process_new = np.zeros(self.max_user_num)
        for k in np.arange(self.max_user_num):
            if k != self.max_user_num - 1:
                user_queries_process_new[sorted_user[k]] = min(user_queries_num[sorted_user[k]], math.ceil(batch_remain_num / (self.max_user_num - k)))
                batch_remain_num -= user_queries_process_new[sorted_user[k]]
            else:
                user_queries_process_new[sorted_user[k]] = batch_remain_num
        user_queries_process = user_queries_process_new

        return user_queries_process
    
