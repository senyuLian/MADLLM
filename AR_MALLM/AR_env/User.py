import numpy as np
import torch as th
from itertools import chain
ObT = 1e-3  # 系统最小观测时间1ms
update_time = int(1 / ObT)  # 用户更新参数时间为1s


class Query(object):
    def __init__(self, user_id, f, d, gene_time, arrival_time, size, query_id):

        self.Qd = {0.2: 0.32, 0.4: 0.77, 0.6: 0.90, 0.8: 0.96, 1: 1}  # QoS与图像压缩率的函数，低分辨率惩罚因子
        self.Qm = {0: 0.6, 1: 1}  # 低精度模型惩罚因子

        self.id = query_id
        self.user_id = user_id
        self.frame_rate = f
        self.denseness_rate = d
        self.geneTime = gene_time
        self.arrivalTime = arrival_time
        self.size = size
        self.softddl = 1000
        self.hardddl = 2000

        self.model = -1
        self.QoS = -1
        self.end_time = -1
        self.end_flag = False
        self.drop_rate = -1
        self.User_QoS = -1

        self.t_ep = -1

#论文 3.2 QoE计算公式
    def qos_compute(self):
        if self.end_flag is True:
            if self.end_time - self.geneTime <= self.softddl:
                self.QoS = self.Qd[self.denseness_rate] * self.Qm[self.model]
            elif self.end_time - self.geneTime > self.hardddl:
                self.QoS = 0
            else:
                self.QoS = self.Qd[self.denseness_rate] * self.Qm[self.model] * \
                           (1 - (self.end_time - self.geneTime - self.softddl) / (self.hardddl - self.softddl))

#针对用户角度Qos计算公式，每一项都多乘一项丢包率
    def user_qos_compute(self):
        if self.end_time - self.geneTime <= self.softddl:
            self.User_QoS = self.Qd[self.denseness_rate] * self.Qm[self.model] * self.drop_rate
        elif self.end_time - self.geneTime > self.hardddl:
            self.User_QoS = 0
        else:
            self.User_QoS = self.Qd[self.denseness_rate] * self.Qm[self.model] * \
                       (1 - (self.end_time - self.geneTime - self.softddl) / (self.hardddl - self.softddl)) * self.drop_rate



class User(object):
    def __init__(self, user_id, frame_rate=30, denseness_rate=1, band_trace=[]):
        self.id = user_id

        self.stop_flag = False  # 用户到达active time后，停止产生请求，等待请求全部处理完成
        self.real_stop_flag = False  # 用户的所有请求被处理完毕
        self.active_flag = False  # 表示用户的

        self.stop_time = -1  # 初始化用户Inactive时间为-1
        self.update_times = 0
        
        self.frame_rate = frame_rate
        self.denseness_rate = denseness_rate

        self.upload_band_trace = band_trace
        self.upload_band = self.upload_band_trace[0]  # 上行带宽，单位kbps
        self.cur_query_list = []  # 当前update time内用户的请求列表
        self.cur_drop_list = []  # 当前update time内用户丢弃的请求列表
        self.drop_flag = False  # 用户前一个update time内的请求是否丢弃的flag
        self.query_list = []  # 当前用户产生的query列表，列表中套列表，元素是每10s产生的query列表

        self.effective_query_list = []  # 每次发出请求时，真正被发出的请求列表
        self.query_num = 0  # 用户产生的请求数量总数

        self.FR_table = [15, 30, 60]  # 用户可选的帧率
        self.D_table = [0.2, 0.4, 0.6, 0.8, 1]  # 用户可选的压缩率
        self.FR_gear = 0  # 帧率档位范围 0 - 2，初始化为1
        self.D_gear = 0  # 压缩率档位范围 0 - 4，初始化为4

        self.frame_rate = frame_rate
        self.denseness_rate = denseness_rate

        self.dense_rate_memory = []
        self.frame_rate_memory = []
        self.frame_drop_num_memory = []

        self.delta_t = 200  # 用户动态行为档位调整参数
        self.softddl = 1000
        self.hardddl = 2000

    def activate(self, cur_time):
        # 激活时第一次生成请求列表
        self.gene_time = cur_time
        self.generate_query(cur_time,t_ep=-1)
        self.active_flag = True

    def deactivate(self):
        self.active_flag = False

    def generate_query(self, cur_time, t_ep):
        self.cur_drop_list = []
        self.cur_query_list = []
        self.drop_flag = False
        last_arrive_time = 0
        for i in range(int(update_time * self.frame_rate * ObT)):
            # 计算大小（是压缩率的函数）单位KB
            size = 150 * self.denseness_rate
            # 计算生成时间
            gene_time = int(cur_time + update_time / ((update_time * ObT) * self.frame_rate) * i)
            # 计算到达时间上传到服务器的时间
            arrival_time = int(np.max([last_arrive_time, gene_time]) + size / (self.upload_band / 8 * ObT))
            last_arrive_time = arrival_time
            query = Query(self.id, self.frame_rate, self.denseness_rate, gene_time, arrival_time, size, self.query_num)
            self.query_num += 1
            if query.arrivalTime > cur_time + update_time:
                query.end_time = query.geneTime + query.hardddl
                query.end_flag = True
                query.QoS = 0
                self.drop_flag = True
                self.cur_drop_list.append(query)   # 若请求超过了更新时间，则丢弃该请求
            else:
                query.t_ep = t_ep
                self.cur_query_list.append(query)  # 否则存入待处理列表
        self.frame_drop_num_memory.append(len(self.cur_drop_list))
        for query in self.cur_query_list:
            query.drop_rate = 1 - len(self.cur_drop_list) / self.frame_rate
        self.query_list.append(self.cur_query_list[:] + self.cur_drop_list[:])
        self.effective_query_list.append(self.cur_query_list[:])  # 将有效请求加入有效请求列表中
        

    def update_parameters(self, cur_time, alg_name, action=None, t_ep=-1):
        
        if alg_name == 'qoe_based' and t_ep != -1:

            if len(self.query_list) >= 2:
                mean_process_time = 0
                mean_upload_time = 0

                observed_queries = list(chain(*self.effective_query_list[-3:])) if len(self.query_list) > 2 else list(chain(*self.effective_query_list))  # 待遍历的所有请求是最后三秒的
                observed_num = 0  # 符合时间约束的请求数量
                for query in observed_queries:
                    if cur_time - 2000 < query.end_time < cur_time:
                        mean_process_time += query.end_time - query.geneTime
                        mean_upload_time += query.arrivalTime - query.geneTime
                        observed_num += 1
                if observed_num == 0:
                    return  # 跳过本轮更新，避免除零错误

                mean_process_time = int(mean_process_time / observed_num) #上上秒内的平均处理时间
                mean_upload_time = int(mean_upload_time / observed_num)  #上上秒内的平均下载时间

                mean_base_time = mean_process_time - mean_upload_time  # 上上秒内所有请求的平均在基站等待+处理的时间
                kf = mean_base_time / self.frame_rate
                kd = mean_upload_time / self.denseness_rate

                query_softddl = 1000
                query_hardddl = 2000
                Qd = {0.2: 0.32, 0.4: 0.77, 0.6: 0.90, 0.8: 0.96, 1: 1}

                arg_max_FR_gear = 0
                arg_max_D_gear = 0
                max_Q = 0
                #指数加权平均估计下一秒的可用带宽
                estimate_band = self.upload_band_trace[self.update_times - 1] * 0.75 + self.upload_band_trace[self.update_times - 2] * 0.25  # 带宽估计

                for f_gear in range(len(self.FR_table)):
                    for d_gear in range(len(self.D_table)):
                        keep_num = 0
                        last_arrival_time = 0
                        for i in range(self.FR_table[f_gear]):
                            size = 150 * self.D_table[d_gear]
                            gene_time = int(0 + update_time / ((update_time * ObT) * self.FR_table[f_gear]) * i)  # 每一帧的产生时间（该秒的开始记为0）
                            arrival_time = int(np.max([last_arrival_time, gene_time]) + size / (estimate_band / 8 * ObT))
                            last_arrival_time = arrival_time
                            if arrival_time > update_time:
                                break
                            else:
                                keep_num += 1

                        t_process_estimate = kf * self.FR_table[f_gear] + kd * self.D_table[d_gear]
                        if t_process_estimate < query_softddl:
                            if max_Q < Qd[self.D_table[d_gear]] * self.FR_table[f_gear] * keep_num / self.FR_table[f_gear]:
                                max_Q = Qd[self.D_table[d_gear]] * self.FR_table[f_gear] * keep_num / self.FR_table[f_gear]
                                arg_max_D_gear = d_gear
                                arg_max_FR_gear = f_gear
                        elif query_softddl < t_process_estimate < query_hardddl:
                            if max_Q < self.FR_table[f_gear] * Qd[self.D_table[d_gear]] * (query_hardddl - t_process_estimate) / query_softddl * keep_num / self.FR_table[f_gear]:
                                max_Q = self.FR_table[f_gear] * Qd[self.D_table[d_gear]] * (query_hardddl - t_process_estimate) / query_softddl * keep_num / self.FR_table[f_gear]
                                arg_max_D_gear = d_gear
                                arg_max_FR_gear = f_gear
                self.D_gear = arg_max_D_gear
                self.FR_gear = arg_max_FR_gear

        elif alg_name == 'rl_directed':
            if action < 1:
                return
            if type(action) == th.Tensor:
                action = action.numpy()
            self.D_gear = int((action - 1) % 5)
            self.FR_gear = int((action - 1) / 5)

        # 根据平均处理时间调整帧率、压缩率
        self.frame_rate = self.FR_table[self.FR_gear]
        self.denseness_rate = self.D_table[self.D_gear]

        self.frame_rate_memory.append([self.FR_gear, cur_time])
        self.dense_rate_memory.append([self.D_gear, cur_time])

        self.upload_band = self.upload_band_trace[self.update_times]  # 依据trace更新带宽
        self.update_times += 1

        # 根据调整后的帧率，压缩率产生新的update_time的请求
        if alg_name == 'rl_directed':
            self.generate_query(cur_time, t_ep)
        else:
            self.generate_query(cur_time, t_ep=-1)

   #论文中带宽估计部分
    def estimate_band(self):
        if self.active_flag:
            if len(self.query_list) >= 3:
                estimate_band = self.upload_band_trace[self.update_times - 1] * 0.75 + self.upload_band_trace[self.update_times - 2] * 0.25
            elif len(self.query_list) == 2:
                estimate_band = self.upload_band_trace[self.update_times - 1] * 1
            else:
                estimate_band = self.upload_band_trace[0]
        else:
            estimate_band = 0
        return estimate_band

#把用户过去两个 2 秒时间窗内的关键性能指标和环境状态打包成 9 维向量，作为强化学习策略网络的输入，用于学习如何动态调整帧率和压缩率。
    def get_obs(self, cur_time, delta_t):
        if self.active_flag is True and cur_time - self.gene_time >= 2000:
            all_queries = list(chain(*self.effective_query_list))
            query_complete_time = []
            query_upload_time = []
            last_query_complete_time = []
            last_query_upload_time = []
            qos = []
            last_qos = []
            for i in range(len(all_queries)):
                if all_queries[-1-i].end_flag:
                    if cur_time - delta_t <= all_queries[-1-i].end_time <= cur_time:
                    
                        query_complete_time.append(all_queries[-1-i].end_time - all_queries[-1-i].geneTime)
                        query_upload_time.append(all_queries[-1-i].arrivalTime - all_queries[-1-i].geneTime)
                        qos.append(all_queries[-1-i].User_QoS)

                    elif cur_time - 2 * delta_t <= all_queries[-1-i].end_time <= cur_time - delta_t:
                        last_query_complete_time.append(all_queries[-1-i].end_time - all_queries[-1-i].geneTime)
                        last_query_upload_time.append(all_queries[-1-i].arrivalTime - all_queries[-1-i].geneTime)
                        last_qos.append(all_queries[-1-i].User_QoS)

                    elif all_queries[-1-i].end_time < cur_time - 2 * delta_t:
                        break
            action = self.frame_rate_memory[self.update_times - 2][0] * 5 + self.dense_rate_memory[self.update_times - 2][0] + 1
            band = self.upload_band_trace[self.update_times - 1]
            last_band = self.upload_band_trace[self.update_times - 2]

            if len(query_complete_time) > 0 and len(last_query_complete_time) > 0:
                return np.mean(query_complete_time), np.mean(query_upload_time), np.mean(last_query_complete_time), np.mean(last_query_upload_time), np.sum(qos), np.sum(last_qos), action, band, last_band
            elif len(query_complete_time) == 0 and len(last_query_complete_time) > 0:
                return 0, 0, np.mean(last_query_complete_time), np.mean(last_query_upload_time), np.sum(qos), np.sum(last_qos), action, band, last_band
            elif len(query_complete_time) > 0 and len(last_query_complete_time) == 0:
                return np.mean(query_complete_time), np.mean(query_upload_time), 0, 0, np.sum(qos), np.sum(last_qos), action, band, last_band
            else:
                return 0, 0, 0, 0, np.sum(qos), np.sum(last_qos), action, band, last_band
        else:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0
