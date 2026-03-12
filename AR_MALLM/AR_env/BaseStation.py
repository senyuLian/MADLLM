import numpy as np


class Histogram(object):
    def __init__(self):
        self.query_bins = []
        self.deltaT = 200

    def add_in_gram(self, query):
        binID = int((query.geneTime + query.softddl) / self.deltaT)
        if len(self.query_bins) < binID + 1:
            for i in range(len(self.query_bins), binID + 1):
                self.query_bins.append([])
        self.query_bins[binID].append(query)

    def compute(self):
        num = 0
        for temp in self.query_bins:
            if len(temp):
                num += len(temp)
        return num

    # 计算直方图中每个用户的请求数量
    def compute_for_users(self, user_num):
        nums = np.zeros(user_num)
        for temp in self.query_bins:
            if len(temp):
                for query in temp:
                    nums[query.user_id] += 1
        return nums

    # # 计算 delta_t 之前到现在直方图中请求的个数
    # # 输入：预设观察时间长度，当前系统时间，观察直方图所属用户的产生时间
    # def estimate_arrival_rate(self, delta_t, sys_time, user_gene_time):
    #     iteration = -1  # 从直方图中最新的bin开始计数
    #     counter = 0  # 计数器
    #     counter_flag = True  # 结束探索的标志
    #     if sys_time - user_gene_time >= delta_t:
    #         observe_time = delta_t
    #     else:
    #         observe_time = sys_time - user_gene_time
    #     while counter_flag:
    #         if self.compute() == 0:
    #             break
    #         if len(self.query_bins[iteration]):
    #             for query in self.query_bins[iteration]:
    #                 if sys_time - observe_time < query.arrivalTime < sys_time:
    #                     counter += 1
    #                 if query.arrivalTime < sys_time - observe_time:
    #                     counter_flag = False  # 一旦在该bin内发现有超出探索时间的请求，就可以停止探索了
    #         iteration -= 1
    #         if iteration < -len(self.query_bins):
    #             counter_flag = False
    #         if counter_flag is False:
    #             break
    #     if observe_time > 0:
    #         return counter / observe_time * 1000
    #     else:
    #         return 0


class Batch(object):
    def __init__(self, size_log, model, gpu_type, throughputlist, gpu_process_up):
        self.throughputList = np.array(throughputlist)
        self.throughputList = self.throughputList * gpu_process_up
        self.size_log = size_log  # 0 - 6
        self.real_size = pow(2, self.size_log)
        self.model_type = model
        self.gpu_type = gpu_type
        self.throughput = self.throughputList[self.gpu_type][self.model_type][self.size_log]
        self.batch_query = []


class GPU(object):
    def __init__(self, id):
        self.id = id
        self.busy_flag = False  # gpu是否在忙的标记
        self.batch_flag = False  # gpu是否决定下个batch参数的标记
        self.now_endTime = 0  # 处理当前正在处理的Batch的剩余时间
        self.batch = None

    def update(self, cur_time):
        if cur_time == self.now_endTime and self.busy_flag is True:
            self.busy_flag = False
            self.now_endTime = 0


class BaseStation(object):
    def __init__(self):
        self.gpu_num = 1
        self.gpu_cluster = []
        self.Histogram = Histogram()
        for i in range(0, self.gpu_num):
            self.gpu_cluster.append(GPU(i))
