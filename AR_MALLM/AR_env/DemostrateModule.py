import matplotlib.pyplot as plt
import numpy as np
import os
from rw_config import get_config
from User import ObT


class DemostrateModule:
    def __init__(self):
        self.config = get_config('config')
        self.res_path = self.config['Ex_config']['res_path']
        self.fig_path = self.config['Ex_config']['fig_path']

        self.font_label = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 20,
                      }
        self.font_legend = {'family': 'Times New Roman',
                       'weight': 'normal',
                       'size': 16,
                       }

        self.sub = [321, 322, 323, 324, 325]
        self.sub_point = ['r.', 'g.', 'b.', 'y.', 'k.']
        if not os.path.exists(self.fig_path):
            os.mkdir(self.fig_path)

    def TrainEvaluate(self):
        sum = np.loadtxt(self.res_path + 'qos_list')
        std = np.loadtxt(self.res_path + 'std_list')

        reward = np.loadtxt(self.res_path + 'sr_list')

        step = self.config['Ex_config']['test_step']
        penalty = self.config['AR_env_config']['penalty']
        num = self.config['Ex_config']['test_num']

        y = sum - penalty * std
        best_x = np.argmax(y)
        best_sum = sum[best_x]
        best_std = std[best_x]
        show_max_sum = '[' + str(best_x * self.config['Ex_config']['test_step']) + ' ' + str(best_sum) + ']'
        show_max_std = '[' + str(best_x * self.config['Ex_config']['test_step']) + ' ' + str(best_std) + ']'
        baseline_sum = 3526.08 * np.ones(len(sum))
        baseline_std = 209.67 * np.ones(len(sum))
        # baseline_sum = 3075.55 * np.ones(len(sum))
        # baseline_std = 110.92 * np.ones(len(sum))

        # baseline_sum = 4240.30 * np.ones(len(sum))
        # baseline_std = 66.18 * np.ones(len(sum))

        max_user_num = self.config['AR_env_config']['max_user_num']
        figsize = (14, 10)
        labelsize = 15
        fig = plt.figure(figsize=figsize)
        # fig.suptitle(self.config['Ex_config']['version_id'])
        plt.title(self.config['Ex_config']['version_id'], fontdict=self.font_label)
        episodes = np.arange(0, len(sum) * step, step)

        plt.xlabel('Episode', fontdict=self.font_label)
        plt.tick_params(labelsize=labelsize)
        ax1 = fig.add_subplot(111)
        labels = ax1.get_xticklabels() + ax1.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax1.set_ylabel('Sum of QoE', fontdict=self.font_label)
        ax1.set_ylim(np.min([0.98 * np.min(sum), 0.98*baseline_sum[0]]), np.max([1.02 * np.max(sum), 1.02*baseline_sum[0]]))
        ax1.plot(best_x * step, best_sum, 'r*')
        plt.annotate(show_max_sum, xytext=(best_x * step, best_sum), xy=(best_x * step, best_sum))
        l1, = ax1.plot(episodes, sum, 'b', lw=2, ms=10)
        l1b, = ax1.plot(episodes, baseline_sum, 'b', ls='--', lw=2, ms=10)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Std of QoE for ' + str(max_user_num) + 'users', fontdict=self.font_label)
        ax2.set_ylim(np.min([0.98 * np.min(std), 0.98 * baseline_std[0]]), np.max([1.02 * np.max(std), 1.02 * baseline_std[0]]))
        plt.tick_params(labelsize=labelsize)
        labels = ax2.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax2.plot(best_x * step, best_std, 'r*')
        plt.annotate(show_max_std, xytext=(best_x * step, best_std), xy=(best_x * step, best_std))
        l2, = ax2.plot(episodes, std, 'g', lw=2, ms=10)
        l2b, = ax2.plot(episodes, baseline_std, 'g', ls='--', lw=2, ms=10)

        plt.legend(handles=[l1, l1b, l2, l2b], labels=['sum', 'base_sum', 'std', 'base_std'], loc='best', prop=self.font_legend)
        plt.savefig(self.fig_path + 'train_eval.png')

        figsize = (14, 10)
        labelsize = 15
        fig = plt.figure(figsize=figsize)
        plt.title(self.config['Ex_config']['version_id'], fontdict=self.font_label)
        episodes = np.arange(0, len(sum) * step, step)
        # print(episodes)
        plt.xlabel('Episode', fontdict=self.font_label)
        plt.tick_params(labelsize=labelsize)

        plt.ylim(0.95 * np.min(reward) if np.min(reward) > 0 else 1.05 * np.min(reward), 1.05 * np.max(reward) if np.max(reward) > 0 else 0)
        plt.plot(episodes, reward, 'y', lw=2, ms=10)
        plt.savefig(self.fig_path + 'reward_eval.png')

    # 用于绘制同时开始的QoS
    def MeanwhileStartTrace(self):
        qos = np.loadtxt(self.res_path + 'user_qos')
        max_user_num = self.config['AR_env_config']['max_user_num']
        user_t = []
        user_y = []
        for i in range(0, max_user_num):
            user_t.append([])
            user_y.append([])

        for i in range(0, len(qos)):
            for j in range(0, max_user_num):
                if qos[i][0] == j:
                    user_t[j].append(qos[i][3])
                    user_y[j].append(qos[i][1])

        for i in range(max_user_num):
            print(np.sum(user_y[i]))

        fig = plt.figure(figsize=[15, 10])
        fig.suptitle('user qos with systime')
        for i in range(max_user_num):
            ax = fig.add_subplot(self.sub[i])
            ax.set_ylim([0, 1.1])
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            ax.plot(user_t[i], user_y[i], self.sub_point[i])
            ax.set_ylabel('user qos')
            ax.set_xlabel('systime/sec')
        plt.savefig(self.fig_path + 'user_qos')

    # 绘制同时开始的帧率
    def MeanwhileFrame(self):
        max_user_num = self.config['AR_env_config']['max_user_num']
        frame = []
        dense = []
        for i in range(max_user_num):
            print(np.mean(15 * pow(2, np.loadtxt(self.res_path + 'frame_' + str(i))[:, 0])))
            frame.append(np.loadtxt(self.res_path + 'frame_' + str(i)))
            dense.append(np.loadtxt(self.res_path + 'dense_' + str(i)))

        for i in range(max_user_num):
            print(np.mean(0.2 * (np.loadtxt(self.res_path + 'dense_' + str(i))[:, 0] + 1)))

        fig = plt.figure(figsize=[10, 9])
        fig.suptitle('user frame with systime')

        for i in range(max_user_num):
            ax = fig.add_subplot(self.sub[i])
            ax.set_ylim([0, 61])
            ax.set_ylabel('frame')
            ax.set_xlabel('systime/sec')
            ax.plot(frame[i][:, 1], 2 ** frame[i][:, 0] * 15, self.sub_point[i])

        plt.savefig(self.fig_path + 'frame')

        fig = plt.figure(figsize=[10, 9])
        fig.suptitle('user dense with systime')

        for i in range(max_user_num):
            ax = fig.add_subplot(self.sub[i])
            ax.set_ylim([0, 1.1])
            ax.set_ylabel('dense')
            ax.set_xlabel('systime/sec')
            ax.plot(dense[i][:, 1], (dense[i][:, 0] + 1) * 0.2, self.sub_point[i])

        plt.savefig(self.fig_path + 'dense')

    def self_train_eval(self):
        self_sum = np.loadtxt(self.res_path + 'a_qos_list')
        step = self.config['Ex_config']['test_step']
        figsize = (14, 10)
        labelsize = 15
        fig = plt.figure(figsize=figsize)
        plt.title(self.config['Ex_config']['version_id'], fontdict=self.font_label)
        episodes = np.arange(0, len(self_sum) * step, step)

        plt.xlabel('Episode', fontdict=self.font_label)
        plt.tick_params(labelsize=labelsize)
        ax1 = fig.add_subplot(111)
        labels = ax1.get_xticklabels() + ax1.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        ax1.set_ylabel('QoE of single RL Agent', fontdict=self.font_label)

        ax1.set_ylim(0.98 * np.min(self_sum), 1.02 * np.max(self_sum))
        l1, = ax1.plot(episodes, self_sum, 'b', lw=2, ms=10)

        plt.savefig(self.fig_path + 'self_train_eval.png')

    def eval_multi_stage_performance(self):
        qos = np.loadtxt(self.res_path + 'user_qos')
        max_user_num = self.config['AR_env_config']['max_user_num']
        stage_time = self.config['AR_env_config']['user_arrive_time']
        user_active_time = self.config['AR_env_config']['user_active_time'] / ObT  # 每位用户活跃的时间, 单位ObT

        real_stage_time = np.append(np.array(stage_time), np.array(stage_time) + np.ones(max_user_num) * user_active_time)

        user_t = []
        user_y = []
        for i in range(0, max_user_num):
            user_t.append([])
            user_y.append([])

        for i in range(0, len(qos)):
            for j in range(0, max_user_num):
                if qos[i][0] == j:
                    user_t[j].append(qos[i][3])
                    user_y[j].append(qos[i][1])

        user_sum = np.zeros(max_user_num)
        for i in range(max_user_num):
            user_sum[i] = np.sum(user_y[i])
        stages = []
        for i in range(len(real_stage_time) - 3):
            stages.append([real_stage_time[i + 1], real_stage_time[i + 2]])

        stages_index = [[0, 2], [0, 3], [0, 4], [0, 5],
                        [1, 5], [2, 5], [3, 5]]

        user_sum4stages = np.zeros([len(stages), max_user_num])
        for i in range(max_user_num):
            for j in range(len(user_t[i])):
                for k in range(len(stages)):
                    if stages[k][0] <= user_t[i][j] < stages[k][1]:
                        user_sum4stages[k][i] += user_y[i][j]
        sum4stages = np.zeros(len(stages))
        std4stages = np.zeros(len(stages))
        for k in range(len(stages)):
            sum4stages[k] = np.sum(user_sum4stages[k])
            std4stages[k] = np.std(user_sum4stages[k][stages_index[k][0]:stages_index[k][1]])
        print('sum:')
        for x in sum4stages:
            print(x)
        print('std:')
        for x in std4stages:
            print(x)


DM = DemostrateModule()
DM.TrainEvaluate()
# DM.MeanwhileStartTrace()
# DM.MeanwhileFrame()
# DM.eval_multi_stage_performance()
