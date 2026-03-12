from rw_config import get_config
import torch as th
from AR_env.assist_func import one_hot

class Episode_Batch(dict):
    def __init__(self):
        dict.__init__(self)
        self.index = 0
        config = self.load_config()
        self.batch_size = config['RL_config']['batch_size']
        self.max_seq_length = config['RL_config']['max_seq_length'] + 1
        self.config = config
        self._set_up(self.config)

        
        

    @staticmethod
    def load_config():
        config = get_config('config')
        return config

    def insert(self, key, value):
        self[key] = value

    def _set_up(self, config):
        self.insert('state', th.zeros(self.batch_size, self.max_seq_length, config['env_config']['n_state']))
        self.insert('obs', th.zeros(self.batch_size, self.max_seq_length, config['env_config']['n_agent'], config['agent_config']['n_obs']))
        self.insert('actions', th.zeros(self.batch_size, self.max_seq_length, config['env_config']['n_agent'], 1))
        self.insert('avail_actions', th.zeros(self.batch_size, self.max_seq_length, config['env_config']['n_agent'], config['agent_config']['n_action']))
        self.insert('reward', th.zeros(self.batch_size, self.max_seq_length, 1))
        # self.insert('independent_reward', th.zeros(self.batch_size, self.max_seq_length, config['env_config']['n_agent']))
        self.insert('terminated', th.zeros(self.batch_size, self.max_seq_length, 1))
        self.insert('actions_onehot', th.zeros(self.batch_size, self.max_seq_length, config['env_config']['n_agent'], config['agent_config']['n_action']))
        self.insert('filled', th.zeros(self.batch_size, self.max_seq_length, 1))

    # def update_batch(self, key, value, t):
    #     if key == 'actions':
    #         value = value.view([self.config['env_config']['n_agent'], 1])
    #         self[key][0][t] = value
    #         for i in range(len(value)):
    #             self['actions_onehot'][0][t][i] = one_hot(self.config['agent_config']['n_action'], value[i])
    #         self['filled'][0][t] = 1
    #     else:
    #         self[key][0][t] = value
    def update_batch(self, key, value, t):
        # 确保value是张量
        if not isinstance(value, th.Tensor):
            value = th.tensor(value, dtype=torch.float32)

        # 获取正确的形状配置
        n_agents = self.config['env_config']['n_agent']

        # 确保value有正确的形状
        if value.dim() == 1:
            # 如果是一维，重塑为 [n_agents, 1]
            value = value.view(n_agents, 1)
        elif value.dim() == 2:
            # 如果是二维，确保第二维是1
            if value.size(1) != 1:
                value = value.view(-1, 1)

        # 存储数据
        if t >= len(self.data[key]):
            # 扩展数据列表
            self.data[key].extend([None] * (t - len(self.data[key]) + 1))
        self.data[key][t] = value

    # def train_preprocess(self):
