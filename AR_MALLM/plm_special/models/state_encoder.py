import torch.nn as nn

class UserObsEncoder(nn.Module):
    def __init__(self, n_obs=24, n_action=15, embed_dim=128):
        super().__init__()
        self.n_obs = n_obs
        self.n_action = n_action
        self.embed_dim = embed_dim

        # 编码网络
        self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # obs1 当前窗口请求的平均处理时间
        self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # obs2 当前窗口请求的平均上传时间
        self.fc3 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # obs3 上一窗口请求的请求的平均处理时间
        self.fc4 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # obs4 上一窗口请求的请求的平均上传时间
        self.fc5 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # obs5 上一窗口所有请求的累计用户Qos
        self.fc6 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # obs6 当前窗口所有请求的累计用户Qos
        self.fc_action = nn.Sequential(nn.Linear(n_action, embed_dim), nn.LeakyReLU())  # 上一时刻动作one-hot
        self.fc_band1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # 当前估计的上行带宽
        self.fc_band2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # 上一时刻的上行带宽

    def forward(self, user_obs):
        # user_obs.shape: (batch_size, max_user_num, n_obs)
        batch_size, seq_len = user_obs.shape[0], user_obs.shape[1]
        user_obs = user_obs.reshape(batch_size * seq_len, self.n_obs) #（1，24）

        # 拆分字段
        obs1 = user_obs[..., 0:1]  
        obs2 = user_obs[..., 1:2]  
        obs3 = user_obs[..., 2:3]  
        obs4 = user_obs[..., 3:4]  
        obs5 = user_obs[..., 4:5]  
        obs6 = user_obs[..., 5:6]  
        action = user_obs[..., 6:6+self.n_action]  
        band1 = user_obs[..., 6+self.n_action:6+self.n_action+1] 
        band2 = user_obs[..., 6+self.n_action+1:6+self.n_action+2] 

        # 编码各字段
        features1 = self.fc1(obs1).reshape(batch_size, seq_len, -1) #[batch_size, max_user_num, embed_dim]
        features2 = self.fc2(obs2).reshape(batch_size, seq_len, -1)
        features3 = self.fc3(obs3).reshape(batch_size, seq_len, -1)
        features4 = self.fc4(obs4).reshape(batch_size, seq_len, -1)
        features5 = self.fc5(obs5).reshape(batch_size, seq_len, -1)
        features6 = self.fc6(obs6).reshape(batch_size, seq_len, -1)
        features_action = self.fc_action(action).reshape(batch_size, seq_len, -1)
        features_band1 = self.fc_band1(band1).reshape(batch_size, seq_len, -1)
        features_band2 = self.fc_band2(band2).reshape(batch_size, seq_len, -1)

        # 返回所有编码特征（可用于拼接或进一步处理）
        return features1, features2, features3, features4, features5, features6, features_action, features_band1, features_band2
