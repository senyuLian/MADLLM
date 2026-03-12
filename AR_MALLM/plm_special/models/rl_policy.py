import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
    

INF = 1e5


class OfflineRLPolicy(nn.Module):
    def __init__(
            self,
            state_feature_dim,
            bitrate_levels,
            state_encoder,
            plm,
            plm_embed_size,
            max_length=None,
            max_ep_len=49, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out=None,
            residual=False, 
            which_layer=-1,  # for early stopping: specify which layer to stop
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device
        

        self.num_agents = 5  #################################
        self.max_ep_len = max_ep_len
        self.bitrate_levels = bitrate_levels
        self.max_length = max_length

        self.plm = plm
        self.plm_embed_size = plm_embed_size

        # =========== multimodal encoder (start) ===========
        self.state_encoder = state_encoder  # Now this is UserObsEncoder
        self.state_feature_dim = state_feature_dim
        
        # Initialize embeddings for different observation features
        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_agent_id = nn.Linear(1, plm_embed_size).to(device)
        self.embed_pre_r = nn.Linear(1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)

        # 修改嵌入层定义，将输入维度从1改为state_feature_dim
        self.embed_delay = nn.Linear(state_feature_dim, plm_embed_size).to(device)       # obs1 
        self.embed_throughput = nn.Linear(state_feature_dim, plm_embed_size).to(device)  # obs2 
        self.embed_loss = nn.Linear(state_feature_dim, plm_embed_size).to(device)        # obs3 
        self.embed_jitter = nn.Linear(state_feature_dim, plm_embed_size).to(device)      # obs4 
        self.embed_buffer = nn.Linear(state_feature_dim, plm_embed_size).to(device)      # obs5 
        self.embed_playtime = nn.Linear(state_feature_dim, plm_embed_size).to(device)    # obs6 
        self.embed_action_feat = nn.Linear(state_feature_dim, plm_embed_size).to(device) # 动作特征
        self.embed_band1 = nn.Linear(state_feature_dim, plm_embed_size).to(device)      # 带宽1
        self.embed_band2 = nn.Linear(state_feature_dim, plm_embed_size).to(device)       # 带宽2

        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device) #层归一化
        # =========== multimodal encoder (end) ===========
    
        self.action_head = nn.Linear(plm_embed_size, bitrate_levels).to(device)

        self.device = device
        self.device_out = device_out

        # the following are used for evaluation ###########################
        self.states_dq_list = [deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length) for _ in range(self.num_agents)]
        self.agent_ids_dq_list = [deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length) for _ in range(self.num_agents)]
        self.pre_rs_dq_list = [deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length) for _ in range(self.num_agents)]
        self.returns_dq_list = [deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length) for _ in range(self.num_agents)]
        self.actions_dq_list = [deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length) for _ in range(self.num_agents)]

        self.residual = residual
        self.which_layer = which_layer

        self.modules_except_plm = nn.ModuleList([
            self.embed_agent_id, self.embed_pre_r, self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, 
            self.embed_ln, self.embed_delay, self.embed_throughput, self.embed_loss,
            self.embed_jitter, self.embed_buffer, self.embed_playtime, self.embed_action_feat,
            self.embed_band1, self.embed_band2, self.action_head
        ])

    def forward(self, agent_ids, pre_rs, states, actions, returns, timesteps, attention_mask=None):


        assert actions.shape[0] == 1, 'batch size should be 1 to avoid CUDA memory exceed'

        # Step 1: process actions, returns and timesteps
        agent_ids = agent_ids.to(self.device)
        pre_rs = pre_rs.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        timesteps = timesteps.to(self.device)

        agent_id_embeddings = self.embed_agent_id(agent_ids)
        pre_r_embeddings = self.embed_pre_r(pre_rs)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns)
        time_embeddings = self.embed_timestep(timesteps).squeeze()

        agent_id_embeddings = agent_id_embeddings + time_embeddings
        pre_r_embeddings = pre_r_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Step 2: process states using UserObsEncoder
        states = states.to(self.device)
        # state_encoder returns 9 features (features1-6, action_feat, band1, band2)
        state_features = self.state_encoder(states)
        
        # Embed each feature separately and add time embeddings
        delay_emb = self.embed_delay(state_features[0]) + time_embeddings
        throughput_emb = self.embed_throughput(state_features[1]) + time_embeddings
        loss_emb = self.embed_loss(state_features[2]) + time_embeddings
        jitter_emb = self.embed_jitter(state_features[3]) + time_embeddings
        buffer_emb = self.embed_buffer(state_features[4]) + time_embeddings
        playtime_emb = self.embed_playtime(state_features[5]) + time_embeddings
        action_feat_emb = self.embed_action_feat(state_features[6]) + time_embeddings
        band1_emb = self.embed_band1(state_features[7]) + time_embeddings
        band2_emb = self.embed_band2(state_features[8]) + time_embeddings
        
        

        # Step 3: stack returns, state features, actions embeddings
        stacked_inputs = []
        action_embed_positions = np.zeros(returns_embeddings.shape[1]) #序列长度
        
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((
                returns_embeddings[0, i:i+1],
                agent_id_embeddings[0, i:i+1],
                pre_r_embeddings[0, i:i+1],
                delay_emb[0, i:i+1],
                throughput_emb[0, i:i+1],
                loss_emb[0, i:i+1],
                jitter_emb[0, i:i+1],
                buffer_emb[0, i:i+1],
                playtime_emb[0, i:i+1],
                action_feat_emb[0, i:i+1],
                band1_emb[0, i:i+1],
                band2_emb[0, i:i+1],
                action_embeddings[0, i:i+1]
            ), dim=0)
            stacked_inputs.append(stacked_input)
            action_embed_positions[i] = (i + 1) * (4 + 9)  # 1 return + 9 state features
        
        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0) # 合并多个时间步
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :] # 截断到plm_embed_size此处
        stacked_inputs_ln = self.embed_ln(stacked_inputs) # 归一化均值为0，方差为1

        # Step 4: feed to PLM
        if attention_mask is None:
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]),  #批大小和序列长度
                                      dtype=torch.long, device=self.device)

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln

        # Step 5: predict actions
        logits_used = logits[:, action_embed_positions - 2]  # -1 because we have 1 return + 9 features # 提取动作位置的隐藏状态
        action_pred = self.action_head(logits_used)

        return action_pred

    def sample(self, agent_id, pre_r, state, target_return, timestep, **kwargs): #############################

        # 确保agent_id在有效范围内
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"agent_id must be between 0 and {self.num_agents-1}, got {agent_id}")

        # 获取指定参与者的历史队列
        returns_dq = self.returns_dq_list[agent_id]
        agent_ids_dq = self.agent_ids_dq_list[agent_id]
        pre_rs_dq = self.pre_rs_dq_list[agent_id]
        states_dq = self.states_dq_list[agent_id]
        actions_dq = self.actions_dq_list[agent_id]

        # Step 1: stack previous state, action, return features
        prev_stacked_inputs = []
        for i in range(len(states_dq)):
            prev_return_embeddings = returns_dq[i]
            prev_agent_id_embeddings = agent_ids_dq[i]
            prev_pre_r_embeddings = pre_rs_dq[i]
            prev_state_embeddings = states_dq[i]
            prev_action_embeddings = actions_dq[i]
            prev_stacked_inputs.append(torch.cat((prev_return_embeddings, prev_agent_id_embeddings, prev_pre_r_embeddings, prev_state_embeddings, prev_action_embeddings), dim=1))
        prev_stacked_inputs = torch.cat(prev_stacked_inputs, dim=1)

        # Step 2: process target return and timesteps
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)

        return_embeddings = self.embed_return(target_return)
        time_embeddings = self.embed_timestep(timestep)
        return_embeddings = return_embeddings + time_embeddings

        # Step 3: process state using UserObsEncoder
        state = state.to(self.device)
        state_features = self.state_encoder(state)
        
            # Embed each feature separately and add time embeddings
        agent_id_tensor = torch.as_tensor([agent_id], dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        agent_id_embeddings = self.embed_agent_id(agent_id_tensor) + time_embeddings
        pre_r_tensor = torch.as_tensor([pre_r], dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        pre_r_embeddings = self.embed_pre_r(pre_r_tensor) + time_embeddings

        delay_emb = self.embed_delay(state_features[0]) + time_embeddings
        throughput_emb = self.embed_throughput(state_features[1]) + time_embeddings
        loss_emb = self.embed_loss(state_features[2]) + time_embeddings
        jitter_emb = self.embed_jitter(state_features[3]) + time_embeddings
        buffer_emb = self.embed_buffer(state_features[4]) + time_embeddings
        playtime_emb = self.embed_playtime(state_features[5]) + time_embeddings
        action_feat_emb = self.embed_action_feat(state_features[6]) + time_embeddings
        band1_emb = self.embed_band1(state_features[7]) + time_embeddings
        band2_emb = self.embed_band2(state_features[8]) + time_embeddings
            # Concatenate all state embeddings
        state_embeddings = torch.cat([
            delay_emb, throughput_emb, loss_emb, jitter_emb, 
            buffer_emb, playtime_emb, action_feat_emb, 
            band1_emb, band2_emb
        ], dim=1)

        # Step 4: stack return, state and previous embeddings
        stacked_inputs = torch.cat((return_embeddings, agent_id_embeddings, pre_r_embeddings, state_embeddings), dim=1)
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]
        stacked_inputs_ln = self.embed_ln(stacked_inputs)

        attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), 
                                dtype=torch.long, device=self.device)

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln

        # Step 5: predict the bitrate for next chunk
        logits_used = logits[:, -1:]
        action_pred = self.action_head(logits_used)
        action_pred = action_pred.reshape(-1)
        bitrate, _ = self._sample(action_pred)

        # compute action embeddings 
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = (bitrate + 1) / self.bitrate_levels
        action_embeddings = self.embed_action(action_tensor) + time_embeddings
        
        # update deques
        self.returns_dq_list[agent_id].append(return_embeddings)
        self.agent_ids_dq_list[agent_id].append(agent_id_embeddings)
        self.pre_rs_dq_list[agent_id].append(pre_r_embeddings)
        self.states_dq_list[agent_id].append(state_embeddings) 
        self.actions_dq_list[agent_id].append(action_embeddings)

        return bitrate
    
    def clear_dq(self):
        self.agent_ids_dq.clear()
        self.pre_rs_dq.clear()
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()

        self.agent_ids_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.pre_rs_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))

    def _sample(self, logits):
        pi = F.softmax(logits.detach(), 0).cpu().numpy() ############
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob