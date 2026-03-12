import os
import random
import numpy as np
from baseline_special.utils.constants import BITRATE_LEVELS
try:  # baselines use a different conda environment without torch, so we need to skip ModuleNotFoundError when runing baselines
    import torch
except ModuleNotFoundError:
    pass


def process_batch(batch, device='cpu'):
    """
    Process batch of data.
    """
    agent_ids, pre_rs, states, actions, returns, timesteps = batch
    
    # 确保states是张量并移动到设备
    if isinstance(states, (list, tuple)):
        # 如果states是列表，先转换为numpy数组再转为张量
        states = torch.as_tensor(np.array(states), dtype=torch.float32).to(device)
    else:
        # 如果已经是数组/张量，直接转换
        states = torch.as_tensor(states, dtype=torch.float32).to(device)
    
    # 添加批次维度（如果不存在）
    if states.dim() == 2:  # (max_length, 24)
        states = states.unsqueeze(0)  # -> (1, max_length, 24)
    
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(1, -1)
    # states = torch.cat(states, dim=0).unsqueeze(0).float().to(device)
    labels = actions.long()
    actions = ((actions + 1) / BITRATE_LEVELS).unsqueeze(2)          #存疑
    agent_ids = torch.as_tensor(agent_ids, dtype=torch.float32, device=device).reshape(1, -1, 1)
    pre_rs = torch.as_tensor(pre_rs, dtype=torch.float32, device=device).reshape(1, -1, 1)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device).reshape(1, -1, 1)
    timesteps = torch.as_tensor(timesteps, dtype=torch.int32, device=device).unsqueeze(0)
    

    return agent_ids, agent_ids, states, actions, returns, timesteps, labels

# def process_batch(batch, device='cpu'):

#     states, actions, returns, timesteps = batch

#     # 转换为张量
#     states_tensor = torch.tensor([states[0]], dtype=torch.float32).unsqueeze(1).to(device)  # [1, 1, 24]
#     actions_tensor = torch.tensor([actions[0]], dtype=torch.float32).to(device)
#     returns_tensor = torch.tensor([returns[0]], dtype=torch.float32).unsqueeze(-1).to(device)
#     timesteps_tensor = torch.tensor([timesteps[0]], dtype=torch.int32).to(device)
#     print(states_tensor.shape)

#     return states_tensor, actions_tensor, returns_tensor, timesteps_tensor, actions_tensor.long()


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)



def calc_mean_reward(result_files, test_dir, str, skip_first_reward=True):
    matching = [s for s in result_files if str in s]
    reward = []
    count = 0
    for log_file in matching:
        count += 1
        first_line = True
        with open(test_dir + '/' + log_file, 'r') as f:
            for line in f:
                parse = line.split()
                if len(parse) <= 1:
                    break
                if first_line:
                    first_line = False
                    if skip_first_reward:
                        continue
                reward.append(float(parse[7]))
    print(count)
    return np.mean(reward)


def clear_dir(directory):
    file_list = os.listdir(directory)
    for file in file_list:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)