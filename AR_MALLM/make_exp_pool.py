import glob
import json
import os
from plm_special.data.exp_pool import ExperiencePool
import pickle

# 1. 匹配目录下所有 training_data_ep*.json
json_files = sorted(glob.glob('training_data*.json'))
if not json_files:
    raise FileNotFoundError('当前目录没有 training_data*.json')

# 2. 创建合并后的 ExperiencePool
pool = ExperiencePool()

# 修改后的处理逻辑
for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'读取 {file_path}，共 {len(data)} 条经验')
    
    for step_dict in data:
        
        pool.add(
            agent_id = step_dict['agent_id'], ########
            pre_r = step_dict['pre_r'],  #########
            state = step_dict['obs'],
            action = step_dict['agent_action'],
            reward = step_dict['reward'], 
            agent_active = step_dict['agent_active']########
        )

# 4. 保存合并结果
out_path = 'AR_exp_pool_coma_90_0.5std_dta_2000.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(pool, f)

print(f'共 {len(pool)} 条经验，已保存到 {out_path}')