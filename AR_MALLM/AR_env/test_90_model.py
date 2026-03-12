import torch as th
from episode_runner_rl import EpisodeRunner
from agents_controller import BasicMAC
from rw_config import get_config
from assist_func import save_variable
import os
import pandas as pd
import pickle
####
import json
####
config = get_config('config')


if not os.path.exists(config['Ex_config']['res_path']):
    os.mkdir(config['Ex_config']['res_path'])


qos_list = []
std_list = []
all_stage_std_sum_list = []


for i in range(0, config['Ex_config']['test_num']):

    th.manual_seed(config['Ex_config']['random_seed'])
    if os.path.isfile(config['Ex_config']['model_path'] + 'agent-' + str(i * config['Ex_config']['test_step']) + '.pkl'):
        agent_nn = th.load(config['Ex_config']['model_path'] + 'agent-' + str(i * config['Ex_config']['test_step']) + '.pkl', map_location='cpu', weights_only=False )

    else:
        print(f"Model not found: {config['Ex_config']['model_path']}, breaking at i={i}")
        break
    runner = EpisodeRunner(config)
    test_mac = BasicMAC(test_mode=True, agent_nn=agent_nn, config=config)
    # test_mac.cuda()
    runner.setup(test_mac)

    batch, qos, std, all_std, sum_reward, _ = runner.run(test_mode=True)

    # 保存当前episode的数据
    runner.save_training_data(f'test_data_ep{i}.json')

    # 打印最终统计信息
    print(f"Test run {i+1} completed:")
    print(f"  Total QoS: {qos}")
    print(f"  Total STD: {std}")
    print(f"  Stage STD sum: {all_std}")
    print(f"  Sum reward: {sum_reward}")


    qos_list.append(qos)
    all_stage_std_sum_list.append(all_std)
    std_list.append(std)

    with open(config['Ex_config']['res_path'] + 'qos_list', 'a') as qos_file:
        qos_file.write(str(qos) + '\n')
    with open(config['Ex_config']['res_path'] + 'all_std_list', 'a') as qos_file:
        qos_file.write(str(all_std) + '\n')
    with open(config['Ex_config']['res_path'] + 'sr_list', 'a') as sr_file:
        sr_file.write(str(sum_reward) + '\n')
    with open(config['Ex_config']['res_path'] + 'std_list', 'a') as std_file:
        std_file.write(str(std) + '\n')

    

    print(str(i * config['Ex_config']['test_step']) + ' :the test sum of qos = ' + str(qos))
    print(str(i * config['Ex_config']['test_step']) + ' :the std sum of qos = ' + str(std))
    print(str(i * config['Ex_config']['test_step']) + ' :total reward = ' + str(sum_reward))
    print(str(i * config['Ex_config']['test_step']) + ' :stage_sum_std = ' + str(all_std))

