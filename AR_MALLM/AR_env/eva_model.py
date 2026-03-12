import torch as th
from episode_runner_rl import EpisodeRunner
from agents_controller import BasicMAC
from rw_config import get_config
from assist_func import save_variable
import os
import numpy as np

config = get_config('config')

if not os.path.exists(config['Ex_config']['res_path']):
    os.mkdir(config['Ex_config']['res_path'])

np.random.seed(0)
th.manual_seed(config['Ex_config']['random_seed'])
agent_nn = th.load(config['Ex_config']['model_path'] + config['Ex_config']['model_name'], map_location='cpu', weights_only=False )
runner = EpisodeRunner(config)
test_mac = BasicMAC(test_mode=True, agent_nn=agent_nn, config=config)
runner.setup(test_mac)

batch, qos, std, _, _, ct_lst = runner.run(test_mode=True)

save_variable(runner.env.Batches, config['Ex_config']['res_path'] + 'batch')
save_variable(runner.env.User_Qos_memory, config['Ex_config']['res_path'] + 'user_qos')
for i in range(runner.env.max_user_num):
    save_variable(batch['obs'].numpy()[0][:, 0], config['Ex_config']['res_path'] + 'obs_' + str(i))
    save_variable(runner.env.Users[i].frame_rate_memory, config['Ex_config']['res_path'] + 'frame_' + str(i))
    save_variable(runner.env.Users[i].dense_rate_memory, config['Ex_config']['res_path'] + 'dense_' + str(i))

save_variable(ct_lst, 'ct_lst_test')
print('the test total qos = ' + str(qos))
print('the std of users qos = ' + str(std))
