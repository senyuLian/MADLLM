# write / read config file with json module
import json


def init_config_file():
    config = dict()

    # configuration for env
    config['env_config'] = dict({'n_state': 130,
                            'n_agent': 5}) 

    # configuration for agentÅ
    config['agent_config'] = dict({'n_action': 16,
                              'n_obs': 24,
                              'epsilon_start': .5,
                              'epsilon_finish': .0,
                              'epsilon_anneal_time': 90 * 5000})

    # configuration for RL
    config['RL_config'] = dict({'gamma': 0.9,
                           'batch_size': 1,
                           'buffer_size': 1,
                           'max_seq_length': 90, #评估时间步
                           'critic_lr': 0.0005,
                           'actor_lr': 0.00005,
                           'optim_alpha': 0.99,   # parameter for RMSProp
                           'optim_eps': 0.00001,  # parameter for RMSProp
                           'td_lambda': 0.8,
                           'grad_norm_clip': 5,
                           'target_update_interval': 10,
                           'device': 'cpu',  # cpu / cuda
                           'normalize_r_flag': True})

    # config['AR_env_config'] = dict({'max_user_num': 5,
    #                            'gpu_num': 1,
    #                            'user_arrive_time': [0, 0, 0, 0, 0],
    #                            'user_active_time': 60,  # unit: second
    #                            'sys_active_time': 60,  # unit: second
    #                            'user_band_trace': 'fix',
    #                            'fixed_band': [60000, 60000, 60000, 60000, 60000],  # unit: Kbps
    #                            'user_band_trace_path': 'user_band/',
    #                            'user_band_trace_group_id': 9,
    #                            'BS_alg': 'new_heur_inter',
    #                            'user_alg': 'rl_directed',  # rl_directed是使用COMA QoE-based是对比算法
    #                            'penalty': 0.0,
    #                            'max_band': 100000,
    #                            'min_band': 5000,
    #                            'init_actions': [1, 1, 1, 1, 1],
    #                            'throughput':
    #                             [[[25.6, 35.7, 51.9, 66.1, 73.7, 84.7, 91.2],
    #                               [9.4, 9.4, 12.3, 14.7, 16.0, 16.8, 0]],
    #                              [[23.8, 35.7, 63.5, 90.9, 109.6, 131.1, 141.9],
    #                               [14.5, 18.9, 24.2, 28.4, 31.5, 33.9, 34.6]]],
    #                             'gpu_process_up': 1})


    config['AR_env_config'] = dict({'max_user_num': 5,  
                               'gpu_num': 1,
                               'user_arrive_time':[0, 10000, 20000, 30000, 40000],
                               'user_active_time': 50,  # unit: second
                               'sys_active_time': 90,  # unit: second
                               'user_band_trace': 'fix',
                               'fixed_band': [60000, 60000, 60000, 60000, 60000],  # unit: Kbps
                               'user_band_trace_path': 'user_band/',
                               'user_band_trace_group_id': 9,
                               'BS_alg': 'new_heur_inter',
                               'user_alg': 'rl_directed',  # rl_directed是使用COMA QoE-based是对比算法
                               'penalty': 1,
                               'max_band': 100000,
                               'min_band': 5000,
                               'init_actions': [1, 1, 1, 1, 1],
                               'throughput':
                                [[[25.6, 35.7, 51.9, 66.1, 73.7, 84.7, 91.2],
                                  [9.4, 9.4, 12.3, 14.7, 16.0, 16.8, 0]],
                                 [[23.8, 35.7, 63.5, 90.9, 109.6, 131.1, 141.9],
                                  [14.5, 18.9, 24.2, 28.4, 31.5, 33.9, 34.6]]],
                                'gpu_process_up': 1})

    config['Ex_config'] = dict({'max_episode_num': 5000,
                           'version_id': 'mallm-90-0.5p',
                           'test_model_id': 4990,
                           'random_seed': 0,
                           'test_num': 250,
                           'test_step': 20
                           })

    config['A2C_config'] = dict({
        'device': 'cpu',
        'Optimizer': 'RMSprop',
        'lr': 1e-4,
        'update_interval': 5,
        'gamma': 0.99
    })

    config['ddpg_config'] = dict({
        'buffer_size': int(100000),
        'batch_size': 64,
        'gamma': 0.99,
        'TAU': 1e-3,
        'lr_actor': 1e-3,
        'lr_critic': 1e-4,
        'WEIGHT_DECAY': 0,
        'device_name': 'cuda:0'
    })

    config['dqn_config'] = dict({
        'buffer_size': int(100000),
        'batch_size': 64,
        'lr': 1e-4,
        'device': 'cpu'
    })

    config['Ex_config'].update({'model_path': './models/' + str(config['Ex_config']['version_id']) + '/'})
    config['Ex_config']['model_name'] = 'agent-' + str(config['Ex_config']['test_model_id']) + '.pkl'
    config['Ex_config']['res_path'] = './res/' + str(config['Ex_config']['version_id']) + '/'
    config['Ex_config']['fig_path'] = './figs/' + str(config['Ex_config']['version_id']) + '/'

    with open('config.json', 'w') as f:
        json.dump(config, f)


# read config content from specified json file
def get_config(filename):
    f = open(filename + '.json')
    ob = json.load(f)
    return ob


init_config_file()
