import os
class Config:
    _base_dir = '' if 'AR_MALLM' in os.getcwd() else 'AR_MALLM/'

    artifacts_dir = _base_dir + 'artifacts/'
    results_dir = artifacts_dir + 'results/'
    exp_pools_dir = artifacts_dir + 'exp_pools/'
    # 默认的经验池路径
    default_exp_pool_path = exp_pools_dir + 'AR_exp_pool_coma_90_0std_dta_2000.pkl'

    # --- 训练超参数 (Training Hyper-parameters) ---
    lr = 1e-4
    weight_decay = 1e-4
    warmup_steps = 2000
    num_epochs = 80
    grad_accum_steps = 32
    seed = 100003
    eval_per_epoch = 1
    
    # --- 强化学习配置 (RL Config) ---
    gamma = 1.0           # 折扣因子
    scale = 1000          # 奖励缩放
    context_window = 20   # 即原代码中的 'w'
    state_feature_dim = 256
    target_return_scale = 1.0
    penalty = 1.0         # 环境惩罚项系数
    sample_step = None       # 经验采样步长###

    # plm special
    plm_types = ['gpt2', 'llama', 'llava', 't5-lm', 'opt', 'mistral']
    plm_sizes = ['xxs', 'xs', 'small', 'base', 'large', 'xl', 'xxl']  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.
    plm_dir = _base_dir + ('../../downloaded_plms' if 'AR_NetLLM' in _base_dir else '../downloaded_plms')
    plm_ft_dir = _base_dir + 'data/ft_plms'
    plm_embed_sizes = {
        'gpt2': {
            'base': 1024,
            'small': 768,
            'large': 1280,
            'xl': 1600,
        },
        'llama': {
            'base': 4096,
        },
        't5-lm': {
            'base': 768,
            'small': 512,
            'large': 4096,
            'xl': 2048,
        },
        'llava': {
            'base': 4096,
        },
        'mistral': {
            'base': 4096,
        },
        'opt': {
            'large': 5120,
            'base': 4096,
            'small': 2560,
            'xs': 2048,
            'xxs': 512,
        },
    }
    plm_layer_sizes = {
        'gpt2': {
            'base': 24,
            'small': 12,
            'large': 36,
            'xl': 48
        },
        'llama': {
            'base': 32,
        },
        't5-lm': { 
            'base': 12,
            'small': 6,
            'large': 24,
            'xl': 24
        },
        'llava': {
            'base': 32,
        },
        'mistral': {
            'base': 32,
        },
        'opt': {
            'large': 40,
            'base': 32,
            'small': 32,
            'xs': 32,
            'xxs': 16,
        },
    }


cfg = Config()
