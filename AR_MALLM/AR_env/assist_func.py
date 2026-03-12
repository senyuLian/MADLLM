import numpy as np
import torch as th

# 计算数组的非空元素
def compute(a):
    x = 0
    for temp in a:
        if temp != []:
            x += len(temp)
    return x


# 对环境状态进行初始化
def reset(dim):
    s_init = []
    for i in range(0, dim):
        s_init.append(0)
    return s_init


# 对数据进行标准化
def normalize(s):
    s_normal = s
    s_max = np.max(s_normal)
    s_min = np.min(s_normal)
    if s_max - s_min > 0:
        for w in range(0, len(s)):
            s[w] = (s[w] - s_min) / (s_max - s_min)


def save_variable(var, filename):
    var = np.array(var)  # 将输入的数组变量转换为Numpy数组
    np.savetxt(filename, var)


def d3_scale(dat, out_range=(-1, 1)):
    domain = [np.min(dat, axis=0), np.max(dat, axis=0)]

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b =  1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat))


def one_hot(out_dim, tensor):
    y_onehot = tensor.new(*tensor.shape[:-1], out_dim).zero_()
    y_onehot.scatter_(-1, tensor.long(), 1)
    return y_onehot.float()


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]
