import numpy as np
import pickle


def read_data(filename):
    f = open(filename,'rb')
    data = pickle.load(f)
    obs = data['observations']
    actions = data['actions']
    actions = actions.reshape(-1,actions.shape[2])
    indices = np.random.permutation(obs.shape[0])
    obs = obs[indices[:],:]
    actions = actions[indices[:],:]
    return obs, actions
