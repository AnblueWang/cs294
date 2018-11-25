import model 
import tensorflow as tf 
import gym 
import numpy as np 
import matplotlib.pyplot as plt 
import data_utils
import pickle
import load_policy
import os 

BATCH_SIZE = 200
LEARNING_RATE = 0.001
HIDDEN_SIZE = 100
SKIP_STEP = 100
NUM_TRAIN = 20000//BATCH_SIZE*2
NUM_EPOCH = 20

def simulate(model,envname,num_rollouts=20,max_timesteps=None,render=False):
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        # print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.apply(obs.reshape(1,-1))
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    return np.mean(returns),np.std(returns),np.array(observations)

def simulateExpert(policy_fn,envname,num_rollouts=20,max_timesteps=None,render=False):
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        # print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    return np.mean(returns),np.std(returns)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_file', type=str)
    parser.add_argument('file_name', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    envname = args.envname
    file_name = args.file_name
    fss = file_name.split('/')
    dgfile_name = fss[0]+'/c'+fss[1]

    dgObs, dgActions = data_utils.read_data(dgfile_name)

    bc = model.behaviorCloning(file_name,envname,HIDDEN_SIZE,BATCH_SIZE,LEARNING_RATE,
        SKIP_STEP,True)
    bc.build_graph()
    bcAvgs = []
    bcStds = []

    dg = model.behaviorCloning(dgfile_name,envname,HIDDEN_SIZE,BATCH_SIZE,LEARNING_RATE,
        SKIP_STEP,True)
    dg.build_graph()
    dgAvgs = []
    dgStds = []

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_file)
    epAvgs = []
    epStds = []
    print('loaded and built')

    for i in range(NUM_EPOCH):
        print(i)
        bc.train(NUM_TRAIN*(i+1))
        r,s,_ = simulate(bc,envname,args.num_rollouts,args.max_timesteps,args.render)
        bcAvgs.append(r)
        bcStds.append(s)
        
        dg.X,dg.Y = data_utils.read_data(dgfile_name)
        dg.train(NUM_TRAIN*(i+1))
        print('Dagger data length:',dg.X.shape[0])
        dr,ds,do = simulate(dg,envname,args.num_rollouts,args.max_timesteps,args.render)
        dgAvgs.append(dr)
        dgStds.append(ds)
        dgObs = np.concatenate((dgObs,do),axis=0)
        print("length of new observations:",len(do))
        dgA = np.array(list(map(lambda x: policy_fn(x[None,:]),do)))
        print(dgA.shape)
        dgActions = dgActions.reshape(-1,dgA.shape[1],dgA.shape[2])
        dgActions = np.concatenate((dgActions,dgA),axis=0)
        storeData(dgObs,dgActions,dgfile_name)

        er,es = simulateExpert(policy_fn,envname,args.num_rollouts,args.max_timesteps,args.render)
        epAvgs.append(er)
        epStds.append(es)

    plotCurve(envname,bcAvgs,bcStds,dgAvgs,dgStds,epAvgs,epStds)

def storeData(observations,actions,file):
    expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

    with open(file, 'wb') as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

def plotCurve(envname,bcAvg,bcStd,dgAvg,dgStd,epAvg,epStd):
    plt.figure(envname+'-s3')
    plt.errorbar(range(1,1+NUM_EPOCH),bcAvg,yerr=bcStd,fmt='-o')
    plt.errorbar(range(1,1+NUM_EPOCH),dgAvg,yerr=dgStd,fmt='-o')
    plt.errorbar(range(1,1+NUM_EPOCH),epAvg,yerr=epStd,fmt='-o')
    plt.legend(['Behavior Cloning', 'Dagger', 'Expert'],
           loc='upper left',
           numpoints=1,
           fancybox=True)
    plt.xlabel('num_iteration')
    plt.ylabel('average reward')
    plt.savefig(envname+'-s3')

if __name__ == '__main__':
    main()