import copy
import os

import gym
import gym_minigrid
from gym_minigrid import wrappers

import numpy as np
import pickle
import torch

from imitation.rewards.discrim_nets import ActObsMLP
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import bc

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy


from utils.env_utils import minigrid_get_env
from cnn_modules.cnn_discriminator import ActObsCNN

import argparse

import matplotlib.pyplot as plt


import argparse

CORPUS = sorted(['get', 'to', 'the', 'green', 'goal', 'square', 'blue'])
WORD_TO_INDEX = {w : i for i,w in enumerate(CORPUS)}
INDEX_TO_WORD = dict(zip(WORD_TO_INDEX.values(), WORD_TO_INDEX.keys()))

def sentence2onehot(sentence):
    one_hot_vectors = np.zeros((len(sentence), len(CORPUS)), dtype=np.float32)
    for i,word in enumerate(sentence):
        index = WORD_TO_INDEX[word]
        one_hot_vectors[i][index] = 1.0
    return one_hot_vectors

sentence = ['get', 'to', 'the', 'green', 'goal', 'square']
one_hot_vectors = sentence2onehot(sentence)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    "-e",
    help="minigrid gym environment to train on",
    default="MiniGrid-Empty-6x6-v0",
)
parser.add_argument("--run", "-r", help="Run name", default="sample_run")

parser.add_argument(
    "--save-name", "-s", help="BC weights save name", default="saved_testing"
)

parser.add_argument("--traj-name", "-t", help="Run name", default="saved_testing")


parser.add_argument(
    "--seed", type=int, help="random seed to generate the environment with", default=1
)

parser.add_argument(
    "--nepochs", type=int, help="number of epochs to train till", default=50
)

parser.add_argument(
    "--flat",
    "-f",
    default=False,
    help="Partially Observable FlatObs or Fully Observable Image ",
    action="store_true",
)

parser.add_argument(
    "--vis-trained",
    default=False,
    help="Render 10 traj of trained BC",
    action="store_true",
)

args = parser.parse_args()
# args.env = "MiniGrid-Empty-8x8-v0"

save_path = "./logs/" + args.env + "/gail/" + args.run + "/"
os.makedirs(save_path, exist_ok=True)
traj_dataset_path = "./traj_datasets/" + args.traj_name + ".pkl"

print(f"Expert Dataset: {args.traj_name}")
with open(traj_dataset_path, "rb") as f:
    trajectories = pickle.load(f)

_env = gym.make(args.env)
mission_statement = _env.mission
mission_statement_one_hot = sentence2onehot(mission_statement.split(' '))
for i,traj in enumerate(trajectories):
    num_observations = traj.acts.shape[0]
    # L, H = mission_statement_one_hot.shape[0], mission_statement_one_hot.shape[1]
    # traj.mission_one_hot = mission_statement_one_hot.reshape(1, L, H).repeat(num_observations, axis=0)
    # traj.mission_str = mission_statement
    traj.mission = [mission_statement] * num_observations

transitions = rollout.flatten_trajectories(trajectories)


train_env = minigrid_get_env(args.env, 1, args.flat)
assert len(train_env.venv.envs) == 1
mission_statement = train_env.venv.envs[0].mission
if args.flat:
    discrim_type = ActObsMLP(
        action_space=train_env.action_space,
        observation_space=train_env.observation_space,
        hid_sizes=(32, 32),
    )
    policy_type = ActorCriticPolicy
else:
    discrim_type = ActObsCNN(
        action_space=train_env.action_space,
        observation_space=train_env.observation_space,
    )
    policy_type = ActorCriticCnnPolicy

base_ppo = PPO(policy_type, train_env, verbose=1, batch_size=64, n_steps=50)

logger.configure(save_path)

gail_trainer = adversarial.GAIL(
    train_env,
    expert_data=transitions,
    expert_batch_size=64,
    gen_algo=base_ppo,
    n_disc_updates_per_round=1,
    normalize_reward=False,
    normalize_obs=False,
    disc_opt_kwargs = {"lr":0.0001},
    discrim_kwargs={"discrim_net": discrim_type},
)

total_timesteps = 100000
i = 0
gail_trainer.train(total_timesteps=total_timesteps)
gail_trainer.gen_algo.save("gens/gail_gen_"+str(i))

with open('discrims/gail_discrim'+str(i)+'.pkl', 'wb') as handle:
    pickle.dump(gail_trainer.discrim, handle, protocol=pickle.HIGHEST_PROTOCOL)


if args.vis_trained:
    for traj in range(10):
        obs = train_env.reset()
        train_env.render()
        for i in range(20):
            action, _ = gail_trainer.gen_algo.predict(obs, deterministic=True)
            obs, reward, done, info = train_env.step(action)
            train_env.render()
            if done:
                break
        print("done")
