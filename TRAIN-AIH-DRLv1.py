import random
import pandas as pd
import numpy as np
import collections
import subprocess
import pickle
import logging
import copy
from collections import namedtuple, deque

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def run_simulator(type_):
    #Environtment
    env = BlockchainEnvironment()

    agent_dict = {
        "propose": Agent(state_size=4, action_size=2, random_seed=10),
        "fix_blocksize": Agent(state_size=4, action_size=1, random_seed=10),
        "fix_interval": Agent(state_size=4, action_size=1, random_seed=10)
    }

    logger.setLevel(logging.WARNING)

    prefix = "train"
    def ddpg(agent, n_episodes=2000, step_type='propose'):
        scores_deque = deque(maxlen=100)
        scores = []
        infos = []
        state = env.reset()
        max_score = -np.Inf
        for i_episode in range(1, n_episodes+1):
            action = agent.act(state)
            score = 0
            next_state, reward, done, _info = env.step(action, prefix, step_type, i_episode)
            agent.step(state, action, reward, next_state, done)
            infos.append(_info)
            state = next_state
            score = reward
            if done:
                break 
            scores_deque.append(score)
            scores.append(score)
            if i_episode % 10 == 0:
                print('\rEpisode {}\tAverage Score: {:.7f}\tScore: {:.7f}'.format(i_episode, np.mean(scores_deque), score), end="")
            if i_episode % 50 == 0:
                torch.save(agent.actor_local.state_dict(), "AIH_"+prefix+"_"+step_type + '_checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), "AIH_"+prefix+"_"+step_type + '_checkpoint_critic.pth')
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
        return scores, infos


    step_types = ['propose','fix_interval', 'fix_blocksize']
    for step_type in step_types:
        scores, infos = ddpg(agent_dict[step_type],step_type=step_type)

        with open("AIH_"+prefix+"_"+step_type+"_scores.pckl", "wb") as f:
            pickle.dump(scores, f)

        with open("AIH_"+prefix+"_"+step_type+"_infos.pckl", "wb") as f:
            pickle.dump(infos, f)


def main(argv):
  _Strategy = ''
  try:
    opts, args = getopt.getopt(argv,"h:s:",["strategy="])
  except getopt.GetoptError:
    print ('try python main.py -s <simulator type>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
        print ('python main.py -s <simulator type>')
        sys.exit()
    elif opt in ("-s", "--strategy"):
        _Strategy = arg
  _Director = ScrapingDirector(_Strategy)

if __name__ == '__main__':
    main(sys.argv[1:])
