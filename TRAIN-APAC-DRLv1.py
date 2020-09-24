import random
import pandas as pd
import numpy as np
import collections
import subprocess
import logging
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

logger = logging.getLogger()
logger.setLevel(logging.INFO)


LIMIT_SEALED_BLOCK = 10995116277760 # 10 Gb
CONFIG_URL = """simulator/src/simulation.config"""
OUTPUT_URL = """simulator/src/dist/{}/"""
HEIGHT_PER_EPISODE = 100
CONS_ALGO = """SimBlock.node.consensusAlgo.ProofOfAuthority"""
NUM_OF_NODES=600
OUT_FOLDER = "PoA/APAC/{}/{}/episode-{}" # (prefix, type, episode)

MINIMUM_VALIDATOR = 34
NUM_OF_VALIDATOR = 100


# APAC configuration
AVG_TRANSACTION_SIZE = 12400
STD_TRANSACTION_SIZE = 1000
OPERATION_SPEED = 6854

# AIH configuration
#AVG_TRANSACTION_SIZE = 7900
#STD_TRANSACTION_SIZE = 1000
#OPERATION_SPEED = 5607

#Normalization Param
# Computing; data: (588375, 153844, 405276.1538461539, 409339.0)
MAX_COMPUTING = 600000
MIN_COMPUTING = 120000

# Block interval; data : //1000*60;//1000*30*5;//1000*60*10;//1000*60*60
MAX_BLOCKINTERVAL = 3600000 
MIN_BLOCKINTERVAL = 60000

# Blocksize; data : //6110;//8000;//535000;//0.5MB;//1MB
MAX_BLOCKSIZE = 10485760 
MIN_BLOCKSIZE = 6110


# Transmition Rate; data :(316, 5908, 539.6932715030989, 535.0)
MAX_TRANSMITIONRATE = 5000
MIN_TRANSMITIONRATE = 100


# Simulation Environment

class BlockchainEnvironment():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tot_reward = 0
        self.df_node = None
        self.df_block = None
        self.df_block_orphans = None
        self.sealed_block = 0
        self.final_time = 0
        self.pref_block_size = 0
        self.block_size = 0
        self.cstep = 0
        return np.array([0.5, 0.5, 0.5, 0.5])
        
    # Setup configuration
    def update_config(self, interval, blksize, output):
        with open("simulator/src/simulation.config", "w") as text_file:
            text_file.write("""interval={}
algo={}
num_nodes={}
blockheight={}
block_size={}
output={}
minimum_validator={}
num_of_validator={}
step={}
operation_speed={}""".format(interval, 
                             CONS_ALGO, 
                             NUM_OF_NODES, 
                             HEIGHT_PER_EPISODE, 
                             blksize, 
                             output, 
                             MINIMUM_VALIDATOR, 
                             NUM_OF_VALIDATOR, 
                             self.cstep,
                             OPERATION_SPEED))

    def run_simulator(self):
        cmd = "gradle run"
        # no block, it start a sub process.
        p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for line in p.stdout:
            logging.info(line)
        p.wait()
        return p.returncode
    
    def z_norm_trx(self, t):
        return abs(t - AVG_TRANSACTION_SIZE)/STD_TRANSACTION_SIZE
    
    def mm_norm_computing(self, v):
        max_ = MAX_COMPUTING 
        min_ = MIN_COMPUTING
        return (v - min_) / (max_ - min_)
    
    
    def mm_norm_blockinterval(self, v):
        max_ = MAX_BLOCKINTERVAL 
        min_ = MIN_BLOCKINTERVAL
        return (v - min_) / (max_ - min_)
    
    def mm_denorm_blockinterval(self, v):
        max_ = MAX_BLOCKINTERVAL 
        min_ = MIN_BLOCKINTERVAL
        return int(v * (max_ - min_) + min_);
    
    def mm_norm_blocksize(self, v):
        # data //6110;//8000;//535000;//0.5MB;//1MB
        max_ = MAX_BLOCKSIZE 
        min_ = MIN_BLOCKSIZE
        return (v - min_) / (max_ - min_)
    
    def mm_denorm_blocksize(self, v):
        # data //6110;//8000;//535000;//0.5MB;//1MB
        max_ = MAX_BLOCKSIZE 
        min_ = MIN_BLOCKSIZE
        return int(v * (max_ - min_) + min_);
    
    def mm_norm_transmitions_rate(self, v):
        # data (316, 5908, 539.6932715030989, 535.0)
        max_ = MAX_TRANSMITIONRATE
        min_ = MIN_TRANSMITIONRATE
        return (v - min_) / (max_ - min_)

    
    def generate_transaction(self):
        size = int(1048576/AVG_TRANSACTION_SIZE)+1
        array = np.random.normal(loc=AVG_TRANSACTION_SIZE, scale=STD_TRANSACTION_SIZE, size=size)
        return np.mean(array)
    
    def get_propagation(self):
        df_prop = pd.read_csv(self.out_url+"selected_propagation.csv")
        #df_prop = df_prop[df_prop['propagation'] != 0]
        return max(df_prop.propagation), np.mean(df_prop.propagation), np.median(df_prop.propagation)
    
    def get_computing_capability(self):
        df_node_mining = self.df_node[self.df_node.nodeID.isin(self.df_block.minter)]
        #df_prop = df_prop[df_prop['propagation'] != 0]
        return max(df_node_mining.miningPower), min(df_node_mining.miningPower), np.mean(df_node_mining.miningPower), np.median(df_node_mining.miningPower)

    def get_transmition(self):
        df_flow = pd.read_csv(self.out_url+"block_flow.csv")
        df_flow['transmition'] = df_flow['reception-timestamp'] - df_flow['transmission-timestamp']
        return min(df_flow.transmition), max(df_flow.transmition), np.mean(df_flow.transmition), np.median(df_flow.transmition)
    
    def get_reward(self, block_size, trx_size, interval, c1, c2, c3):
        if c1 and c2 and c3:
            return ((block_size / trx_size) / interval) * 1000
        return 0
    
    def gini(self, x):
        # (Warning: This is a concise implementation, but it is O(n**2)
        # in time and memory, where n = len(x).  *Don't* pass in huge
        # samples!)

        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        return g
    
    def set_perent(self, id_, perent, validators, minter, counter):
        if perent != 0 and counter <= MINIMUM_VALIDATOR:
            df_new = self.df_block[self.df_block['id'] == perent]
            for index2, row2 in df_new.iterrows():
                if minter not in validators[str(row2['id'])]:
                    validators[str(row2['id'])].append(minter)
                    self.set_perent(row2['id'], row2['parent'], validators,  minter, counter+1)

    def get_distribution(self):
        validators_ = {}
        validatorId = self.df_node[self.df_node['validator'] == True].nodeID.values
        
        for index, row in self.df_block.iterrows():
            if str(row['id']) not in validators_:
                validators_[str(row['id'])] = list()
            validators_[str(row['id'])].append(row['minter'])
            self.set_perent(row['id'], row['parent'], validators_, row['minter'], 1)


        dist_minter = {}
        for id_ in validatorId:
            dist_minter[str(id_)] = 0
        for key, value in validators_.items():
            for i in value:
                dist_minter[str(i)] += 1
        
        
        distribution = [v for k, v  in dist_minter.items()]
        gini_dist = self.gini(distribution)
        return  gini_dist <= 1 and gini_dist >= 0, gini_dist
    
    def get_ttf(self, interval):
        time_distribution = max(self.df_block.time)
        return  time_distribution <= interval * MINIMUM_VALIDATOR * HEIGHT_PER_EPISODE, interval * MINIMUM_VALIDATOR * HEIGHT_PER_EPISODE,  time_distribution
    
    def get_security(self):
        return  MINIMUM_VALIDATOR >= (NUM_OF_VALIDATOR+1)/3, MINIMUM_VALIDATOR/NUM_OF_VALIDATOR
    
    # start 100 blockchain simulation with genesis
    # response state, reward, doneflag info
    def step(self, action, prefix, type_step, _step):
        info = {}
        self.cstep += 1
        out_folder = OUT_FOLDER.format(prefix, type_step, _step)
        self.out_url = OUTPUT_URL.format(out_folder) 
        if type_step == 'propose':
            interval = self.mm_denorm_blockinterval(action[0])
            self.block_size = self.mm_denorm_blocksize(action[1])
        elif type_step == 'fix_blocksize':
            self.block_size = 5242880
            interval = self.mm_denorm_blockinterval(action[0])
        elif type_step == 'fix_interval':
            self.block_size = self.mm_denorm_blocksize(action[0])
            interval = (1000*60*10)
            
        
        info['prefix'] = prefix
        info['episode'] = _step
        info['step'] = self.cstep
        info['block_size'] = interval
        info['block_interval'] = self.block_size
        
        
        self.update_config(interval, self.block_size, out_folder)
        
        self.run_simulator()
        
        
        self.df_node = pd.read_csv(self.out_url+"list_node.csv")
        self.df_block = pd.read_csv(self.out_url+"list_block.csv")
        self.df_block_orphans = self.df_block[self.df_block['type'] == 'Orphan']
        self.df_block = self.df_block[self.df_block['type'] == 'OnChain']
        propagation_ = self.get_propagation()
        comp_capability_ = self.get_computing_capability()
        transmition_ = self.get_transmition()
        C1, val_distribution = self.get_distribution()
        C2, info['ttf_expected'], info['ttf_reality'] = self.get_ttf(interval)
        C3, info['security_tolerance'] = self.get_security()
        #logging.debug((C1, val_distribution))
        #logging.debug((C2, info['ttf_expected'], info['ttf_reality']))
        #logging.debug((C3, val_decimal))
        trx_size = self.generate_transaction()
        state = np.array([val_distribution, 
                          self.z_norm_trx(trx_size), 
                          self.mm_norm_computing(comp_capability_[2]), 
                          self.mm_norm_transmitions_rate(transmition_[2])])
        
        info['dezentralisation'] = val_distribution
        info['comp_capability_'] = comp_capability_[2]
        info['trans_rate'] = transmition_[2]
        
        new_number_of_sealed_block = HEIGHT_PER_EPISODE - MINIMUM_VALIDATOR
        sealed_block_size = (self.block_size * new_number_of_sealed_block) + (MINIMUM_VALIDATOR * self.pref_block_size)
        self.pref_block_size = self.block_size
        self.sealed_block += sealed_block_size
        
        info['number_of_sealed_block'] = new_number_of_sealed_block
        info['minumum_validator'] = MINIMUM_VALIDATOR
        info['s_block_size_prev'] = (MINIMUM_VALIDATOR * self.pref_block_size)
        info['s_block_size_new'] = (self.block_size * new_number_of_sealed_block)
        info['s_block_size_tot'] = sealed_block_size
        info['s_block_size_tot'] = self.sealed_block
        info['trx_size'] = trx_size
        info['max_ttf'] = max(self.df_block.time)
        reward = self.get_reward(sealed_block_size, trx_size, max(self.df_block.time), C1, C2, C3)
        
        info['reward'] = reward
        info['tot_reward'] = LIMIT_SEALED_BLOCK
        info['tot_reward'] = self.tot_reward
        self.tot_reward += reward 
        if self.sealed_block > LIMIT_SEALED_BLOCK:
            return (state, reward, True, info)
        return (state, reward, False, info)


# In[2]:


#Environtment
env = BlockchainEnvironment()

#import matplotlib.pyplot as plt
#%matplotlib inline


# In[5]:


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(0, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        return F.tanh(self.fc2(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(0, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# In[6]:


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        
        logging.warning(action)
        return np.clip(action, 0.0000001, 7.0)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# In[7]:


agent_dict = {
    "propose": Agent(state_size=4, action_size=2, random_seed=10),
    "fix_blocksize": Agent(state_size=4, action_size=1, random_seed=10),
    "fix_interval": Agent(state_size=4, action_size=1, random_seed=10)
}

logger.setLevel(logging.WARNING)

import pickle


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
            torch.save(agent.actor_local.state_dict(), "APAC_"+prefix+"_"+step_type + '_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), "APAC_"+prefix+"_"+step_type + '_checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
    return scores, infos


step_types = ['propose','fix_interval', 'fix_blocksize']
for step_type in step_types:
    scores, infos = ddpg(agent_dict[step_type],step_type=step_type)

    with open("APAC_"+prefix+"_"+step_type+"_scores.pckl", "wb") as f:
        pickle.dump(scores, f)

    with open("APAC_"+prefix+"_"+step_type+"_infos.pckl", "wb") as f:
        pickle.dump(infos, f)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(1, len(scores)+1), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

# news = scores[:100]

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(1, len(news)+1), news)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

# In[ ]:




