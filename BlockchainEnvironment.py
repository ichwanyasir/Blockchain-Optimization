import random
import pandas as pd
import numpy as np
import collections
import subprocess
import logging
import copy
from collections import namedtuple, deque

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONFIG_URL = """simblock_{}/simulator/src/simulation.config"""
OUTPUT_URL = """simblock_{}/simulator/src/dist/{}/"""
HEIGHT_PER_EPISODE = 100
CONS_ALGO = """SimBlock.node.consensusAlgo.ProofOfAuthority"""
NUM_OF_NODES=600
OUT_FOLDER = "PoA/{}/{}/{}/episode-{}" # (prefix, type, episode)

MINIMUM_VALIDATOR = 34
NUM_OF_VALIDATOR = 100

# APAC configuration
T_CONF = {
    "APAC": {
        "AVG_TRANSACTION_SIZE": 12400,
        "STD_TRANSACTION_SIZE": 1000,
        "OPERATION_SPEED": 6854
    },
    "AIH": {
        "AVG_TRANSACTION_SIZE": 7900,
        "STD_TRANSACTION_SIZE": 1000,
        "OPERATION_SPEED": 5607
    }
}
# AIH configuration
#AVG_TRANSACTION_SIZE = 7900
#STD_TRANSACTION_SIZE = 1000
#OPERATION_SPEED = 5607

#Normalization Param
# Computing; data: (588375, 153844, 405276.1538461539, 409339.0)
MAX_COMPUTING = 600000
MIN_COMPUTING = 120000

# Block interval; data : 1000*1//1000*60;//1000*30*5;//1000*60*10;//1000*60*60
MAX_BLOCKINTERVAL = 3600000
MIN_BLOCKINTERVAL = 5000

# Blocksize; data : //6110;//8000;//535000;//0.5MB;//1MB;//10MB;
MAX_BLOCKSIZE = 10485760 
MIN_BLOCKSIZE = 6110


# Transmition Rate; data :(316, 5908, 539.6932715030989, 535.0)
MAX_TRANSMITIONRATE = 5000
MIN_TRANSMITIONRATE = 100

# Simulation Environment
class BlockchainEnvironment():
    def __init__(self, type_):
        self.reset()
        self.type = type_
        self.t_conf = T_CONF[type_]
        
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
        with open(CONFIG_URL.format(self.type), "w") as text_file:
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
                             self.t_conf['OPERATION_SPEED']))

    def run_simulator(self):
        cmd = "cd simblock_{} && gradle run".format(self.type)
        # no block, it start a sub process.
        p = subprocess.Popen(cmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        for line in p.stdout:
            logging.info(line)
        p.wait()
        return p.returncode
    
    def z_norm_trx(self, t):
        return abs(t - self.t_conf["AVG_TRANSACTION_SIZE"])/self.t_conf["STD_TRANSACTION_SIZE"]
    
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
        return int(v * (max_ - min_) + min_)
    
    def mm_norm_transmitions_rate(self, v):
        # data (316, 5908, 539.6932715030989, 535.0)
        max_ = MAX_TRANSMITIONRATE
        min_ = MIN_TRANSMITIONRATE
        return (v - min_) / (max_ - min_)

    
    def generate_transaction(self):
        size = int(self.block_size/self.t_conf['AVG_TRANSACTION_SIZE'])+1
        array = np.random.normal(loc=self.t_conf['AVG_TRANSACTION_SIZE'], scale=self.t_conf['STD_TRANSACTION_SIZE'], size=size)
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
        # print(c1, c2, c3)
        troughput = (block_size/trx_size)/(interval/1000)
        if c1 and c2 and c3:
            return troughput/10, troughput
        return 0, troughput
    
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
        # print(time_distribution, interval, MINIMUM_VALIDATOR, HEIGHT_PER_EPISODE, (interval * MINIMUM_VALIDATOR * HEIGHT_PER_EPISODE))
        return  time_distribution <= interval * MINIMUM_VALIDATOR * HEIGHT_PER_EPISODE, interval * MINIMUM_VALIDATOR * HEIGHT_PER_EPISODE,  time_distribution
    
    def get_security(self):
        return  MINIMUM_VALIDATOR >= (NUM_OF_VALIDATOR+1)/3, MINIMUM_VALIDATOR/NUM_OF_VALIDATOR
    
    # start 100 blockchain simulation with genesis
    # response state, reward, doneflag info
    def step(self, action, prefix, type_step, _step):
        info = {}
        self.cstep += 1
        out_folder = OUT_FOLDER.format(self.type, prefix, type_step, _step)
        self.out_url = OUTPUT_URL.format(self.type, out_folder) 
        if type_step == 'propose':
            interval = self.mm_denorm_blockinterval(action[0])
            self.block_size = self.mm_denorm_blocksize(action[1])
        elif type_step == 'fix_blocksize':
            self.block_size = 5242880
            interval = self.mm_denorm_blockinterval(action[0])
        elif type_step == 'fix_interval':
            self.block_size = self.mm_denorm_blocksize(action[0])
            interval = (1000*60*10)
        elif type_step == 'static':
            self.block_size = 5242880
            interval = (1000*60*10)
            
            
        
        info['prefix'] = prefix
        info['episode'] = _step
        info['step'] = self.cstep
        info['block_size'] = self.block_size
        info['block_interval'] = interval
        
        
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
        info['C1'] = C1
        info['C2'] = C2
        info['C3'] = C3

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
        self.sealed_block += sealed_block_size
        
        info['number_of_sealed_block'] = new_number_of_sealed_block
        info['minumum_validator'] = MINIMUM_VALIDATOR
        info['s_block_size_prev'] = (MINIMUM_VALIDATOR * self.pref_block_size)
        info['s_block_size_new'] = (self.block_size * new_number_of_sealed_block)
        info['s_block_size_pref_n_new'] = sealed_block_size
        info['s_block_size_tot'] = self.sealed_block
        info['trx_size'] = trx_size
        info['max_ttf'] = max(self.df_block.time)
        reward, troughput = self.get_reward(sealed_block_size, trx_size, max(self.df_block.time), C1, C2, C3)
        
        info['troughput'] = troughput
        info['reward'] = reward
        #info['tot_reward'] = LIMIT_SEALED_BLOCK
        info['tot_reward'] = self.tot_reward
        self.pref_block_size = self.block_size
        self.tot_reward += reward 
        # if self.sealed_block > LIMIT_SEALED_BLOCK:
        #     return (state, reward, True, info)
        return (state, reward, False, info)