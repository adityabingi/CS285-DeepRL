from cs285.infrastructure.utils import *


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def add_transition(self, transition):
        # add new transition into our list of transitions

        observation, action, reward, next_observation, terminal = transition

        if self.obs is None:
            self.obs = observation[-self.max_size:]
            self.acs = action[-self.max_size:]
            self.rews = reward[-self.max_size:]
            self.next_obs = next_observation[-self.max_size:]
            self.terminals = terminal[-self.max_size:]  
        else:
            self.obs = np.concatenate([self.obs, observation])[-self.max_size:]
            self.acs = np.concatenate([self.acs, action])[-self.max_size:]
            self.rews = np.concatenate([self.rews, reward])[-self.max_size:]
            self.next_obs = np.concatenate([self.next_obs, next_observation])[-self.max_size:]
            self.terminals = np.concatenate([self.terminals, terminal])[-self.max_size:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert self.obs.shape[0] == self.acs.shape[0] == self.rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0] 
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.rews[rand_indices], self.next_obs[rand_indices], self.terminals[rand_indices]

    def __len__(self):
        return self.obs.shape[0]

def Transition(observation, action, reward, next_observation, terminal):

    return (np.array([observation], dtype=np.float32),
           np.array([action], dtype=np.float32),
           np.array([reward], dtype=np.float32),
           np.array([next_observation], dtype=np.float32),
           np.array([terminal], dtype=np.float32))