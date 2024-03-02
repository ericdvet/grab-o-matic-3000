import numpy as np

class NaiveModel(object):
    def __init__(self, dof):
        self.dof = dof

    def __call__(self, env, obs=None):
        # a trained policy could be used here, but we choose a random action
        low, high = env.action_spec
        action = np.random.uniform(low, high)
        """if ( )
        print(obs.get('robot0_eef_pos'))"""
        return action