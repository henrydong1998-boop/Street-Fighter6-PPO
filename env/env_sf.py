import numpy as np
import time
from CV_test import CV_test
import input_manager as inpm

class SFEnv:
    """
    import CV module to get game state information for RL training.
    The observation consists of:
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        self.itr = 0
        self.input_manager = inpm.InputManager

        # ---- Action & observation spaces ----
        self.action_space = np.array([e.value for e in self.input_manager.InputClass], dtype=int)

        self.obs_dim = CV_test().shape[0]
        self.obs

    def _get_obs(self):
        obs=CV_test()
        return obs
    

    def reset(self):
        obs=CV_test()
        self.itr = 0
        return obs

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.input_manager.accept_prediction(action)
        self.input_manager.output_actions()
        time.sleep(0.02)  # wait for environment to update
        obs = self._get_obs()

        # SF6 reward
        
        reward_delt_health = 0
        reward_opn_delt_health = 0
        reward_atk = 0
        reward_gud = 0
        reward_miss = 0
        reward = reward_delt_health + reward_opn_delt_health + reward_atk + reward_gud + reward_miss

        done = False  # HalfCheetah never terminates early

        info = {"reward": reward}
        self.itr = self.itr + 1
        
        return obs, reward, done, info

    def close(self):
        pass
