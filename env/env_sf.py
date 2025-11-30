import numpy as np
import time
from env.CV_test import CV_test
import env.input_manager as inpm
from ultralytics import YOLO
from env.util import update_buffer_svd, random_projection
from wincam import DXCamera
import torch
import numpy as np


class SFEnv:
    """
    import CV module to get game state information for RL training.
    The observation consists of:
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        self.itr = 0
        self.input_manager = inpm.InputManager("./mai_combos.json")

        # ---- Action & observation spaces ----
        # self.action_space = np.array([e.value for e in self.input_manager.InputClass], dtype=int)
        self.action_space = torch.tensor([e.value for e in self.input_manager.InputClass], dtype=int)

        self.obs = None

        self.model = YOLO('./yolo/best.pt')
        self.model1 = YOLO('./yolo/best2.pt')

        self.obs_dim= CV_test(self.model,self.model1,self.input_manager)[0].shape[0]
        self.actor_state = None
        self.opponent_state = None


    def _get_obs(self):
        obs,actor_state, opponent_state=CV_test(self.model,self.model1,self.input_manager)
        self.actor_state = actor_state
        self.opponent_state = opponent_state
        return obs
    

    def reset(self):
        obs,actor_state, opponent_state=CV_test(self.model,self.model1,self.input_manager)
        self.actor_state = actor_state
        self.opponent_state = opponent_state
        self.itr = 0
        return obs

    def step(self, dist):
        # action = np.clip(action, -1, 1)
        action_probs = dist.probs.squeeze(0)
        print(f'action: {action_probs}')
        filter = inpm.filter_input(self.actor_state, self.opponent_state)
        action_probs[filter == False] = 0.0
        probs_modified = action_probs.clone()
        probs_normalized = probs_modified / probs_modified.sum()
        action = torch.multinomial(probs_normalized, num_samples=1)
        #To Do: change this to a multi processing pipe
        self.input_manager.accept_prediction(int(action))
        self.input_manager.output_actions()
        # time.sleep(0.02)  # wait for environment to update
        self.obs = self._get_obs()

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
        
        return self.obs, reward, done, info

    def close(self):
        pass
