import numpy as np
import time
from env.CV_test import CV_test
import env.input_manager as inpm
from wincam import DXCamera
from ultralytics import YOLO
from env.util import update_buffer_svd, random_projection
from collections import deque
from itertools import islice
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
        self.action_space = torch.tensor([e.value for e in inpm.InputClass], dtype=int)

        self.obs = None

        self.camera = DXCamera(0, 32, 1920, 1080, fps=30)
        self.model = YOLO('./yolo/best.pt')
        self.model1 = YOLO('./yolo/best2.pt')

        self.obs_dim= CV_test(self.camera, self.model,self.model1)[0].shape[0]
        self.actor_state = None
        self.opponent_state = None
        self.actor_bbox = None
        self.opponent_bbox = None
        self.actor_health_history = deque(maxlen=6)
        self.opponent_health_history = deque(maxlen=6)
        self.combo_history = deque(maxlen=10)
        self.neutral_history = deque(maxlen=30)


    def _get_obs(self):
        obs, actor_state, opponent_state, actor_bbox, opponent_bbox, actor_health, opponent_health = CV_test(self.camera, self.model,self.model1)
        self.actor_state = actor_state
        self.opponent_state = opponent_state
        self.actor_bbox = actor_bbox
        self.opponent_bbox = opponent_bbox
        self.actor_health_history.append(actor_health)
        self.opponent_health_history.append(opponent_health)
        return obs


    def reset(self):
        obs, actor_state, opponent_state, actor_bbox, opponent_bbox, actor_health, opponent_health = CV_test(self.camera, self.model,self.model1)
        self.actor_state = actor_state
        self.opponent_state = opponent_state
        self.actor_bbox = actor_bbox
        self.opponent_bbox = opponent_bbox
        self.actor_health_history.append(actor_health)
        self.opponent_health_history.append(opponent_health)
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
        action = int(torch.multinomial(probs_normalized, num_samples=1))

        self.input_manager.update_facing(self.actor_bbox, self.opponent_bbox)
        self.input_manager.accept_prediction(action)
        self.input_manager.output_actions()
        self.obs = self._get_obs()

        # SF6 rewardasakssio
        reward_delt_health = 0
        reward_opn_delt_health = 0
        reward_atk = 0
        reward_guard = 0
        reward_miss = 0
        reward_neutral = 0
        reward_combo = 0

        if self.actor_health_history[-1] == 0 or self.opponent_health_history[-1] == 0:
            self.actor_health_history = deque(maxlen=6)
            self.opponent_health_history = deque(maxlen=6)

        if len(self.actor_health_history) > 1 and len(self.actor_health_history) < 6:
            delt_health = self.actor_health_history[-2] - self.actor_health_history[-1]
        elif len(self.actor_health_history) == 6:
            delt_health = sum(islice(self.actor_health_history, 3)) - \
                          sum(islice(self.actor_health_history, 3, 6))

        if len(self.opponent_health_history) > 1 and len(self.opponent_health_history) < 6:
            opn_delt_health = self.opponent_health_history[-2] - self.opponent_health_history[-1]
        elif len(self.opponent_health_history) == 6:
            opn_delt_health = sum(islice(self.opponent_health_history, 3)) - \
                              sum(islice(self.opponent_health_history, 3, 6))

        reward_delt_health -= max(0, delt_health) * 200
        reward_opn_delt_health += max(0, opn_delt_health) * 200

        # award successful combo
        if self.input_manager.get_action_dict(inpm.InputClass(action + 1))["combo_breaks"][0] != -1:
            if len(self.combo_history) > 0 \
               and self.combo_history[-1] == action:
                reward_combo = 20 if self.opponent_state == 5 else -10
            self.combo_history.append(action)

        if self.actor_state == 15: # actor hit
            reward_delt_health -= 12
        if self.opponent_state == 5: # opponent hit
            reward_opn_delt_health += 12
        if self.actor_state in range(10, 13):
            reward_atk = 5
        if self.actor_state == 13 or self.actor_state == 14: # actor guard
            reward_guard = 20
        if self.opponent_state!=5 and self.actor_state >=10 and self.actor_state<15: # actor miss
            reward_miss = -10

        if self.actor_state == 17: # actor neutral
            self.neutral_history.append(1)
        else:
            self.neutral_history.append(0)
        if sum(self.neutral_history) > max(10, len(self.neutral_history) * 0.67):
            self.reward_neutral = -20

        reward = reward_delt_health + \
                 reward_opn_delt_health + \
                 reward_atk + \
                 reward_guard + \
                 reward_miss + \
                 reward_neutral + \
                 reward_combo

        print(f"Reward:{reward}")
        done = False  # HalfCheetah never terminates early

        info = {"reward": reward}
        self.itr = self.itr + 1

        return self.obs, reward, done, info

    def close(self):
        pass
