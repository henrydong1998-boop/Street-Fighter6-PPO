import numpy as np
import time
from env.CV_test import CV_test
import env.input_manager as inpm
from wincam import DXCamera
from ultralytics import YOLO
from env.util import update_buffer_svd, random_projection
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
        self.actor_health_history = []
        self.opponent_health_history = []
        self.combo_history = []


    def _get_obs(self):
        obs, actor_state, opponent_state, actor_bbox, opponent_bbox, actor_health, opponent_health = CV_test(self.camera, self.model,self.model1)
        self.actor_state = actor_state
        self.opponent_state = opponent_state
        self.actor_bbox = actor_bbox
        self.opponent_bbox = opponent_bbox
        self.actor_health_history.append(actor_health)
        self.opponent_health_history.append(opponent_health)
        if len(self.actor_health_history) > 100:
            self.actor_health_history = self.actor_health_history[90: ]
        if len(self.opponent_health_history) > 100:
            self.opponent_health_history = self.opponent_health_history[90: ]
        return obs


    def reset(self):
        obs, actor_state, opponent_state, actor_bbox, opponent_bbox, actor_health, opponent_health = CV_test(self.camera, self.model,self.model1)
        self.actor_state = actor_state
        self.opponent_state = opponent_state
        self.actor_bbox = actor_bbox
        self.opponent_bbox = opponent_bbox
        self.actor_health_history.append(actor_health)
        self.opponent_health_history.append(opponent_health)
        if len(self.actor_health_history) > 100:
            self.actor_health_history = self.actor_health_history[90: ]
        if len(self.opponent_health_history) > 100:
            self.opponent_health_history = self.opponent_health_history[90: ]
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
        # TODO: change this to a multi processing pipe
        right = self.input_manager.update_facing(self.actor_bbox, self.opponent_bbox)
        print(f"facing {"right" if right else "left"}")
        self.input_manager.accept_prediction(action)
        # time.sleep(0.01)
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

        if len(self.combo_history) > 100:
            self.combo_history = self.combo_history[90 : ]

        if len(self.actor_health_history) > 1:
            delt_health = self.actor_health_history[-2] - self.actor_health_history[-1]
        if len(self.opponent_health_history) > 1:
            opn_delt_health = self.opponent_health_history[-2] - self.opponent_health_history[-1] 

        reward_delt_health -= delt_health * 200
        reward_opn_delt_health += opn_delt_health * 200

        # award successful combo
        if self.input_manager.get_action_dict(inpm.InputClass(action + 1))["combo_breaks"][0] != -1:
            if len(self.combo_history) > 0 \
               and self.combo_history[-1] == action \
               and self.opponent_state == 5:
                reward_combo = 10
            self.combo_history.append(action)

        if self.actor_state == 15: # actor hit
            reward_delt_health -= 12
        if self.opponent_state == 5: # opponent hit
            reward_opn_delt_health += 16
        if self.actor_state in range(10, 13): # actor combos
            reward_atk = 10
        if self.actor_state == 13 or self.actor_state == 14: # actor guard
            reward_guard = 1
        if self.opponent_state!=5 and self.actor_state >=10 and self.actor_state<15: # actor miss
            reward_miss = -2
        if self.actor_state == 17: # actor neutral
            reward_neutral = -8

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
