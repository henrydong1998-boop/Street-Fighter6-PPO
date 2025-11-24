import gym
from gym import spaces
import numpy as np
import mujoco
import mujoco.viewer

class SFEnv(gym.Env):
    """
    A minimal Gym wrapper for a HalfCheetah-style MuJoCo XML.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, xml_path="half_cheetah.xml"):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._viewer = None

        # ---- Action & observation spaces ----
        # MuJoCo automatically provides these
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        self.obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        qvel = self.data.qvel.ravel()
        return np.concatenate([qpos, qvel]).astype(np.float32)
    

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return np.concatenate([self.data.qpos, self.data.qvel])

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.data.ctrl[:] = action.detach().cpu().numpy().flatten()
        #print("Action applied:", action.detach().cpu().numpy().flatten())
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # HalfCheetah reward
        #reward_ctrl = -0.1 * np.square(action).sum()
        reward_ctrl = -0.1 * np.square(action).sum()
        #print("action:", action)
        reward_run = self.data.qvel[0]
        #print("action:", action)
        reward = reward_run + reward_ctrl

        done = False  # HalfCheetah never terminates early

        info = {"reward_run": reward_run, "reward_ctrl": reward_ctrl}
        # print("ctrl:", self.data.ctrl)
        # print("qvel[0]:", self.data.qvel[0])
        return obs, reward, done, info

    def render(self, mode="human"):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def close(self):
        pass
