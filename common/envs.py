import atexit
import os
import sys
import threading
import traceback

import cloudpickle
import gym
import numpy as np

import mujoco_py
from dm_control.mujoco import engine


OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

BONUS_THRESH = 0.3


def make_env(mode, config):
    suite, task = config.task.split("_", 1)
    if suite == "kitchen":
        env = Kitchen(get_kitchen_tasks(task), size=(64, 64), mode=mode)
        env = GymWrapper(env)
    elif suite == 'shadowhand' or suite == 'baoding':
        env = BaodingEnv(size=(64, 64))
        env = GymWrapper(env)
    else:
        raise NotImplementedError(suite)
    env = TimeLimit(env, config.time_limit)
    return env


def get_kitchen_tasks(task="id"):
    if task == "mixed" or task == "id":
        task = "microwave+kettle+light switch+slide cabinet"
    elif task == "partial" or task == "ood":
        task = "microwave+kettle+bottom burner+light switch"
    elif task == "complete":
        task = "microwave+kettle+bottom burner+light switch+slide cabinet"
    return task.split("+")


class Kitchen:
    def __init__(self, task=["microwave"], size=(64, 64), mode="train", proprio=False):
        from .RPL.adept_envs import adept_envs

        self._env = gym.make("kitchen_relax-v1")
        self._all_tasks = task.copy()
        self._current_tasks = task.copy()
        self._REMOVE_TASKS_WHEN_COMPLETE = {"train": False, "eval": True}[mode]
        self._img_h = size[0]
        self._img_w = size[1]
        self._proprio = proprio
        self.tasks_to_complete = [
            "bottom burner",
            "top burner",
            "light switch",
            "slide cabinet",
            "hinge cabinet",
            "microwave",
            "kettle",
        ]

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, *args, **kwargs):
        obs, _, done, info = self._env.step(*args, **kwargs)
        reward_dict = self._compute_reward_dict(obs)
        img = self.render(mode="rgb_array", size=(self._img_h, self._img_w))
        obs_dict = dict(image=img)

        if self._proprio:
            obs_dict["proprio"] = obs[:9]

        return obs_dict, reward_dict["reward"], done, info

    def reset(self, *args, **kwargs):
        self._current_tasks = self._all_tasks.copy()
        obs = self._env.reset(*args, **kwargs)
        img = self.render(mode="rgb_array", size=(self._img_h, self._img_w))
        # obs_dict = dict(image=img, **reward_dict)
        obs_dict = dict(image=img)

        if self._proprio:
            obs_dict["proprio"] = obs[:9]

        return obs_dict

    def _compute_reward_dict(self, obs):
        reward_dict = {}
        completions = []
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            obs_obj = obs[..., element_idx]
            obs_goal = OBS_ELEMENT_GOALS[element]
            distance = np.linalg.norm(obs_obj - obs_goal)
            complete = distance < BONUS_THRESH
            reward_dict["reward " + element] = 1.0 * complete
            if complete:
                completions.append(element)
        reward_dict["reward"] = sum(
            [reward_dict["reward " + obj] for obj in self._current_tasks]
        )
        if self._REMOVE_TASKS_WHEN_COMPLETE:
            for element in self.tasks_to_complete:
                if element in self._current_tasks and element in completions:
                    self._current_tasks.remove(element)

            obs_dict = self.obs_dict
        return reward_dict

    def render(self, mode="human", size=(1920, 2550)):
        if mode == "rgb_array":
            camera = engine.MovableCamera(self._env.sim, size[0], size[1])
            camera.set_pose(
                distance=1.86, lookat=[-0.3, 0.5, 2.0], azimuth=90, elevation=-60
            )
            img = camera.render()
            return img
        else:
            self._env.render(mode, size)

    @property
    def observation_space(self):
        spaces = {}
        spaces["image"] = gym.spaces.Box(
            0, 255, (self._img_h, self._img_w, 3), dtype=np.uint8
        )

        if self._proprio:
            spaces["proprio"] = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)

        return gym.spaces.Dict(spaces)


class BaodingEnv:
    def __init__(self, size=(128, 128)):
        import pddm
        from pddm.envs.baoding.baoding_env import BaodingEnv

        self._env = BaodingEnv()
        self.size = size

        self._env.sim_robot.renderer._camera_settings["distance"] = 0.3
        self._env.sim_robot.renderer._camera_settings["azimuth"] = -67.5
        self._env.sim_robot.renderer._camera_settings["elevation"] = -42.5

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._env, attr)

    @property
    def observation_space(self):
        spaces = {}
        spaces["image"] = gym.spaces.Box(
            0, 255, (*self.size, 3), dtype=np.uint8
        )
        return gym.spaces.Dict(spaces)

    def step(self, action):
        _, reward, done, info = self._env.step(action)
        img = self._env.render(
            mode="rgb_array", width=self.size[0], height=self.size[1]
        )
        # obs = self._env.obs_dict.copy()
        # obs["obs"] = state
        obs = dict(image=img)
        return obs, reward, done, info

    def reset(self):
        _ = self._env.reset()
        img = self._env.render(
            mode="rgb_array", width=self.size[0], height=self.size[1]
        )
        # obs = self._env.obs_dict.copy()
        # obs["obs"] = state
        # obs["image"] = img
        obs = dict(image=img)
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.render(
            mode="rgb_array", width=self.size[0], height=self.size[1]
        )



class GymWrapper:
    def __init__(self, env, obs_key="image", act_key="action"):
        self._env = env
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._act_is_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return {
            **spaces,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, done, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = float(reward)
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = info.get("is_terminal", done)
        return obs

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["reward"] = 0.0
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs



class Dummy:
    def __init__(self):
        pass

    @property
    def obs_space(self):
        return {
            "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        return {"action": gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

    def step(self, action):
        return {
            "image": np.zeros((64, 64, 3)),
            "reward": 0.0,
            "is_first": False,
            "is_last": False,
            "is_terminal": False,
        }

    def reset(self):
        return {
            "image": np.zeros((64, 64, 3)),
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()

