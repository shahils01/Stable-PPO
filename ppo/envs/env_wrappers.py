"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from ppo.utils.util import tile_images

def _stack_obs(obs_list):
    if len(obs_list) == 0:
        return np.array([])
    if isinstance(obs_list[0], dict):
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0].keys()}
    return np.stack(obs_list)

def _assign_obs(obs, idx, new_obs):
    if isinstance(obs, dict):
        for k in obs.keys():
            obs[k][idx] = new_obs[k]
    else:
        obs[idx] = new_obs

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, _ = env.reset()
            else:
                if np.all(done):
                    ob, _ = env.reset()

            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob, _ = env.reset()
            remote.send((ob))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob, _ = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _stack_obs(list(obs)), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return _stack_obs(obs)


    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return _stack_obs([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_images(self):
        for remote in self.remotes:
            remote.send(('render', "rgb_array"))
        return [remote.recv() for remote in self.remotes]

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame) 

    def get_images(self):
        for remote in self.remotes:
            remote.send(('render', "rgb_array"))
        return [remote.recv() for remote in self.remotes]


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    noise_scale = 0.1
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, terminated, truncated, info = env.step(data)
            # Adding noise into the observation
            # noise = np.random.normal(0, noise_scale, size=ob.shape)
            # ob += noise
            done = terminated or truncated
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, _ = env.reset()
            else:
                if np.all(done):
                    ob, _ = env.reset()

            remote.send((ob, reward, terminated, truncated, info))   
        elif cmd == 'reset':
            ob, _ = env.reset()
            remote.send((ob))
        elif cmd == 'reset_task':
            ob, _ = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.action_space))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space, action_space)        

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, terminated, truncated, infos = zip(*results)   
        return _stack_obs(list(obs)), np.stack(rews), np.stack(terminated), np.stack(truncated), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return _stack_obs(results)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return _stack_obs([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="human"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def get_images(self):
        for remote in self.remotes:
            remote.send(('render', "rgb_array"))
        return [remote.recv() for remote in self.remotes]


def choosesimpleworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob, _ = env.reset(data)
            remote.send((ob))
        elif cmd == 'reset_task':
            ob, _ = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseSimpleSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=choosesimpleworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _stack_obs(list(obs)), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        obs = [remote.recv() for remote in self.remotes]
        return _stack_obs(obs)

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def get_images(self):
        for remote in self.remotes:
            remote.send(('render', "rgb_array"))
        return [remote.recv() for remote in self.remotes]

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = zip(*results)
        obs = _stack_obs(list(obs))
        rews = np.array(rews)
        dones = np.array(dones)
        infos = np.array(infos)

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    new_obs = self.envs[i].reset()
                    _assign_obs(obs, i, new_obs)
            else:
                if np.all(done):
                    new_obs = self.envs[i].reset()
                    _assign_obs(obs, i, new_obs)

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return _stack_obs(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            frames = []
            for env in self.envs:
                try:
                    frames.append(env.render(mode=mode))
                except TypeError:
                    frames.append(env.render())
            return np.array(frames)
        elif mode == "human":
            for env in self.envs:
                try:
                    env.render(mode=mode)
                except TypeError:
                    env.render()
        else:
            raise NotImplementedError

    def get_images(self):
        frames = []
        for env in self.envs:
            try:
                frames.append(env.render(mode="rgb_array"))
            except TypeError:
                frames.append(env.render())
        return frames



class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, terminated, truncated, infos = zip(*results)
        obs = _stack_obs(list(obs))
        rews = np.array(rews)
        terminated = np.array(terminated)
        truncated = np.array(truncated)
        infos = np.array(infos)

        dones = terminated or truncated

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    new_obs, _ = self.envs[i].reset()
                    _assign_obs(obs, i, new_obs)
            else:
                if np.all(done):
                    new_obs, _ = self.envs[i].reset()
                    _assign_obs(obs, i, new_obs)
        self.actions = None

        return obs, rews, terminated, truncated, infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, _ = zip(*results)
        return _stack_obs(list(obs))

    def close(self):
        for env in self.envs:
            env.close()

    def save_replay(self):
        for env in self.envs:
            env.save_replay()
    
    def render(self, mode="human"):
        if mode == "rgb_array":
            frames = []
            for env in self.envs:
                try:
                    frames.append(env.render(mode=mode))
                except TypeError:
                    frames.append(env.render())
            return np.array(frames)
        elif mode == "human":
            for env in self.envs:
                try:
                    env.render(mode=mode)
                except TypeError:
                    env.render()
        else:
            raise NotImplementedError

    def get_images(self):
        frames = []
        for env in self.envs:
            try:
                frames.append(env.render(mode="rgb_array"))
            except TypeError:
                frames.append(env.render())
        return frames
