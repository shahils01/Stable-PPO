import torch
import gymnasium as gym
from ppo.envs.env_wrappers import ShareVecEnv

class IsaacLabVecEnvWrapper(ShareVecEnv):
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.num_envs = self.env.unwrapped.num_envs

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Isaac Lab typically returns a dict for obs, we assume policy obs are in "policy"
        # If your env returns a flat tensor, you can adjust this.
        # self._check_obs_structure()

    def _check_obs_structure(self, sample):
        # Helper to determine how to extract observations
        # sample = self.env.observation_space.sample()
        if isinstance(sample, dict) and "policy" in sample:
            self._obs_key = "policy"
        else:
            self._obs_key = None # Assume flat observation

    def _extract_obs(self, obs):
        if self._obs_key:
            return obs[self._obs_key]
        return obs

    def reset(self):
        obs, _ = self.env.reset()
        self._check_obs_structure(obs)
        obs = self._extract_obs(obs)
        # Your runner expects: (num_envs, n_agents, obs_dim) or (num_envs, obs_dim)
        # IsaacLab returns (num_envs, obs_dim). We might need to unsqueeze if your code expects n_agents dim
        # For single-agent RL in IsaacLab, usually just returning the tensor is enough.
        return obs

    def step_async(self, actions):
        # Isaac Lab is synchronous, so we just store actions for step_wait
        self.actions = actions
        self.device = actions.device

    def step_wait(self):
        # Helper to handle tensor/numpy conversion if needed
        # Your PPO code seems to use PyTorch, so passing tensors is preferred!
        
        # Ensure actions are on the correct device
        if isinstance(self.actions, torch.Tensor):
            actions = self.actions.to(self.device)
        else:
            actions = torch.tensor(self.actions, device=self.device)

        obs, rews, terminated, truncated, infos = self.env.step(actions)
        
        obs = self._extract_obs(obs)
        
        # Combine done flags (terminated or truncated)
        dones = terminated | truncated
        
        # Your runner might expect numpy. If you want high performance, 
        # modify your runner to handle tensors. For compatibility, we can leave as tensors
        # because your PPO.py uses 'check(obs)' which handles tensors.
        
        return obs, rews, dones, infos

    def close(self):
        self.env.close()

    def get_images(self):
        # Isaac Lab rendering is complex, often best handled by the app launcher
        return None