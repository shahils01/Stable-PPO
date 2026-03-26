import time
import wandb
import numpy as np
from functools import reduce
import torch
import imageio
import gymnasium as gym
import torch.nn.functional as F

from ppo.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


def faulty_action(action, faulty_node):
    action_fault = action.copy()
    if faulty_node >= 0:
        action_fault[:, faulty_node, :] = 0.
        # action[:, faulty_node, :] = 0.
    # return action
    return action_fault


class MujocoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(MujocoRunner, self).__init__(config)
        self.reward_list = []
        self.reward_list_training = []
        self.plt_name = 'ppo'
    
    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs = self.collect(step)

                # Obser reward and next obs
                obs, rewards, terminated, truncated, infos = self.envs.step(actions)
                rewards = rewards.reshape(-1, 1)
                dones = terminated.reshape(-1, 1)
                truncated = truncated.reshape(-1, 1)
                                
                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                # Bootstrap reward for truncated episodes (Done after computing train_episode_rewards)
                rewards += self.all_args.gamma * np.mean(values, axis=-1,keepdims=True) * truncated

                data = obs, rewards, dones, infos, \
                       values, actions, action_log_probs
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                fps = int(total_num_steps / (end - start))  # Calculate FPS first
                print("\n Scenario {} Algo {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(
                        self.all_args.scenario,
                        self.algorithm_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        fps
                    ))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)
                    
                    if self.use_wandb:
                        wandb.log({"average_episode_rewards": aver_episode_rewards}, step=total_num_steps)
                        
                    #self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, total_num_steps)
                    if self.reward_list_training is None:
                        self.reward_list_training = np.array(np.mean(done_episodes_rewards))
                    else:
                        self.reward_list_training = np.hstack((self.reward_list_training, np.array(np.mean(done_episodes_rewards))))

                    np.save(self.plt_name+'_training.npy', self.reward_list_training)
                
                    done_episodes_rewards = []

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset() 
        print('obs dim = ', obs.shape)
        self.buffer.obs[0] = np.expand_dims(obs, axis=1).copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.masks[step]))
                
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        
        return values, actions, action_log_probs

    def insert(self, data):
        obs, rewards, dones, infos, \
        values, actions, action_log_probs  = data

        dones_env = np.all(dones, axis=1)

        masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
        active_masks[dones.reshape(-1) == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), 1), dtype=np.float32)

        self.buffer.insert(obs, actions, action_log_probs, values, rewards, masks, active_masks)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.n_eval_rollout_threads)]

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.all_args.n_eval_rollout_threads, 1), dtype=np.float32)
        
        all_frames = []

        while True:
            self.trainer.prep_rollout()
            eval_actions = \
                self.trainer.policy.act(eval_obs,
                                        eval_masks)
            eval_actions = _t2n(eval_actions)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_terminated, eval_truncated, eval_infos = self.eval_envs.step(eval_actions)
            eval_rewards = eval_rewards.reshape(-1, 1)
            eval_dones = eval_terminated | eval_truncated
            eval_dones = eval_dones.reshape(-1,1)
            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), 1),
                                                          dtype=np.float32)

            for eval_i in range(self.all_args.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    one_episode_rewards[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = 'eval_average_episode_rewards'
                key_max = 'eval_max_episode_rewards'
                eval_env_infos = {key_average: eval_episode_rewards,
                                  key_max: [np.max(eval_episode_rewards)]}

                self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}."
                      .format(np.mean(eval_episode_rewards)))
                
                if self.use_wandb:
                        wandb.log({"eval_average_episode_rewards": np.mean(eval_episode_rewards)}, step=total_num_steps)
                
                if self.reward_list is None:
                    self.reward_list = np.array(np.mean(eval_episode_rewards))
                else:
                    self.reward_list = np.hstack((self.reward_list, np.array(np.mean(eval_episode_rewards))))

                np.save(self.plt_name+'_eval.npy', self.reward_list)
        
                break
            
        # Save video
        '''if len(frames) > 0:
            #video_dir = os.path.join(self.run_dir, 'videos')
            video_dir = os.path.join('/home/shahils/Desktop/marl_ws/Multi-Agent-Transformer_old/mat/scripts/videos')
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f'step_{total_num_steps}.mp4')
            imageio.mimsave(video_path, all_frames, fps=30)'''
