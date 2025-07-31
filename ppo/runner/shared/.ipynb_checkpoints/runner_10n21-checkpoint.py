import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
from .consensus import Consensus
from .dlmf import DLMF
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner10n21(Runner):
    def __init__(self, config):
        super(Runner10n21, self).__init__(config)
        
        # Intialize consensus
        self.consensus = Consensus(self.envs.adjacency_matrix)
        
        # Intialize consensus
        self.dlmf = DLMF(self.envs)
        self.cost_list = []
        self.plt_name = 'ten_nodes_8'
        # Ensure the 'results' directory exists
        os.makedirs('results', exist_ok=True)
        
        self.step = 0
               
    def run(self):
        self.warmup()
        # reset env
        obs = self.envs.reset()
        print('obs intial = ', obs)
        
        const = np.dot(self.envs.B, np.transpose(obs[0,:])) + self.envs.C
        print('Initial constraint = ', const)

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
                    
            episode_reward = 0

            for step in range(self.episode_length):
                store = True
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                actions = actions.reshape((1,-1))
                #actions = 5*np.tanh(actions[0,:,:]).reshape((1,-1))
                                
                # Adjust actions via DLMF
                dlmf_action, iterations = self.dlmf.forward(actions[0,:], obs)
                if iterations == 19999:
                    print('Hit max iterations for DLMF')
                    dlmf_action = np.zeros(self.num_agents)
                    store = False
                #dlmf_action = actions
                                                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(dlmf_action.reshape((-1,1)), step)
                                
                '''if np.all(obs < obs_low or obs > obs_high):
                    obs = self.envs.reset()
                    break'''
                                
                # Get average reward using consensus
                avg_reward_consensus, _, _, conv_steps = self.consensus.forward(rewards[0,:])                
                
                episode_reward += avg_reward_consensus[0]
                
                data = obs, avg_reward_consensus.reshape((1,-1)), dones, infos, values, dlmf_action.reshape((1,-1)), action_log_probs, rnn_states, rnn_states_critic
                
                # Check for constraint satisfaction
                #const = np.dot(self.envs.B, np.transpose(obs[0,:])) + self.envs.C
                #print('constraint = ', const)
                
                # insert data into buffer
                if store:
                    self.insert(data)
                self.step += 1
                            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            for count, info in enumerate(infos):
                                if 'individual_reward' in infos[count][agent_id].keys():
                                    idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": episode_reward}) #np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                else:
                    idv_rews = 0
                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                        idv_rews += np.mean(self.buffer[agent_id].rewards) * self.episode_length
                        
                    local_cost_new = self.envs.a * obs**2 + self.envs.b * obs + self.envs.c
                    print('local_cost_new = ', np.sum(local_cost_new))
                    #print('Episode return = ', episode_reward)
                    print('mean team reward = ', idv_rews/10)
                    # Check for constraint satisfaction
                    const = np.dot(self.envs.B, np.transpose(obs[0,:])) + self.envs.C
                    print('constraint = ', const)
                    
                    if self.cost_list is None:
                        self.cost_list = np.array(np.sum(local_cost_new))
                    else:
                        self.cost_list = np.hstack((self.cost_list, np.array(np.sum(local_cost_new))))

                    #np.save(self.plt_name+'.npy', self.reward_list)  # 3: GELU, 4: 3 and deeper NN (64,64, 64), 5:3 and lambd=0.98, 6:3 and lambd=0.95
                    np.save(os.path.join('results', self.plt_name + '.npy'), self.cost_list)
                        
                self.log_train(train_infos, total_num_steps)
                
            #obs = self.envs.reset()

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
                
        if self.env_name == 'TenNodeTwoConstraintEnv' or self.env_name == 'FourNodeOneConstraintEnv':
            for agent_id in range(self.num_agents):
                avg_state_consensus, obs_H2, obs_Hinfty, conv_steps = self.consensus.forward(obs[0,:], agent_id)
                observer_obs = np.array([[obs_H2, obs_Hinfty, 0, 0]])
                self.buffer[agent_id].share_obs[0] = observer_obs.copy()                
                self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
                
        else:   
            share_obs = []
            for o in obs:
                share_obs.append(list(chain(*o)))
            share_obs = np.array(share_obs)

            for agent_id in range(self.num_agents):
                if not self.use_centralized_V:
                    share_obs = np.array(list(obs[:, agent_id]))
                self.buffer[agent_id].share_obs[0] = share_obs.copy()
                self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)

            actions.append(action)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)        

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        dones_env = np.all(dones, axis=0)
        
        rnn_states[dones == True] = np.zeros(((dones_env == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones_env == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones_env == True).sum(), 1), dtype=np.float32)        

        avg_state_consensus, obs_H2, obs_Hinfty, conv_steps = self.consensus.forward(obs[0,:])
        avg_action_consensus, action_H2, action_Hinfty, conv_steps = self.consensus.forward(actions[0,:])

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            
            observer_obs_i = np.array([obs_H2[agent_id],obs_Hinfty[agent_id],action_H2[agent_id],action_Hinfty[agent_id]])
            self.buffer[agent_id].insert(observer_obs_i,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:,agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
