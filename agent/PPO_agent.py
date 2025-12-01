import os
from turtle import update
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from datetime import datetime




class PPOModel(torch.nn.Module):
    def __init__(self, obs_dim, n_actions,is_discrete):
        super().__init__()

        self.n_actions = n_actions
        self.input_dim = obs_dim
        self.is_discrete = is_discrete
        # self._hidden_size1 = 128
        # self._hidden_size2 = 64
        #agent_config = self.load_agent_file(config_file)
        #self._load_params(self, agent_config)

        #self.Encoder=CNNEncoder()


        # self.fc = nn.Sequential(
        # nn.Linear(self.input_dim, self._hidden_size),
        # nn.ReLU(),
        # nn.Linear(self._hidden_size, self._hidden_size),
        # nn.ReLU()
        # )

        self.actor = nn.Sequential(
        nn.Linear(self.input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, self.n_actions)
        )

        self.critic = nn.Sequential(
        nn.Linear(self.input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
        )

        # Log std for continuous actions
        if not self.is_discrete:
            self.log_std = nn.Parameter(torch.zeros(self.n_actions))
    

    def forward(self, obs: torch.Tensor):
        # h = self.fc(x)
        
        value = self.critic(obs).squeeze(-1)
        if self.is_discrete:
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
        else:
            mu = self.actor(obs)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)

        return dist, value
    




class PPOAgent:
    NAME = "PPO"

    def __init__(self, env, model, actor_lr=3e-4, critic_lr=1e-4, gamma=0.99, clip_eps=0.3, 
                 value_coef=0.5, entropy_coef=0.02, batch_size=512, epochs=10, device='cpu'):
        self._env = env
        self._device = device
        self.model = model.to(self._device)
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self._actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=self._critic_lr)
        self._gamma = gamma
        self._clip_eps = clip_eps
        self._value_coef = value_coef
        self._entropy_coef = entropy_coef
        self._batch_size = batch_size
        self._epochs = epochs
        self._iter = 0
        self._sample_count = 0

        

        return
    
    def get_action(self, obs: torch.Tensor):
        """
        Sample an action for environment interaction
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)  # add batch dim
        # print(f"obs: {obs}")
        dist, value = self.model.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if not self.model.is_discrete:
            log_prob = log_prob.sum(axis=-1)  # sum for multi-dimensional continuous actions
        #print("Sampled action:", action.cpu().numpy())
        return action, log_prob, value, dist

    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Compute log probability, entropy, and value for PPO update
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)  # add batch dim
        dist, value = self.model.forward(obs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        if not self.model.is_discrete:
            log_prob = log_prob.sum(axis=-1)
            entropy = entropy.sum(axis=-1)
        return log_prob, entropy, value
    
    def compute_returns(self, rewards, dones, last_value):
        """
        Compute discounted returns
        """
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self._gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self._device)
    
    def collect_trajectories(self, num_steps):
        """
        Collect rollout trajectories from the environment
        """
        obs_list, action_list, log_prob_list, reward_list, done_list, value_list = [], [], [], [], [], []

        obs = self._env.reset()
        for i in range(num_steps):
            with torch.no_grad():
                action, log_prob, value, dist = self.get_action(obs)

            next_obs, reward, done, _ = self._env.step(dist)
            
            obs_list.append(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            #action_list.append(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
            action_list.append(torch.tensor(action, dtype=torch.float32))
            log_prob_list.append(log_prob)
            reward_list.append(torch.tensor(reward, dtype=torch.float32))
            done_list.append(done)
            value_list.append(value.unsqueeze(0))

            obs = next_obs
            #do not break on done to collect fixed length trajectories
            if done:
                obs = self._env.reset()
        
        obs_tensor = torch.cat(obs_list).to(self._device)
        actions_tensor = torch.cat(action_list).to(self._device)
        log_probs_tensor = torch.cat(log_prob_list).to(self._device)
        values_tensor = torch.cat(value_list).to(self._device)
        rewards_tensor = torch.tensor(reward_list, dtype=torch.float32, device=self._device)
        dones_tensor = torch.tensor(done_list, dtype=torch.float32, device=self._device)

        # Compute last value for bootstrapping
        with torch.no_grad():
            # if the last rollout step ended an episode we shouldn't bootstrap
            if done:
                last_value = torch.tensor(0.0, dtype=torch.float32, device=self._device)
            else:
                # convert obs to tensor, add batch dim, move to device
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
                _, last_value = self.model.forward(obs_t)
                last_value = last_value.squeeze(0)  # scalar tensor on device

        returns = self.compute_returns(rewards_tensor, dones_tensor, last_value)
        #advantages = returns - values_tensor
        advantages = returns - values_tensor.squeeze(-1)  # shape [2048]
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return obs_tensor, actions_tensor, log_probs_tensor, returns, advantages
    
    def ppo_update(self, obs, actions, log_probs_old, returns, advantages):
        """
        Update PPO policy using mini-batches
        """
        obs = obs.to(self._device)
        actions = actions.to(self._device)
        log_probs_old = log_probs_old.to(self._device)
        returns = returns.to(self._device)
        advantages = advantages.to(self._device)
        # print('obs', obs.shape)
        # print('actions', actions.shape)
        # print('log_probs_old', log_probs_old.shape)
        # print('returns', returns.shape)
        # print('advantages', advantages.shape)
        for i in range(self._epochs):
            idxs = np.arange(len(obs))
            np.random.shuffle(idxs)
            for start in range(0, len(obs), self._batch_size):
                end = start + self._batch_size
                batch_idx = idxs[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_log_probs_old = log_probs_old[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                log_probs, entropy, values = self.evaluate_action(batch_obs, batch_actions)
                # print('log_probs', log_probs.shape)
                # print('batch_log_probs_old', batch_log_probs_old.shape)

                # PPO ratio
                ratios = torch.exp(log_probs - batch_log_probs_old)

                # Clipped surrogate loss
                # print('ratio', ratios.shape)
                # print('batch_advantages', batch_advantages.shape)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self._clip_eps, 1.0 + self._clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self._entropy_coef * entropy.mean()

                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)

                # Backprop and optimize actor
                self.actor_optimizer.zero_grad()
                policy_loss.backward()  
                self.actor_optimizer.step()

                # Backprop and optimize critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

                # Total loss
                # loss = policy_loss + self._value_coef * value_loss - self._entropy_coef * entropy.mean()

                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

    def train(self, num_steps_per_update=2048, total_updates=1000):
        """
        Main training loop
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("./output", exist_ok=True)
        log_file = os.path.join("./output", f"trajectory_log_{timestamp}.txt")
        model_file = os.path.join("output", f"ppo_model_{timestamp}.pth")
        for update in range(total_updates):
            obs, actions, log_probs, returns, advantages = self.collect_trajectories(num_steps_per_update)
            self.ppo_update(obs, actions, log_probs, returns, advantages)

            # Save model after each trajectory
            torch.save(self.model.state_dict(), model_file)
            # Compute trajectory metrics
            total_reward = returns.sum().item()
            traj_length = len(returns)
            print(f"Update {update+1}/{total_updates} completed"+f" AVG Reward {total_reward/len(returns):.2f}")
            with open(log_file, "a") as f:
                f.write(f"Update {update+1}, AVG Reward {total_reward}/{len(returns):.2f}, Length: {traj_length}, Sample: {update*num_steps_per_update}\n")
