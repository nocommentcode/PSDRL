import gym
import torch
import numpy as np

from ..logging.logger import Logger
from ..common.replay import Dataset
from ..common.utils import preprocess_image
from ..networks.value import Network as ValueNetwork
from ..training.policy import PolicyTrainer
from ..common.settings import TP_THRESHOLD
from ..agent.agent_model import AgentModel
from ..agent.neural_linear_agent_model import NeuralLinearAgentModel
from ..agent.lp_bnn_agent_model import LPBNNAgentModel
import torch.nn as nn


class PSDRL(nn.Module):
    def __init__(self, config: dict, actions: list, logger: Logger, seed: int = None):
        super().__init__()
        self.device = "cpu" if not config["gpu"] else "cuda:0"
        self.random_state = np.random.RandomState(seed)

        self.num_actions = len(actions)
        self.actions = torch.tensor(actions).to(self.device)

        self.epsilon = config["algorithm"]["policy_noise_start"]
        epsilon_start = config["algorithm"]["policy_noise_start"]
        epsilon_end = config["algorithm"]["policy_noise_end"]
        n_steps = config["experiment"]["steps"]
        self.epsilon_decay_fact = (epsilon_end / epsilon_start) ** (1 / n_steps)

        self.update_freq = config["algorithm"]["update_freq"]
        self.warmup_length = config["algorithm"]["warmup_length"]
        self.warmup_freq = config["algorithm"]["warmup_freq"]
        self.discount = config["value"]["discount"]

        self.dataset = Dataset(
            logger,
            config["replay"],
            config["experiment"]["time_limit"],
            self.device,
            seed,
        )

        self.value_network = ValueNetwork(
            config["representation"]["embed_dim"],
            config["value"],
            self.device,
            config["transition"]["gru_dim"],
        )

        self.policy_trainer = PolicyTrainer(
            config["value"],
            config["transition"]["gru_dim"],
            self.value_network,
            self.device,
            config["replay"]["batch_size"],
            self.actions,
        )

        if config["algorithm"]["bayesian"] == "neural-linear":
            self.model = NeuralLinearAgentModel(config, self.device, self.actions)
        elif config["algorithm"]["bayesian"] == "lpbnn":
            self.model = LPBNNAgentModel(config, self.device, self.actions)
        elif config["algorithm"]["bayesian"] == "none":
            self.model = AgentModel(config, self.device, self.actions)
        else:
            raise ValueError(f"agent {config['algorithm']['bayesian']} not supported")

    def select_action(self, obs: np.array, step: int):
        """
        Reset the hidden state at the start of a new episode. Return a random action with a probability of epsilon,
        otherwise follow the current policy and sampled model greedily.
        """
        if step == 0:
            self.model.reset_hidden_state()

        if self.random_state.random() < self.epsilon:
            return self.random_state.choice(self.num_actions)

        obs = preprocess_image(obs)
        obs = torch.from_numpy(obs).float().to(self.device)
        obs = self.model.embed_observation(obs)
        try:
            return self._select_action(obs)
        except Exception as e:
            print(f"Error selecting action: {str(e)}")
            return self.random_state.choice(self.num_actions)

    def _select_action(self, obs: torch.tensor):
        """
        Return greedy action with respect to the current value network and all possible transitions predicted
        with the current sampled model (Equation 8).
        """
        states, rewards, terminals, h = self.model.predict(obs, self.model.prev_state)
        v = self.discount * (
            self.value_network.predict(torch.cat((states, h), dim=1))
            * (terminals < TP_THRESHOLD)
        )
        values = (rewards + v).detach().cpu().numpy()

        action = self.random_state.choice(np.where(np.isclose(values, max(values)))[0])

        self.model.set_hidden_state(h[action])

        return self.actions[action]

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay_fact

    def update(
        self,
        current_obs: np.array,
        action: int,
        rew: int,
        obs: np.array,
        done: bool,
        ep: int,
        timestep: int,
    ):
        """
        Add new transition to replay buffer and if it is time to update:
         - Update the representation model (Equation 1)
         - Update transition model (Equation 2) and terminal models (Equation 3).
         - Update posterior distributions model (Equation 4).
         - Sample new model from posteriors.
         - Update value network based on the new sampled model (Equation 5).
        """
        current_obs, obs = preprocess_image(current_obs), preprocess_image(obs)
        self.dataset.add_data(current_obs, action, obs, rew, done)
        update_freq = (
            self.update_freq if timestep > self.warmup_length else self.warmup_freq
        )
        self.update_epsilon()

        if ep and timestep % update_freq == 0:
            self.model.train_(self.dataset)
            self.policy_trainer.train_(self.model, self.dataset)
            self.dataset.logger.send_log(timestep)

    def play_through_episode(
        self,
        initial_state: np.ndarray,
        num_steps: int,
    ):
        def embed_obs(obs):
            obs = preprocess_image(obs)
            obs = torch.from_numpy(obs).float().to(self.device)
            return self.model.embed_observation(obs)

        states = [initial_state]
        actions = []
        rewards = [0]
        terminals = [0]

        self.model.reset_hidden_state()
        s = embed_obs(initial_state)
        for _ in range(num_steps):
            s_, r, t, _ = self.model.predict(s, self.model.prev_state)
            a = self._select_action(s)

            s = s_[a].unsqueeze(0)
            r = r[a]
            t = t[a]

            states.append(self.model.decode_observation(s).detach().cpu().numpy())
            actions.append(a.item())
            rewards.append(r.item())
            terminals.append(t.item())

        return states, actions, rewards, terminals

    def rollout_predictions(self, env: gym.Env, num_steps: int = 50):
        with torch.no_grad():
            obs, _ = env.reset()

            # play episode on policy
            pred_states, actions, pred_rewards, pred_terminals = (
                self.play_through_episode(obs, num_steps)
            )

            # get actual episode
            actual_states = [obs]
            acutal_reward = [0]
            acutal_terminal = [0]
            done = False
            for a in actions:
                obs, reward, done, _, _ = env.step(a)
                actual_states.append(obs)
                acutal_reward.append(reward)
                acutal_terminal.append(1 if done else 0)

                if done:
                    break

            return (
                (pred_states, actual_states),
                (pred_rewards, acutal_reward),
                (pred_terminals, acutal_terminal),
            )

    def check_prediction_diversity(
        self, env: gym.Env, num_samples: int = 5, num_steps: int = 500
    ):
        with torch.no_grad():
            obs, _ = env.reset()

            trajectories = []
            for _ in range(num_samples):
                self.model.resample_model()
                pred_traj, *_ = self.play_through_episode(obs.copy(), num_steps)
                trajectories.append(pred_traj)

            return trajectories
