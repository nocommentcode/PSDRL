import os
import argparse

import numpy as np
from ruamel.yaml import YAML
import gym

from PSDRL.common.utils import generate_diversity_video_frames, init_env, load
from PSDRL.logging.logger import Logger
from PSDRL.logging import data_manager_factory
from PSDRL.agent.psdrl import PSDRL

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from PSDRL.common.utils import preprocess_image
import torch


class Tests:
    def __init__(
        self, env: gym.Env, agent: PSDRL, config: dict, logger: Logger
    ) -> None:
        self.env = env
        self.agent = agent
        self.logger = logger
        self.stored_prev_state = None

        # test episode
        self.test = config["experiment"]["test"]
        self.test_freq = config["experiment"]["test_freq"]
        self.time_limit = config["experiment"]["time_limit"]

        # rollouts
        self.gen_rollouts = config["logging"]["n_rollouts"] > 0
        self.rollout_freq = config["logging"]["rollout_freq"]
        self.n_rollouts = config["logging"]["n_rollouts"]

        # diversity
        self.gen_diversity = config["logging"]["n_diversity"] > 0
        self.diversity_freq = config["logging"]["diveristy_freq"]
        self.n_diversity = config["logging"]["n_diversity"]

    def store_prev_state(self):
        self.stored_prev_state = self.agent.model.prev_state

    def restore_prev_state(self):
        self.agent.model.set_hidden_state(self.stored_prev_state)

    def run_tests(self, experiment_step: int, episode: int):
        self.store_prev_state()

        if self.test and experiment_step % self.test_freq == 0:
            test_reward = self.run_test_episode()
            print(
                f"Episode {episode}, Timestep {experiment_step}, Test Reward {test_reward}"
            )
            self.logger.log_episode(
                experiment_step,
                train_reward=np.nan,
                test_reward=test_reward,
                epsilon=self.agent.epsilon,
            )

        if self.gen_rollouts and experiment_step % self.rollout_freq == 0:
            rollouts = [self.rollout_predictions() for _ in range(self.n_rollouts)]
            self.logger.log_rollout(rollouts, experiment_step)

        self.restore_prev_state()

    def run_test_episode(self):
        current_observation, _ = self.env.reset()
        episode_step = 0
        episode_reward = 0
        done = False
        while not done:
            action = self.agent.select_action(current_observation, episode_step)
            observation, reward, done, _, _ = self.env.step(action)
            episode_reward += reward
            current_observation = observation
            episode_step += 1
            done = done or episode_step == self.time_limit
        return episode_reward

    def rollout_predictions(self):
        with torch.no_grad():
            obs, _ = self.env.reset()

            # play episode on policy
            pred_states, actions, pred_rewards, pred_terminals = (
                self.play_through_episode(obs, 50)
            )

            # get actual episode
            actual_states = [obs]
            acutal_reward = [0]
            acutal_terminal = [0]
            done = False
            for a in actions:
                obs, reward, done, _, _ = self.env.step(a)
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

    def play_through_episode(
        self,
        initial_state: np.ndarray,
        num_steps: int,
    ):
        def embed_obs(obs):
            obs = preprocess_image(obs)
            obs = torch.from_numpy(obs).float().to(self.agent.device)
            return self.agent.model.embed_observation(obs)

        states = [initial_state]
        actions = []
        rewards = [0]
        terminals = [0]

        self.agent.model.reset_hidden_state()
        s = embed_obs(initial_state)
        for _ in range(num_steps):
            s_, r, t, h = self.agent.model.predict(s, self.agent.model.prev_state)
            a = self.agent._select_action(s)

            self.agent.model.set_hidden_state(h[a])
            s = s_[a].unsqueeze(0)
            r = r[a]
            t = t[a]

            states.append(self.agent.model.decode_observation(s).detach().cpu().numpy())
            actions.append(a.item())
            rewards.append(r.item())
            terminals.append(t.item())

        return states, actions, rewards, terminals


def run_experiment(
    env: gym.Env,
    agent: PSDRL,
    logger: Logger,
    steps: int,
    time_limit: int,
    save: bool,
    save_freq: int,
    tests: Tests,
):
    ep = 0
    experiment_step = 0

    while experiment_step < steps:
        episode_step = 0
        episode_reward = 0

        current_observation, _ = env.reset()
        done = False
        while not done:
            if ep:
                tests.run_tests(experiment_step, ep)

            action = agent.select_action(current_observation, episode_step)
            observation, reward, done, _, _ = env.step(action)
            done = done or episode_step == time_limit
            agent.update(
                current_observation,
                action,
                reward,
                observation,
                done,
                ep,
                experiment_step,
            )

            episode_reward += reward
            current_observation = observation
            episode_step += 1
            experiment_step += 1

            if ep and save and experiment_step % save_freq == 0:
                logger.data_manager.save(agent, experiment_step)

        ep += 1
        print(
            f"Episode {ep}, Timestep {experiment_step}, Train Reward {episode_reward}"
        )

        logger.log_episode(
            experiment_step,
            train_reward=episode_reward,
            test_reward=np.nan,
            epsilon=agent.epsilon,
        )


def main(config: dict):

    data_manager = data_manager_factory(config)
    logger = Logger(data_manager)
    exp_config = config["experiment"]

    env, actions, test_env = init_env(
        exp_config["suite"], exp_config["env"], exp_config["test"]
    )

    agent = PSDRL(config, actions, logger, config["experiment"]["seed"])
    if config["load"]:
        load(agent, config["load_dir"])

    tests = Tests(test_env, agent, config, logger)

    run_experiment(
        env,
        agent,
        logger,
        exp_config["steps"],
        exp_config["time_limit"],
        config["logging"]["save_model"],
        config["logging"]["save_freq"],
        tests,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml = YAML(typ="rt")
        config = yaml.load(f)
        config["experiment"]["env"] = args.env
        config["experiment"]["seed"] = args.seed

    main(config)
