import os
import argparse

import numpy as np
from ruamel.yaml import YAML
import gym

from PSDRL.common.utils import (
    generate_diversity_video_frames,
    init_env,
    load,
    set_seeds,
)
from PSDRL.logging.logger import Logger
from PSDRL.logging import data_manager_factory
from PSDRL.agent.psdrl import PSDRL

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_test_episode(env: gym.Env, agent: PSDRL, time_limit: int):
    current_observation, _ = env.reset()
    episode_step = 0
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(current_observation, episode_step)
        observation, reward, done, _, _ = env.step(action)
        episode_reward += reward
        current_observation = observation
        episode_step += 1
        done = done or episode_step == time_limit
    return episode_reward


def run_experiment(
    env: gym.Env,
    agent: PSDRL,
    logger: Logger,
    test_env: gym.Env,
    steps: int,
    test: int,
    test_freq: int,
    time_limit: int,
    save: bool,
    save_freq: int,
    gen_rollouts: bool,
    rollout_freq: int,
    n_rollouts: int,
    gen_pred_diversity: bool,
    diversity_freq: int,
    n_diversity: int,
):
    ep = 0
    experiment_step = 0

    while experiment_step < steps:
        episode_step = 0
        episode_reward = 0

        current_observation, _ = env.reset()
        done = False
        while not done:
            agent.eval()
            if test and experiment_step % test_freq == 0:
                test_reward = run_test_episode(test_env, agent, time_limit)
                print(
                    f"Episode {ep}, Timestep {experiment_step}, Test Reward {test_reward}"
                )
                logger.log_episode(
                    experiment_step,
                    train_reward=np.nan,
                    test_reward=test_reward,
                    epsilon=agent.epsilon,
                )

            if gen_rollouts and experiment_step % rollout_freq == 0:
                rollouts = [
                    agent.rollout_predictions(test_env) for _ in range(n_rollouts)
                ]
                logger.log_rollout(rollouts, experiment_step)

            if gen_pred_diversity and experiment_step % diversity_freq == 0:
                trajectories = agent.check_prediction_diversity(test_env, n_diversity)
                frames = generate_diversity_video_frames(trajectories)
                logger.log_diversity(frames, experiment_step)

            agent.train()
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

    set_seeds(exp_config["seed"], env, test_env)

    agent = PSDRL(config, actions, logger, config["experiment"]["seed"])
    if config["load"]:
        load(agent, config["load_dir"])

    run_experiment(
        env,
        agent,
        logger,
        test_env,
        exp_config["steps"],
        exp_config["test"],
        exp_config["test_freq"],
        exp_config["time_limit"],
        config["logging"]["save_model"],
        config["logging"]["save_freq"],
        config["logging"]["n_rollouts"] > 0,
        config["logging"]["rollout_freq"],
        config["logging"]["n_rollouts"],
        config["logging"]["n_diversity"] > 0,
        config["logging"]["diveristy_freq"],
        config["logging"]["n_diversity"],
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
