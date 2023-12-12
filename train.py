import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MUJOCO_GL"] = "egl"
logging.getLogger().setLevel("ERROR")


import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf

import agent
import common


def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    parsed, remaining = common.Flags(configs=["defaults"]).parse(known_only=True)
    config = common.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    expertdir = pathlib.Path(config.expertdir).expanduser()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    datadir = pathlib.Path(logdir / "train_episodes").expanduser()
    datadir.mkdir(parents=True, exist_ok=True)
    from distutils.dir_util import copy_tree

    copy_tree(str(config.expertdir), str(datadir))

    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)


    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = "No GPU found. To actually train on CPU remove this assert."
    assert tf.config.experimental.list_physical_devices("GPU"), message
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras.mixed_precision import experimental as prec

        prec.set_policy(prec.Policy("mixed_float16"))
    train_replay = common.Replay(logdir / "train_episodes", **config.replay)
    expert_replay = common.Replay(expertdir, read_fn=read_fn, **config.replay)
    eval_replay = common.Replay(
        logdir / "eval_episodes",
        **dict(
            capacity=config.replay.capacity // 10,
            minlen=config.dataset.length,
            maxlen=config.dataset.length,
        ),
    )
    step = common.Counter(train_replay.stats["total_steps"])
    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    for idx, ep in enumerate(train_replay._complete_eps.values()):
        logger.video(f"demos/{idx}", ep['image'])
        if idx > 25: break
    logger.write()

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.eval_every)
    should_video_eval = common.Every(config.eval_every)

    make_env = functools.partial(common.make_env, config=config)
    print("Create envs.")
    num_eval_envs = min(config.envs, config.eval_eps)
    train_envs = [make_env("train") for _ in range(config.envs)]
    eval_envs = [make_env("eval") for _ in range(num_eval_envs)]

    def per_episode(ep, mode):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f"sum_{mode}_{key}", ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f"mean_{mode}_{key}", ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f"max_{mode}_{key}", ep[key].max(0).mean())
        should = {"train": should_video_train, "eval": should_video_eval}[mode]
        if should(step):
            for key in config.log_keys_video:
                logger.video(f"{mode}_policy_{key}", ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space
    train_driver = common.Driver(train_envs)
    train_driver.on_episode(lambda ep: per_episode(ep, mode="train"))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    eval_driver = common.Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode="eval"))
    eval_driver.on_episode(eval_replay.add_episode)

    random_agent = common.RandomAgent(act_space)
    while eval_replay._loaded_episodes < 1:
        eval_driver(random_agent, episodes=1)

    print("Create agent.")
    train_dataset = iter(train_replay.dataset(**config.dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    expert_dataset = iter(expert_replay.dataset(**config.dataset))
    eval_dataset = iter(eval_replay.dataset(**config.dataset))

    agnt = agent.Agent(config, obs_space, act_space, step)
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(train_dataset), next(expert_dataset))
    train_policy = lambda *args: agnt.policy(
        *args, mode="train"
    )
    eval_policy = lambda *args: agnt.policy(*args, mode="train")

    if (logdir / "variables.pkl").exists():
        agnt.load(logdir / "variables.pkl")
    else:
        print("Pretrain agent.")
        for _ in range(config.pretrain):
            train_agent(next(train_dataset), next(expert_dataset))
        if config.pretrain_bc > 0:
            init_bc = common.CarryOverState(agnt.init_bc)
            for idxx in range(config.pretrain_bc):
                mets = init_bc(next(expert_dataset))
                step.increment()
                logger.add(mets, prefix="pretrain_bc")
                if idxx % 5 == 0: logger.write()
                if idxx % 50 == 0: eval_driver(eval_policy, episodes=config.eval_eps)
    if config.seed_steps > 0:
        # adds config.seed_steps to train buffer from BC policy
        train_driver(train_policy, steps=config.seed_steps)

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next(train_dataset), next(expert_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(report_dataset)), prefix="train")
            logger.write(fps=True)

    train_driver.on_step(train_step)

    while step < config.steps:
        logger.write()
        print("Start evaluation.")
        logger.add(agnt.report(next(eval_dataset)), prefix="eval")
        eval_driver(eval_policy, episodes=config.eval_eps)
        print("Start training.")
        train_driver(train_policy, steps=config.eval_every)
        agnt.save(logdir / "variables.pkl")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
