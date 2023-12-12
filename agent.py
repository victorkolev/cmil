from copy import copy
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common



class Agent(common.Module):
    def __init__(self, config, obs_space, act_space, step):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.tfstep = tf.Variable(int(self.step), tf.int64)
        self.wm = WorldModel(config, obs_space, self.tfstep)
        self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)

    @tf.function
    def policy(self, obs, state=None, mode="train"):
        obs = tf.nest.map_structure(tf.tensor, obs)
        tf.py_function(
            lambda: self.tfstep.assign(int(self.step), read_value=False), [], []
        )
        if state is None:
            latent = self.wm.rssm.initial(len(obs["is_first"]))
            action = tf.zeros((len(obs["is_first"]),) + self.act_space.shape)
            state = latent, action
        latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        sample = (mode == "train") or not self.config.eval_state_mean
        latent, _ = self.wm.rssm.obs_step(
            latent, action, embed, obs["is_first"], sample
        )
        feat = self.wm.rssm.get_feat(latent)
        if mode == "eval":
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
            noise = self.config.eval_noise
        elif mode == "train":
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        action = common.action_noise(action, noise, self.act_space)
        outputs = {"action": action}
        state = (latent, action)
        return outputs, state

    @tf.function
    def init_wm(self, data, state=None):
        state, outputs, _ = self.wm.train(data, state)
        return state

    @tf.function
    def init_bc(self, expert_data, state=None):
        state, outputs, _ = self.wm.train(expert_data, state)
        bc_data = {
            'feat': tf.stop_gradient(outputs['feat']),
            'action': expert_data['action'], 
            }
        metrics = self._task_behavior.init_bc(bc_data)
        return state, metrics


    @tf.function
    def train(self, train_data, expert_data, state=None):
        metrics = {}

        state, outputs, mets = self.wm.train(train_data, state)
        metrics.update(mets)
        start = outputs['post']
        start["action"] = train_data['action']

        expert_data = self.wm.preprocess(expert_data)
        expert_embed = self.wm.encoder(expert_data)
        expert_post, _ = self.wm.rssm.observe(
            expert_embed, expert_data["action"], expert_data["is_first"], None
        )

        expert_post["feat"] = self.wm.rssm.get_feat(expert_post)
        expert_post["action"] = expert_data["action"]

        metrics.update(self.wm.imagine_from_expert(expert_post, self.config.imag_horizon))

        metrics.update(
            self._task_behavior.train(
                self.wm, start, expert_post, train_data["is_terminal"]
            )
        )
        return state, metrics

    @tf.function
    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads["decoder"].cnn_keys:
            name = key.replace("/", "_")
            report[f"openl_{name}"] = self.wm.video_pred(data, key)
        return report


class WorldModel(common.Module):
    def __init__(self, config, obs_space, tfstep):
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.tfstep = tfstep
        self.rssm = common.EnsembleRSSM(**config.rssm)
        self.encoder = common.Encoder(shapes, **config.encoder)
        self.heads = {}
        self.heads["decoder"] = common.Decoder(shapes, **config.decoder)
        if config.pred_discount:
            self.heads["discount"] = common.MLP([], **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name

        self.model_opt = common.Optimizer("model", **config.model_opt)

    def train(self, data, state=None):
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(data, state)
        modules = [self.encoder, self.rssm, *self.heads.values()]
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {"kl": kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = name in self.config.grad_heads
            inp = feat if grad_head else tf.stop_gradient(feat)
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = tf.cast(dist.log_prob(data[key]), tf.float32)
                likes[key] = like
                losses[key] = -like.mean()
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        outs = dict(
            embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics = {f"{name}_loss": value for name, value in losses.items()}
        metrics["model_kl"] = kl_value.mean()
        metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean()
        metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean()
        metrics["model_unc"] = prior["ensemble_mean"].std(-2).mean(-1).mean()
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def imagine_from_expert(self, expert, horizon):
        metrics = {}
        seq = {k: [v[:, :-horizon-1]] for k, v in expert.items()}
        for t in range(horizon):
            action = expert['action'][:, 1+t:-(horizon-t)]

            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.rssm.get_feat(state)
            for key, value in {
                **state,
                "action": action,
                "feat": feat,
            }.items():
                seq[key].append(value)

        seq = {k: tf.stack(v, 0) for k, v in seq.items()}
        eps = seq['ensemble_mean'].std(-2).mean()
        metrics['uncertainty_estimator'] = eps

        return metrics

    def imagine(self, policy, start, is_terminal, horizon):
        start["feat"] = self.rssm.get_feat(start)
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = policy(tf.stop_gradient(seq["feat"][-1])).sample()
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.rssm.get_feat(state)
            for key, value in {
                **state,
                "action": action,
                "feat": feat,
            }.items():
                seq[key].append(value)
        seq = {k: tf.stack(v, 0) for k, v in seq.items()}
        if "discount" in self.heads:
            disc = self.heads["discount"](seq["feat"]).mean()
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
                true_first *= self.config.discount
                disc = tf.concat([true_first[None], disc[1:]], 0)
        else:
            disc = self.config.discount * tf.ones(seq["feat"].shape[:-1])
        seq["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq["weight"] = tf.math.cumprod(
            tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0
        )
        return seq

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            if value.dtype == tf.uint8:
                value = value.astype(dtype) / 255.0 - 0.5
            obs[key] = value
        obs["discount"] = 1.0 - obs["is_terminal"].astype(dtype)
        obs["discount"] *= self.config.discount
        if obs['image'].shape[-3:] == (128, 128, 3):
            obs['image'] = obs['image'][..., ::2, ::2, :]
        assert obs['image'].shape[-3:] == (64, 64, 3), obs['image'].shape
        return obs

    @tf.function
    def video_pred(self, data, key):
        decoder = self.heads["decoder"]
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data["action"][:6, 5:], init)
        openl = decoder(self.rssm.get_feat(prior))[key].mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = tf.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


class ActorCritic(common.Module):
    def __init__(self, config, act_space, tfstep):
        self.config = config
        self.act_space = act_space
        self.tfstep = tfstep
        discrete = hasattr(act_space, "n")
        if self.config.actor.dist == "auto":
            self.config = self.config.update(
                {"actor.dist": "onehot" if discrete else "trunc_normal"}
            )
        if self.config.actor_grad == "auto":
            self.config = self.config.update(
                {"actor_grad": "reinforce" if discrete else "dynamics"}
            )
        self.actor = common.MLP(act_space.shape[0], **self.config.actor)
        self.critics = [
            common.MLP([], **self.config.critic) for _ in range(self.config.num_critics)
        ]
        if self.config.slow_target:
            self.target_critics = [
                common.MLP([], **self.config.critic)
                for _ in range(self.config.num_critics)
            ]
            self._updates = tf.Variable(0, tf.int64)
        else:
            self.target_critics = self.critics
        if config.wgan:
            discriminator_dict = {k: v for k, v in self.config.discriminator.items()}
            discriminator_dict['dist'] = 'mse'
        else: discriminator_dict = self.config.discriminator
        self.discriminator = common.MLP([], **self.config.discriminator)
        self.discriminator_noise = self.config.discriminator_noise

        self.opt_configs = {
                "actor": {k: v for k, v in self.config.actor_opt.items()},
                "critic": {k: v for k, v in self.config.actor_opt.items()},
                "discriminator": {k: v for k, v in self.config.actor_opt.items()},
                }

        self.actor_opt = common.Optimizer("actor", **self.opt_configs["actor"])
        self.bc_opt = common.Optimizer("bc", lr=2e-4, opt='sgd')
        self.critic_opt = common.Optimizer("critic", **self.opt_configs['critic'])
        opt_dict = self.opt_configs['discriminator'] if not config.wgan else config.wgan_opt
        self.discriminator_opt = common.Optimizer(
            "discriminator", **opt_dict
        )

    def init_bc(self, expert):
        metrics = {}
        with tf.GradientTape(persistent=True) as tape:
            lp = self.actor(expert["feat"][:, :-1])
            bc_loss = -lp.log_prob(expert["action"][:, 1:]).mean()
            metrics['bc_loss'] = bc_loss
        self.bc_opt(tape, bc_loss, self.actor)
        return metrics

    def train(self, world_model, start, expert, is_terminal):
        """
        start: dict, k: [batch, time, ...]
        expert: dict, k: [batch, time, ...]
        """
        metrics = {}
        hor = self.config.imag_horizon

        start["ensemble_mean"] = tf.repeat(
            tf.expand_dims(start["mean"], -2),
            repeats=world_model.rssm._ensemble,
            axis=-2,
        )
        start["ensemble_std"] = tf.repeat(
            tf.expand_dims(start["std"], -2),
            repeats=world_model.rssm._ensemble,
            axis=-2,
        )
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with tf.GradientTape(persistent=True) as tape:
            seq = world_model.imagine(self.actor, start, is_terminal, hor)
            # seq: [horizon, batch, time, ...]

            # Discriminator Learning
            feat_expert_dist = tf.concat(
                [expert["feat"][:, :-1], expert["action"][:, 1:]], axis=-1
            )
            feat_policy_dist = tf.concat([seq["feat"][:-1], seq["action"][1:]], axis=-1)

            d = lambda x: self.discriminator(tf.stop_gradient(x)).mean()
            expert_discriminator_reward = d(feat_expert_dist).mean(1)
            policy_disciminator_reward = d(feat_policy_dist).mean(0).mean(1)

            metrics['discriminator_expert_reward'] = expert_discriminator_reward.mean()
            metrics['discriminator_policy_reward'] = policy_disciminator_reward.mean()
            metrics['discriminator_gap'] = (expert_discriminator_reward - policy_disciminator_reward).mean()

            feat_expert_dist += self.discriminator_noise * tf.random.normal(shape=feat_expert_dist.shape)
            feat_policy_dist += self.discriminator_noise * tf.random.normal(shape=feat_policy_dist.shape)

            expert_d = self.discriminator(feat_expert_dist)
            policy_d = self.discriminator(feat_policy_dist)

            if self.config.wgan:
                expert_loss = expert_d.mean().mean()
                policy_loss = -policy_d.mean().mean()
            else:
                expert_loss = expert_d.log_prob(tf.ones_like(expert_d.mean())).mean()
                policy_loss = policy_d.log_prob(tf.zeros_like(policy_d.mean())).mean()
            discriminator_loss = -(expert_loss + policy_loss)

            metrics["discriminator_expert_mean"] = expert_d.mean().mean()
            metrics["discriminator_expert_max"] = expert_d.mean().max()
            metrics["discriminator_expert_min"] = expert_d.mean().min()

            metrics["discriminator_policy_mean"] = policy_d.mean().mean()
            metrics["discriminator_policy_max"] = policy_d.mean().max()
            metrics["discriminator_policy_min"] = policy_d.mean().min()

            metrics["discriminator_loss"] = discriminator_loss

            # Reward computation
            disag_rew = seq["ensemble_mean"].std(-2).mean(-1)

            data_dist = tf.concat(
                [seq["feat"][0, :, :-1], seq["action"][0, :, 1:]], axis=-1
            )
            data_reward = self.discriminator(data_dist).mean()
            model_reward = policy_d.mean() + self.config.unc_penalty * disag_rew[1:]

            disag_rew = disag_rew[1:]
            metrics["disag_rew_mean"] = disag_rew.mean()
            metrics["disag_rew_max"] = disag_rew.max()
            metrics["disag_rew_min"] = disag_rew.min()

            disc = tf.cast(seq["discount"], tf.float32)
            critics = [
                critic(tf.concat([seq["feat"][:-1], seq["action"][1:]], axis=-1))
                for critic in self.critics
            ]
            values = [critic.mode() for critic in critics]
            value = tf.math.reduce_min(tf.stack(values), axis=0)

            # Skipping last time step because it is used for bootstrapping.
            MCReturn = common.lambda_return(
                model_reward[:-1],
                value[:-1],
                disc[1:-1],
                bootstrap=value[-1],
                lambda_=self.config.discount_lambda,
                axis=0,
            )

            metrics.update(
                {f"critic_{i}": value.mean() for i, value in enumerate(values)}
            )
            metrics.update(
                {
                    f"critic_real_data_{i}": value[0].mean()
                    for i, value in enumerate(values)
                }
            )
            metrics.update(
                {
                    f"critic_model_data_{i}": value[1:].mean()
                    for i, value in enumerate(values)
                }
            )
            metrics["critic"] = value.mean()
            metrics["MCReturn"] = MCReturn.mean()

            policy = self.actor(tf.stop_gradient(seq["feat"][:-2]))
            objective = (1.0 - self.config.discount_lambda) * value[
                :-1
            ] + self.config.discount_lambda * MCReturn
            ent = policy.entropy()
            ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
            objective += ent_scale * ent
            weight = tf.stop_gradient(seq["weight"])
            actor_loss = -(weight[:-2] * objective).mean()

            metrics["actor_ent"] = ent.mean()
            metrics["actor_ent_scale"] = ent_scale
            metrics["actor_loss"] = actor_loss

            # BC regularization
            lp = self.actor(expert["feat"][:, :-1])
            bc_loss = -lp.log_prob(expert["action"][:, 1:]).mean()
            actor_loss += self.config.bc_scale * bc_loss

            metrics["bc_loss"] = bc_loss

            # Critics loss
            critics = [
                critic(tf.concat([seq["feat"][:-2], seq["action"][1:-1]], axis=-1))
                for critic in self.critics
            ]
            target_values = [
                critic(tf.concat([seq["feat"][:-1], seq["action"][1:]], axis=-1)).mode()
                for critic in self.target_critics
            ]
            index = tf.random.uniform((), 0, len(self.target_critics), tf.int32)
            target_value = tf.stack(target_values)[index]

            target_MCReturn = common.lambda_return(
                model_reward[:-1],
                target_value[:-1],
                disc[1:-1],
                bootstrap=target_value[-1],
                lambda_=self.config.discount_lambda,
                axis=0,
            )
            target = tf.stop_gradient(target_MCReturn)

            weight = tf.stop_gradient(seq["weight"][1:-1])
            critic_losses_model_data = [
                -(critic.log_prob(target) * weight).mean() for critic in critics
            ]
            critic_loss_model_data = tf.reduce_mean(critic_losses_model_data)
            metrics.update(
                {
                    f"critic_loss_model_data_{i}": l
                    for i, l in enumerate(critic_losses_model_data)
                }
            )
            metrics["critic_loss_model_data"] = critic_loss_model_data

            data_critics = [
                critic(
                    tf.concat(
                        [seq["feat"][0, :, :-2], seq["action"][0, :, 1:-1]], axis=-1
                    )
                )
                for critic in self.critics
            ]

            data_target = data_reward[:, :-1] + disc[0, :, 1:-1] * (
                (1.0 - self.config.discount_lambda) * target_value[0, :, 1:-1]
                + self.config.discount_lambda * target_MCReturn[0, :, 1:-1]
            )

            critic_losses_real_data = [
                -(critic.log_prob(data_target)).mean() for critic in data_critics
            ]
            critic_loss_real_data = tf.reduce_mean(critic_losses_real_data)
            metrics.update(
                {
                    f"critic_loss_real_data_{i}": l
                    for i, l in enumerate(critic_losses_real_data)
                }
            )
            metrics["critic_loss_model_data"] = critic_loss_real_data

            critic_loss = critic_loss_model_data + critic_loss_real_data
            metrics["critic_loss_joint"] = critic_loss

        metrics.update(
            self.discriminator_opt(tape, discriminator_loss, self.discriminator)
        )
        metrics.update(self.actor_opt(tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(tape, critic_loss, self.critics))

        if self.config.wgan:
            for w in self.discriminator.trainable_variables:
                w.assign(tf.clip_by_value(w, -self.config.wgan_clip, self.config.wgan_clip))

        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = (
                    1.0
                    if self._updates == 0
                    else float(self.config.slow_target_fraction)
                )
                for critic, target_critic in zip(self.critics, self.target_critics):
                    for s, d in zip(critic.variables, target_critic.variables):
                        d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)
