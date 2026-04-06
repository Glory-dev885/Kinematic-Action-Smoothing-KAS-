from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import update as awr_update_actor
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_q, update_v


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau),
        critic.params,
        target_critic.params,
    )
    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    target_critic: Model,
    batch: Batch,
    discount: float,
    tau: float,
    expectile: float,
    temperature: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(
        key, actor, target_critic, new_value, batch, temperature
    )
    new_critic, critic_info = update_q(critic, new_value, batch, discount)
    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info,
    }

class KinematicKalmanFilter:
    def __init__(self, q_sigma=1e-4, r_sigma=5e-2, p_init=1.0):
        self.dim_x = 3
        self.dim_z = 1

        self.q_sigma = float(q_sigma)
        self.r_sigma = float(r_sigma)
        self.p_init = float(p_init)

        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64) * self.p_init
        self.Q = np.eye(self.dim_x, dtype=np.float64) * self.q_sigma
        self.R_base = np.eye(self.dim_z, dtype=np.float64) * self.r_sigma
        self.H = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

        self.initialized = False

    def reset(self):
        self.x[:] = 0.0
        self.P = np.eye(self.dim_x, dtype=np.float64) * self.p_init
        self.initialized = False

    @staticmethod
    def F(dt: float) -> np.ndarray:
        dt = float(dt)
        return np.array(
            [
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def predict(self, dt: float):
        Fm = self.F(dt)
        self.x = Fm @ self.x
        self.P = Fm @ self.P @ Fm.T + self.Q

    def innovation(self, z: float) -> float:
        return float(z) - float((self.H @ self.x)[0, 0])

    def update(self, z: float, r_scale: float = 1.0) -> float:
        z = float(z)
        z_vec = np.array([[z]], dtype=np.float64)

        R = self.R_base * float(r_scale)

        y = z_vec - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        S = S + 1e-9 * np.eye(self.dim_z, dtype=np.float64)

        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x, dtype=np.float64) - K @ self.H) @ self.P

        return float(self.x[0, 0])


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        dropout_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
        opt_decay_schedule: str = "cosine",

        kalman_q: float = 1.0,
        kalman_r: float = 5e-2,
        gate_frac: float = 0.20,
        gate_r_mult: float = 20.0,
        blend_beta: float = 0.90,
        kalman_p_init: float = 1.0,
    ):

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=False,
        )

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(schedule_fn),
            )
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(
            actor_def,
            inputs=[actor_key, observations],
            tx=optimiser,
        )

        critic_def = value_net.DoubleCritic(hidden_dims)
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )

        value_def = value_net.ValueCritic(hidden_dims)
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr),
        )

        target_critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
        )

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

        self.action_dim = int(action_dim)
        self.max_action = 1.0
        self.kalman_q = float(kalman_q)
        self.kalman_r = float(kalman_r)
        self.gate_threshold = float(gate_frac) * self.max_action
        self.gate_r_mult = float(gate_r_mult)
        self.blend_beta = float(blend_beta)
        self.kalman_p_init = float(kalman_p_init)

        self.filters = [
            KinematicKalmanFilter(
                q_sigma=self.kalman_q,
                r_sigma=self.kalman_r,
                p_init=self.kalman_p_init,
            )
            for _ in range(self.action_dim)
        ]

    def reset_filters(self):
        for kf in self.filters:
            kf.reset()

    def sample_actions(
        self,
        observations: np.ndarray,
        temperature: float = 1.0,
        apply_kalman: bool = False,
        dt: float = 0.01,
    ) -> jnp.ndarray:
        rng, actions = policy.sample_actions(
            self.rng,
            self.actor.apply_fn,
            self.actor.params,
            observations,
            temperature,
        )
        self.rng = rng

        z = np.asarray(actions, dtype=np.float64)
        z = np.clip(z, -self.max_action, self.max_action)

        if not apply_kalman:
            return z.astype(np.float32)

        a_out = np.zeros_like(z, dtype=np.float64)

        for i in range(self.action_dim):
            zi = float(z[i])
            kf = self.filters[i]

            if not kf.initialized:
                kf.x[0, 0] = zi
                kf.x[1, 0] = 0.0
                kf.x[2, 0] = 0.0
                kf.initialized = True

            kf.predict(dt=float(dt))

            innov = abs(kf.innovation(zi))
            r_scale = self.gate_r_mult if innov > self.gate_threshold else 1.0

            p_f = kf.update(zi, r_scale=r_scale)

            beta = self.blend_beta
            a_blend = (1.0 - beta) * p_f + beta * zi
            a_out[i] = a_blend

        a_out = np.clip(a_out, -self.max_action, self.max_action)
        return a_out.astype(np.float32)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.value,
            self.target_critic,
            batch,
            self.discount,
            self.tau,
            self.expectile,
            self.temperature,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info