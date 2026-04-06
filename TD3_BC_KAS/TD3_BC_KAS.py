import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC_KAS(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        alpha=2.5,
        kalman_q=1.0,
        kalman_r=5e-2,
        gate_threshold=0.2,
        gate_r_mult=20.0,
        blend_beta=0.90,
        kalman_p_init=1.0,
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = float(max_action)
        self.action_dim = int(action_dim)

        self.discount = float(discount)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.total_it = 0

        self.kalman_q = float(kalman_q)
        self.kalman_r = float(kalman_r)
        self.gate_threshold = float(gate_threshold)
        self.gate_r_mult = float(gate_r_mult)
        self.blend_beta = float(blend_beta)
        self.kalman_p_init = float(kalman_p_init)

        self.filters = [
            KinematicKalmanFilter(q_sigma=self.kalman_q, r_sigma=self.kalman_r, p_init=self.kalman_p_init)
            for _ in range(self.action_dim)
        ]

    def reset_filters(self):
        for kf in self.filters:
            kf.reset()

    def select_action(self, state, dt=0.01, apply_kalman=True):
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
        z = self.actor(state_t).cpu().data.numpy().flatten()

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

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor_target(next_state).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % 2 == 0:
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / (Q.abs().mean().detach() + 1e-6)
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename, device_override=None):
        dev = device_override if device_override is not None else device
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=dev))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=dev))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=dev))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=dev))
        self.actor_target = copy.deepcopy(self.actor)


def compute_jerk(action_history):
    if len(action_history) < 3:
        return 0.0
    actions = np.asarray(action_history, dtype=np.float64)
    jerks = np.linalg.norm(actions[2:] - 2.0 * actions[1:-1] + actions[:-2], axis=1)
    return float(np.mean(jerks))