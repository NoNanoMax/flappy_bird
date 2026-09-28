import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent:
    """
    Network:  Shared(4→64→64) → Actor(64→2) + Critic(64→1)
    PPO with clipped objective + GAE.
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,       # GAE lambda
        clip_eps: float = 0.2,
        update_epochs: int = 8,
        ent_coef: float = 0.1,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.ent_coef = ent_coef

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ).to(self.device)
        self.actor = nn.Linear(hidden, 2).to(self.device)
        self.critic = nn.Linear(hidden, 1).to(self.device)

        self.optimizer = optim.Adam(
            list(self.shared.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )

        # episode buffer
        self.states: list[torch.Tensor] = []
        self.actions: list[int] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.values: list[torch.Tensor] = []
        self.dones: list[bool] = []


    def act(self, state: list[float]) -> int:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        h = self.shared(s)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(h).squeeze().detach()

        self.states.append(s.squeeze(0))
        self.actions.append(action.item())
        self.log_probs.append(log_prob.squeeze().detach())
        self.values.append(value)
        return action.item()


    def store(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones.append(done)


    def update(self):
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.log_probs).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        old_values = torch.stack(self.values).squeeze(-1)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        # GAE
        advantages, returns = self._compute_gae(rewards, old_values, dones)
        advantages = advantages / (advantages.std() + 1e-8)

        # PPO clipped update
        for _ in range(self.update_epochs):
            h = self.shared(states)
            logits = self.actor(h)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            new_values = self.critic(h).squeeze(-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(new_values, returns)

            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.shared.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()

        # clear buffer
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def _compute_gae(
        self, rewards, values, dones
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns."""
        n = len(rewards)
        advantages = torch.zeros(n, device=self.device)
        gae = torch.zeros((), device=self.device)
        returns = torch.zeros(n, device=self.device)

        next_value = 0.0
        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.lam * gae * non_terminal
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t].item()

        return advantages, returns


    def save(self, path: str):
        torch.save(
            {
                "shared": self.shared.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.shared.load_state_dict(ckpt["shared"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    def act_greedy(self, state: list[float]) -> int:
        """Deterministic action (for rendering)."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            h = self.shared(s)
            return self.actor(h).argmax(dim=1).item()
