import os
import sys
import time

import numpy as np
import torch

from agent import PPOAgent
from game import FlappyBird


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = FlappyBird(seed=seed)
    agent = PPOAgent()

    os.makedirs("checkpoints", exist_ok=True)
    scores: list[int] = []
    t0 = time.time()

    print(f"{'Ep':>5}  {'Score':>5}  {'Avg10':>6}  {'Avg100':>6}  {'Time':>6}")
    print("─" * 44)

    for ep in range(episodes):
        obs = env.reset()
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, done = env.step(action)
            agent.store(reward, done)

        agent.update()
        scores.append(env.score)

        if ep % 50 == 0 or ep == episodes - 1:
            a10 = np.mean(scores[-10:])
            a100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            elapsed = time.time() - t0
            print(f"{ep:>5}  {env.score:>5}  {a10:>6.1f}  {a100:>6.1f}  {elapsed:>5.0f}s")

        if (ep + 1) % 200 == 0:
            agent.save(f"checkpoints/model_ep{ep + 1:04d}.pt")

    agent.save("checkpoints/final.pt")
    print(f"\nDone!  Avg(last 100) = {np.mean(scores[-100:]):.1f}  "
          f"Max = {max(scores)}  Time = {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
