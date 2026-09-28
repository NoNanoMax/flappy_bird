import os
from pathlib import Path

from agent import PPOAgent
from game import FlappyBird
from render import render_frame


def record_episode(agent: PPOAgent, env: FlappyBird, out_dir: str, seed: int, max_frames: int = 6000):
    """Record one full episode, saving each frame as PNG. Stops at max_frames."""
    os.makedirs(out_dir, exist_ok=True)

    obs = env.reset()
    step = 0

    while not env.done and step < max_frames:
        action = agent.act_greedy(obs)
        frame = render_frame(
            bird_y=env.bird_y,
            pipes=env.pipes,
            score=env.score,
            width=env.width,
            height=env.height,
            ground_h=env.ground_height,
            pipe_w=env.pipe_width,
            pipe_gap=env.pipe_gap,
            bird_x=env.bird_x,
            bird_size=env.bird_size,
        )
        frame.save(os.path.join(out_dir, f"{step:05d}.png"))
        step += 1

        obs, reward, done = env.step(action)

    for _ in range(10):
        frame = render_frame(
            bird_y=env.bird_y,
            pipes=env.pipes,
            score=env.score,
            width=env.width,
            height=env.height,
            ground_h=env.ground_height,
            pipe_w=env.pipe_width,
            pipe_gap=env.pipe_gap,
            bird_x=env.bird_x,
            bird_size=env.bird_size,
        )
        frame.save(os.path.join(out_dir, f"{step:05d}.png"))
        step += 1

    print(f"  {out_dir}: {step} frames, score={env.score}")
    return env.score


def main():
    agent = PPOAgent()
    agent.load("checkpoints/final.pt")

    base = Path("frames")
    base.mkdir(exist_ok=True)

    seeds = [1, 5, 10, 42]
    print(f"Recording {len(seeds)} episodes (max 600 frames each)...")
    scores = []
    for i, seed in enumerate(seeds):
        env = FlappyBird(seed=seed)
        out = str(base / f"ep{i:03d}_seed{seed}")
        score = record_episode(agent, env, out, seed)
        scores.append(score)

    print(f"\nScores: {scores}")
    print(f"Avg: {sum(scores)/len(scores):.1f}, Max: {max(scores)}")


if __name__ == "__main__":
    main()
