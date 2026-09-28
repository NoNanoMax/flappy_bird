import random


class FlappyBird:
    """
    State:  [bird_y, bird_velocity, dist_to_next_pipe]  (3 floats)
    Action: 0 = nothing, 1 = flap
    Reward: +1 per pipe passed, -1 on death, 0 otherwise
    """

    def __init__(self, seed=None):
        # Dimensions
        self.width = 288
        self.height = 512
        self.ground_height = 80

        # Physics (per-frame values, designed for 60 fps)
        self.gravity = 0.5
        self.flap = -8.0

        # Pipes
        self.pipe_speed = 1.5
        self.pipe_gap = 150
        self.pipe_width = 52
        self.pipe_spacing = 150

        # Bird
        self.bird_x = 50
        self.bird_size = 16  # half-size (bird is 32×32)

        self.rng = random.Random(seed)

        # Mutable state
        self.bird_y = 0.0
        self.bird_velocity = 0.0
        self.pipes: list[list[float]] = []
        self.score = 0
        self.done = False



    def reset(self) -> list[float]:
        self.bird_y = self.height / 2
        self.bird_velocity = 0.0
        self.score = 0
        self.done = False
        self.pipes = [
            [self.width + 50, self.rng.randint(150, 350), False]
        ]
        return self._obs()

    def step(self, action: int) -> tuple[list[float], float, bool]:
        if self.done:
            return self._obs(), 0.0, True

    
        self.bird_velocity += self.gravity
        if action == 1:
            self.bird_velocity = self.flap
        self.bird_y += self.bird_velocity

    
        for p in self.pipes:
            p[0] -= self.pipe_speed

    
        if self.pipes[-1][0] < self.width - self.pipe_spacing:
            self.pipes.append(
                [self.pipes[-1][0] + self.pipe_spacing,
                 self.rng.randint(150, 350), False]
            )

    
        self.pipes = [p for p in self.pipes if p[0] + self.pipe_width > -10]

    
        reward = 0.0
        bs = self.bird_size

        # ground / ceiling
        if self.bird_y + bs >= self.height - self.ground_height:
            self.bird_y = self.height - self.ground_height - bs
            self.done = True
            reward = -1.0
        elif self.bird_y - bs <= 0:
            self.bird_y = bs
            self.done = True
            reward = -1.0

        # pipes
        if not self.done:
            for p in self.pipes:
                px, gc, scored = p
                gap_top = gc - self.pipe_gap // 2
                gap_bot = gc + self.pipe_gap // 2

                # horizontal overlap?
                if self.bird_x + bs > px and self.bird_x - bs < px + self.pipe_width:
                    # vertical: inside the gap?
                    if self.bird_y - bs < gap_top or self.bird_y + bs > gap_bot:
                        self.done = True
                        reward = -1.0
                        break

                # scoring: bird fully past the pipe
                if not scored and self.bird_x - bs > px + self.pipe_width:
                    p[2] = True
                    self.score += 1
                    reward = 1.0

        # dense reward: fly towards gap center
        target = self.pipes[0][1] if self.pipes else self.height / 2
        reward -= abs(self.bird_y - target) / self.height * 0.5
        # survival bonus
        if not self.done:
            reward += 0.1
        
        return self._obs(), reward, self.done


    def _obs(self) -> list[float]:
        dist = self.pipes[0][0] - self.bird_x
        gap_center = self.pipes[0][1]
        return [
            self.bird_y / self.height,
            self.bird_velocity / 10.0,
            dist / self.width,
            gap_center / self.height,
        ]
