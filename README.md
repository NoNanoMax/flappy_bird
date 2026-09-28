# Flappy Bird RL

Нейросеть учится играть в Flappy Bird - PPO на PyTorch.

## Что внутри

| Файл | Описание |
|------|----------|
| `game.py` | Движок Flappy Bird |
| `agent.py` | PPO агент |
| `train.py` | Цикл обучения |
| `render.py` | PIL-рендер (кадры для видео) |
| `record.py` | Запись эпизодов в PNG |
| `make_video.py` | PNG → MP4 (ffmpeg) |

## Как запустить

```bash
pip install -r requirements.txt

# Обучение (1000 эпизодов, ~3-5 мин на GPU)
python train.py 1000

# Запись игры (нужен checkpoints/final.pt)
python record.py

# Компиляция видео
python make_video.py
```

## Как это работает

```
State: [bird_y, velocity, dist_to_pipe, gap_center]  (координата птицы, скорость, расстояние до трубы, координата проема)
        ↓
  Neural Network (4→64→64→2, ~8000 параметров)
        ↓
Action: P(flap), P(no_flap)
        ↓
Reward: +0.1/step, +1/pipe, -1/death
```

**Алгоритм:** PPO (Proximal Policy Optimization)
- Policy gradient: `∇log π(a|s) · A(s,a)`
- Clip: `ratio ∈ [0.8, 1.2]` — максимум 20% изменение за шаг
- GAE для оценки advantage

## Зависимости

- Python 3.10+
- PyTorch
- NumPy
- Pillow
- ffmpeg (системный)
