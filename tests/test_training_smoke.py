"""Short CPU training runs of the canonical (tactical) v0 pipeline."""

import gym
import pytest
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from baseline import train_tactical_agent as tactical
from baseline.custom_callbacks import ActionQualityMonitor, MetricsLogger

pytestmark = pytest.mark.slow


class _ShortEpisodes(gym.Wrapper):
    """End every episode after a few moves so episode-end code paths run quickly."""

    def __init__(self, env, max_steps=10):
        super().__init__(env)
        self.max_steps, self.t = max_steps, 0

    def reset(self, **kwargs):
        self.t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.t += 1
        return obs, reward, done or self.t >= self.max_steps, info


def _model(level=3, short_episodes=False):
    make = tactical.make_env(0, use_opening_book=False, level=level)
    env = DummyVecEnv([(lambda: _ShortEpisodes(make())) if short_episodes else make])
    return DQN(
        tactical.MaskedDQNPolicy,
        env,
        learning_starts=50,
        buffer_size=1_000,
        batch_size=32,
        train_freq=1,
        device="cpu",
        verbose=0,
        seed=0,
    )


def test_dqn_learns_without_callbacks():
    model = _model()
    model.learn(300)
    assert model.num_timesteps == 300


def test_model_predicts_in_range_action():
    model = _model()
    obs = model.get_env().reset()
    action, _ = model.predict(obs, deterministic=True)
    assert 0 <= int(action[0]) < 218


def test_learn_with_metrics_callbacks_completes(tmp_path):
    model = _model(short_episodes=True)
    metrics = MetricsLogger(log_dir=str(tmp_path))
    model.learn(300, callback=CallbackList([metrics, ActionQualityMonitor(log_dir=str(tmp_path))]))
    assert metrics.episode_count >= 20
    assert len(metrics.timesteps) == metrics.episode_count
    assert metrics.csv_file.closed
    summary = (tmp_path / "csv" / "all_metrics.csv").read_text().splitlines()
    assert len(summary) == metrics.episode_count + 1  # header + one row per episode
    assert (tmp_path / "csv" / "action_quality.csv").exists()


def test_callbacks_do_not_open_files_until_training_starts(tmp_path):
    MetricsLogger(log_dir=str(tmp_path))
    ActionQualityMonitor(log_dir=str(tmp_path))
    assert not list((tmp_path / "csv").iterdir())


def test_policy_eval_mode_works():
    policy = _model().policy
    policy.eval()
    assert not policy.training


def test_curriculum_runs_all_stages(tmp_path):
    # Opening book on: the final full-game stage (level 0) is the only one that uses it.
    tactical.train_model(
        total_timesteps=2_000,  # every stage stays below learning_starts: tests stage plumbing, not learning
        model_dir=str(tmp_path / "m"),
        log_dir=str(tmp_path / "l"),
        use_opening_book=True,
        num_envs=1,
    )
    saved = sorted(p.name for p in (tmp_path / "m").glob("*_complete.zip"))
    assert len(saved) == 4, saved
    assert (tmp_path / "m" / "chess_tactical_final.zip").exists()


def test_training_errors_propagate_and_skip_final_save(tmp_path, monkeypatch):
    from baseline import train_agent

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(DQN, "learn", boom)
    with pytest.raises(RuntimeError, match="boom"):
        train_agent.train_model(total_timesteps=100, model_dir=str(tmp_path / "m"), log_dir=str(tmp_path / "l"))
    assert not list((tmp_path / "m").glob("*final*"))


def test_keyboard_interrupt_saves_interrupted_model(tmp_path, monkeypatch):
    from baseline import train_agent

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(DQN, "learn", interrupt)
    train_agent.train_model(total_timesteps=100, model_dir=str(tmp_path / "m"), log_dir=str(tmp_path / "l"))
    assert (tmp_path / "m" / "chess_with_opening_book_interrupted.zip").exists()
    assert not list((tmp_path / "m").glob("*final*"))
