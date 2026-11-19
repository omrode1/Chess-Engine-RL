"""Short CPU training runs of the canonical (tactical) v0 pipeline."""

import gym
import pytest
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

import train_tactical_agent as tactical
from utils.custom_callbacks import ActionQualityMonitor, MetricsLogger

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


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="#7: MetricsLogger.on_training_end raises 'All arrays must be of the same length'",
)
def test_learn_with_metrics_callbacks_completes(tmp_path):
    model = _model(short_episodes=True)
    model.learn(
        300, callback=CallbackList([MetricsLogger(log_dir=str(tmp_path)), ActionQualityMonitor(log_dir=str(tmp_path))])
    )


@pytest.mark.xfail(
    strict=True, raises=TypeError, reason="#8: MaskedDQNPolicy.train shadows nn.Module.train, breaking eval()"
)
def test_policy_eval_mode_works():
    _model().policy.eval()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#7/#8: curriculum stops after stage 1; stage 2 would crash on replay_buffer=None",
)
def test_curriculum_runs_all_stages(tmp_path):
    tactical.train_model(
        total_timesteps=4_000,
        model_dir=str(tmp_path / "m"),
        log_dir=str(tmp_path / "l"),
        use_opening_book=False,
        num_envs=1,
    )
    saved = sorted(p.name for p in (tmp_path / "m").glob("*_complete.zip"))
    assert len(saved) == 4, saved
