import gymnasium as gym

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from distracting_envs import *
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from typing import Tuple, Optional
from copy import deepcopy

camera_poses = {
    'angled_view': look_at([0.4, 0.5, 0.6], [0.0, 0.0, 0.35]),
    'default_view': look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
}


def make_env_maniskill3(
        env_id: str,
        obs_mode: str,
        run_name: str,
        num_envs: int = 1,
        render_mode: str = 'all',
        camera_view: str = 'default_view',
        control_mode: str = "pd_ee_delta_pose",
        camera_resolution: Optional[Tuple[int, int]] = None,
        domain_randomize: bool = False,
        rand_lighting: bool = False,
        reconfiguration_freq: int = None,
):
    env_kwargs = dict(obs_mode=obs_mode, render_mode=render_mode, sim_backend="gpu", sensor_configs=dict())
    pose = camera_poses[camera_view]
    env_kwargs['sensor_configs']['pose'] = pose
    env_kwargs['control_mode'] = control_mode

    if camera_resolution is not None:
        width, height = camera_resolution
        env_kwargs['sensor_configs']['width'] = width
        env_kwargs['sensor_configs']['height'] = height

    env_kwargs['rand_lighting'] = rand_lighting

    if domain_randomize:
        env_kwargs['domain_randomize'] = True
        env_kwargs['domain_rand_cfg'] = dict(cube=dict(size_noise=0.0), randomize_color=True)

    env = gym.make(env_id,
                   num_envs,
                   reconfiguration_freq,
                   **env_kwargs)

    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        # Flatten the action space for compatibility with Dreamer/Repo
        env = FlattenActionSpaceWrapper(env)
    env = RecordEpisode(env, output_dir=f'./logdir/repo/{run_name}/train_videos', save_trajectory=False, save_video=True,
                        trajectory_name='trajectory', max_steps_per_video=50, video_fps=30)

    env = ManiSkillVectorEnv(env, num_envs=1, ignore_terminations=True, record_metrics=True)

    return env



