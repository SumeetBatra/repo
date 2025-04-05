import numpy as np
import sapien
import torch
import random
import cv2

from distracting_envs.utils import *
from typing import Dict, Any, Union
from mani_skill.envs.tasks.tabletop import PickSingleYCBEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.utils.structs.pose import Pose
from typing import Any, Dict, List, Union


WARNED_ONCE = False

@register_env('DistractingPickSingleYCB-v1', max_episode_steps=50, asset_download_ids=["ycb"])
class DistractingPickSingleYCBEnv(PickSingleYCBEnv):
    def __init__(self,
                 *args,
                 robot_uids="panda",
                 robot_init_qpos_noise=0.02,
                 rand_lighting=False,
                 distracting_objects=False,
                 distracting_background_places=False,
                 rand_table_texture=False,
                 domain_randomize: bool = False,
                 domain_rand_cfg: Dict[str, Any] = None,
                 sim2real: bool = False,
                 **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # visual randomization params
        self.rand_lighting = rand_lighting
        self.rand_table_texture = rand_table_texture
        self.distracting_objects = distracting_objects
        self.distracting_background_places = distracting_background_places
        self.domain_randomize = domain_randomize
        self.domain_rand_cfg = domain_rand_cfg
        self.sim2real = sim2real
        self.green_screen_img = None
        self.places_img_paths = None
        if distracting_background_places:
            self.places_img_paths = load_places(image_size=64)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_lighting(self, options: dict):
        if self.rand_lighting:
            for scene in self.scene.sub_scenes:
                scene.ambient_light = [np.random.uniform(0., 1.0), np.random.uniform(0., 1.0),
                                       np.random.uniform(0., 1.0)]
                scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
                scene.add_directional_light([0, 0, -1], [1, 1, 1])

        else:
            super()._load_lighting(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Initialize robot arm to a higher position above the table than the default typically used for other table top tasks
            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            elif self.robot_uids == "xmate3_robotiq":
                qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.562, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        if self.rand_table_texture:
            texture_path = './distracting_envs/assets/table_black_foam.jpg'
            texture_2d = sapien.render.RenderTexture2D(texture_path)
            table_actor = self.scene.actors['table-workspace']
            for obj in table_actor._objs:
                render_body_comp = obj.find_component_by_type(RenderBodyComponent)
                for render_shape in render_body_comp.render_shapes:
                    for part in render_shape.parts:
                        part.material.set_base_color_texture(texture_2d)

    def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        obs, info = super().reset(seed, options)
        if self.distracting_background_places:
            width, height = self._sensors['base_camera'].camera.width, self._sensors['base_camera'].camera.height
            img_path = random.choices(self.places_img_paths.samples)[0][0]
            img = cv2.resize(cv2.imread(img_path), (width, height))
            self.green_screen_img = torch.from_numpy(img).to(self.device).to(torch.int16)
            seg_ids = torch.tensor([0], dtype=torch.int16, device=self.device)
            seg_ids = torch.concatenate([seg_ids, self.unwrapped.scene.actors["ground"].per_scene_id])
            obs = segment_obs(obs, self.green_screen_img, seg_ids)
        if self.sim2real:
            del obs['agent']['qvel']
        return obs, info

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        obs, rew, terminate, truncate, info = super().step(action)
        if self.distracting_background_places:
            seg_ids = torch.tensor([0], dtype=torch.int16, device=self.device)
            seg_ids = torch.concatenate([seg_ids, self.unwrapped.scene.actors["ground"].per_scene_id])
            obs = segment_obs(obs, self.green_screen_img, seg_ids)
        if self.sim2real:
            del obs['agent']['qvel']
        return obs, rew, terminate, truncate, info