import numpy as np
import sapien
import torch
import random
import cv2

from distracting_envs.utils import *
from typing import Dict, Any, Union
from mani_skill.envs.tasks.tabletop import PullCubeEnv
from mani_skill.utils.structs import Actor


@register_env("DistractingPullCube-v1", max_episode_steps=50)
class DistractingPullCube(PullCubeEnv):
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

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        if self.domain_randomize:
            from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
            self.batched_rng = BatchedRNG.from_seeds(self._episode_seed)
            size_noise = self.domain_rand_cfg["cube"]['size_noise']
            half_sizes = self.batched_rng.uniform(self.cube_half_size - size_noise, self.cube_half_size + size_noise)
            if self.domain_rand_cfg["cube"]["randomize_color"]:
                colors = np.random.uniform(0, 1, (self.num_envs, 3))
            else:
                colors = np.array([[12, 42, 160]]).repeat(self.num_envs, axis=0) / 255
            cubes = []
            for i in range(self.num_envs):
                cube = actors.build_cube(
                    self.scene,
                    half_size=half_sizes[i],
                    color=colors[i].tolist() + [1],
                    name=f'cube_{i}',
                    initial_pose=sapien.Pose(p=[0, 0, half_sizes[i]]),
                    scene_idxs=[i]
                )
                cubes.append(cube)
            self.obj = Actor.merge(cubes, 'cube')
        else:
            # create cube
            self.obj = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="cube",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            )

        # create target
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

        if self.rand_table_texture:
            texture_path = 'distracting_envs/assets/table_black_foam.jpg'
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
            del obs['extra']['is_grasped']
        return obs, info

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        obs, rew, terminate, truncate, info = super().step(action)
        if self.distracting_background_places:
            seg_ids = torch.tensor([0], dtype=torch.int16, device=self.device)
            seg_ids = torch.concatenate([seg_ids, self.unwrapped.scene.actors["ground"].per_scene_id])
            obs = segment_obs(obs, self.green_screen_img, seg_ids)
        if self.sim2real:
            del obs['agent']['qvel']
            del obs['extra']['is_grasped']
        return obs, rew, terminate, truncate, info