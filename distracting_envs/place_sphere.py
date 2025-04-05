from distracting_envs.utils import *
from typing import Dict, Any, Union
from mani_skill.envs.tasks.tabletop import PlaceSphereEnv
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs import Pose as PoseStruct


@register_env("DistractingPlaceSphere-v1", max_episode_steps=50)
class DistractingPlaceSphere(PlaceSphereEnv):
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
        self.green_screen_img = None
        self.places_img_paths = None
        self.bg_seg_id = 17
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
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the sphere
        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="sphere",
            body_type="dynamic",
        )

        # load the bin
        self.bin = self._build_bin(self.radius)

        # randomize the cube colors
        if self.domain_randomize:
            table_actor = self.scene.actors['sphere']
            for obj in table_actor._objs:
                render_body_comp = obj.find_component_by_type(RenderBodyComponent)
                for render_shape in render_body_comp.render_shapes:
                    for part in render_shape.parts:
                        color = np.random.uniform(0, 1, 3).tolist() + [1]
                        part.material.set_base_color(color)



    def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        obs, info = super().reset(seed, options)
        if self.distracting_background_places:
            img_path = random.choices(self.places_img_paths.samples)[0][0]
            img = cv2.resize(cv2.imread(img_path), (64, 64))
            self.green_screen_img = torch.from_numpy(img).to(self.device).to(torch.int16)
            seg_ids = torch.tensor([0], dtype=torch.int16, device=self.device)
            seg_ids = torch.concatenate([seg_ids, self.unwrapped.scene.actors["ground"].per_scene_id])
            obs = segment_obs(obs, self.green_screen_img, seg_ids)
        return obs, info

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        obs, rew, terminate, truncate, info = super().step(action)
        if self.distracting_background_places:
            seg_ids = torch.tensor([0], dtype=torch.int16, device=self.device)
            seg_ids = torch.concatenate([seg_ids, self.unwrapped.scene.actors["ground"].per_scene_id])
            obs = segment_obs(obs, self.green_screen_img, seg_ids)
        return obs, rew, terminate, truncate, info
