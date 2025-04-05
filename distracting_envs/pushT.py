from distracting_envs.utils import *
from mani_skill.envs.tasks.tabletop import PushTEnv


@register_env("DistractingPushT-v1", max_episode_steps=100)
class DistractingPushTEnv(PushTEnv):
    def __init__(self,
                 *args,
                 robot_uids="panda_stick",
                 robot_init_qpos_noise=0.02,
                 rand_lighting=False,
                 distracting_objects=False,
                 distracting_background_places=False,
                 rand_table_texture=False,
                 domain_randomize: bool = False,
                 domain_rand_cfg: Dict[str, Any] = None,
                 **kwargs
    ):
        self.distracting_background_places = distracting_background_places
        self.places_image_paths = None
        if distracting_background_places:
            self.places_image_paths = load_places(64)
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

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
