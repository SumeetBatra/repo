import torchvision.datasets as datasets
import torchvision.transforms as TF
import sapien
import torch
import numpy as np
import cv2
import random

from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils import sapien_utils
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from sapien.core import Pose
from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent
from mani_skill.utils.structs.pose import Pose as PoseStruct

from pathlib import Path
from typing import Dict, Any, Union


def load_places(image_size=64):
    datasets_dir = Path("./data")
    fp = datasets_dir / "places365_standard/val"
    return datasets.ImageFolder(str(fp),
                         TF.Compose([
                             TF.RandomResizedCrop((image_size, image_size)),
                             TF.ToTensor()
                         ]))


def segment_obs(obs: Dict[str, Any], img: torch.Tensor, seg_ids: torch.Tensor):
    device = img.device
    # only green-screen out the background
    # seg_id = torch.tensor([background_seg_id], dtype=torch.int, device=device)
    for cam_name in obs['sensor_data'].keys():
        camera_data = obs['sensor_data'][cam_name]
        seg = camera_data['segmentation']
        mask = torch.zeros_like(seg)
        mask[torch.isin(seg, seg_ids)] = 1
        camera_data['rgb'] = camera_data['rgb'] * (1 - mask) + img * mask
    return obs