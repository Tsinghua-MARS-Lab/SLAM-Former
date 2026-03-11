import os.path as osp
import cv2
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class HabitatHM3D_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        self.max_interval = 8
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        self.scenes = os.listdir(self.ROOT)

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.ROOT, scene)
            basenames = sorted(
                [f[:-4] for f in os.listdir(scene_dir) if f.endswith(".npz")],
                key=lambda x: int(x),
            )

            num_imgs = len(basenames)
            # TODO: because current minghui's training data is backward moving, now use seq from -1 to 0
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            start_img_ids.extend([(scene, id) for id in start_img_ids_])
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            # offset groups
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

        self.invalid_scenes = {scene: False for scene in self.scenes}

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        invalid_seq = True
        scene, start_id = self.start_img_ids[idx]  # 获取指定索引idx对应的场景名scene和起始图像id
        
        # 添加最大重试次数，防止无限循环导致分布式训练卡住
        max_retries = 100
        retry_count = 0

        while invalid_seq:
            retry_count += 1
            
            # 超过重试次数限制，抛出异常
            if retry_count > max_retries:
                raise RuntimeError(
                    f"[HabitatHM3D] Failed to get valid views after {max_retries} retries. "
                    f"idx={idx}, scene={scene}, num_views={num_views}. "
                    f"This may indicate insufficient valid frames in the dataset."
                )
            
            # 超过50次时打印警告
            if retry_count == 50:
                print(f"[HabitatHM3D WARNING] Already retried {retry_count} times for idx={idx}, scene={scene}")
            
            # 如果当前场景被标记为invalid则随机选择一个新的场景和起始图像id
            scene_retry = 0
            while self.invalid_scenes[scene]:
                scene_retry += 1
                if scene_retry > len(self.start_img_ids):
                    raise RuntimeError(
                        f"[HabitatHM3D] All scenes are invalid! Cannot find valid scene after {scene_retry} attempts."
                    )
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]

            all_image_ids = self.scene_img_list[self.sceneids[start_id]]  # 获取当前场景的所有图像id列表
            pos, ordered_video = self.get_seq_from_start_id(
                num_views, start_id, all_image_ids, rng, max_interval=self.max_interval
            )  # 根据起始图像id和其他参数生成图像序列的索引pos 并返回有序视频
            image_idxs = np.array(all_image_ids)[pos]  # 从all_image_ids提取图像序列

            views = []
            load_failed = False
            for view_idx in image_idxs:
                scene_id = self.sceneids[view_idx]
                scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

                basename = self.images[view_idx]

                try:
                    # Load RGB image
                    rgb_image = imread_cv2(osp.join(scene_dir, "image_" + basename + ".png"))
                    # Load depthmap
                    depthmap = imread_cv2(
                        osp.join(scene_dir, "depth_" + basename + ".png"), cv2.IMREAD_UNCHANGED
                    )
                    depthmap = depthmap.astype(np.float32) / 1000
                    depthmap[~np.isfinite(depthmap)] = 0  # invalid

                    camera_params = np.load(osp.join(scene_dir, basename + ".npz"))
                    intrinsics = np.float32(camera_params["intrinsics"])
                    camera_pose = np.eye(4, dtype=np.float32)
                    camera_pose[:3, :3] = camera_params["R_cam2world"]
                    camera_pose[:3, 3] = camera_params["t_cam2world"]
                except Exception as e:
                    print(f"[HabitatHM3D] Error loading {scene} {basename}: {e}, skipping scene")
                    self.invalid_scenes[scene] = True
                    load_failed = True
                    break

                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

                views.append(
                    dict(
                        img=rgb_image,
                        depthmap=depthmap.astype(np.float32),
                        camera_pose=camera_pose.astype(np.float32),
                        camera_intrinsics=intrinsics.astype(np.float32),
                        dataset="habitatHM3D",
                        label=self.scenes[scene_id] + "_" + basename,
                        instance=f"{str(idx)}_{str(view_idx)}",
                        is_metric=self.is_metric,
                        is_video=ordered_video,
                        quantile=np.array(0.98, dtype=np.float32),
                        img_mask=True,
                        ray_mask=False,
                        camera_only=True,
                        depth_only=False,
                        single_view=False,
                        reset=False,
                    )
                )
            
            # 只有成功加载所有视图才退出循环
            if not load_failed and len(views) == num_views:
                invalid_seq = False
                
        return views
