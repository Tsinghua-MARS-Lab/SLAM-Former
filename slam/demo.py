import os,sys
import torch 
import numpy as np
import re
import cv2 
import glob
import argparse

import time
import open3d as o3d
from PIL import Image
from rich import print
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import rerun as rr
import rerun.blueprint as rrb

sys.path.append('src')
from slamformer.models.slamformer import SLAMFormer

current_directory = os.path.dirname(os.path.abspath(__file__))

sys.path.append(current_directory+'/../')

import slam.utils as utils
from slam.rerun_helper import log_camera, log_window

def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


class SLAM:
    def __init__(
        self, 
        outdir='output/tmp', 
        kf_th=0.1, 
        bn_every=10,
        vis=False,
        save_gmem=True,
        ckpt_path='path/to/ckpt.pth',
        target_size=518,
        retention_ratio=0.5
        ):

        self.outdir = outdir
        self.kf_th=kf_th
        self.save_gmem = save_gmem
        self.bn_every=bn_every
        self.vis = vis
        self.ckpt_path = ckpt_path
        self.target_size = target_size


        self.times = []
        self.kf_time = []
        self.backend_time = []

        # model params
        ckpt, ckpt_raw = self.load_checkpoint()
        use_conv_head = self.checkpoint_uses_conv_head(ckpt)
        if use_conv_head:
            print("Detected ConvHead in checkpoint; setting use_conv_head=True.")

        self.model = SLAMFormer(
            retention_ratio=retention_ratio,
            bn_every=bn_every,
            use_conv_head=use_conv_head,
        )
        self.model = self.model.eval()
        self.load_model(ckpt, ckpt_raw)
        self.model.eval()
        self.model.to('cuda')

        # SLAM params
        self.fid = -1
        self.kid = -1
        self.kfids = []
        self.last_kfid = 0
        self.kf_timestamps = []
        # frontend
        self.frontend_times = 0
        # Token map
        self.map = None
        self.map_opt = None

        self.signal_backend = False
        self.backend_every = self.bn_every #10
        # 
        self.extrins = []
        self.intrins = []
        self.frames = []
        self.frame_colors = []
        self.kf_frames = []

        # 
        self.K = None
        self.update_K = False

        # vis
        if self.vis:
            self.entity="world"
            rr.init("SLAM", spawn=True)
            rr.log(self.entity, rr.ViewCoordinates.RIGHT_HAND_Z_UP)
            self.Twk = np.eye(4)
            self.K = np.eye(3)

    def load_checkpoint(self):
        ckpt_raw = torch.load(self.ckpt_path, map_location='cuda', weights_only=False)

        if isinstance(ckpt_raw, dict):
            if "model" in ckpt_raw:
                ckpt = ckpt_raw["model"]
                print("Loaded state_dict from 'model' key in checkpoint.")
            else:
                ckpt = ckpt_raw
        else:
            ckpt = ckpt_raw

        ckpt = utils.strip_module(ckpt)
        return ckpt, ckpt_raw

    def checkpoint_uses_conv_head(self, ckpt):
        return any(key.startswith("point_head.upsample_blocks.") for key in ckpt)

    def load_model(self, ckpt=None, ckpt_raw=None):
        if ckpt is None:
            ckpt, ckpt_raw = self.load_checkpoint()

        self.model.load_state_dict(ckpt, strict=False)
        del ckpt, ckpt_raw 

    @property
    def time(self):
        torch.cuda.synchronize()
        return time.perf_counter()

    def kf_detect(self, image):
        if self.kid == -1:
            self.extrins.append(torch.eye(4))
            return True

        frame = utils.load_image(image, target_size=self.target_size)
        _,H,W = frame.shape


        st = self.time #time.perf_counter()
        token = self.model.KFT(torch.stack([self.kf_frames[-1],frame.cuda()]))
        if self.vis:
            # scale the pose to global
            res = self.model.extract(token, cam_only=True)
            #z = res['local_points'][0,0,:,:,-1].cpu().numpy()
            if not hasattr(self,'depth_lask_kf'):
                scale=1
            else:
                scale=1 #np.median(self.depth_last_kf/(z+1e-6))
            camera_pose = res['camera_poses']

            extrinsic = torch.inverse(camera_pose)
            if extrinsic.shape[1] > 1:
                extrinsic_ref=extrinsic.cpu()[0,-2]
                extrinsic = extrinsic.cpu()[0,-1]
                Tki = torch.inverse(camera_pose[0,0])@camera_pose[0,1]
                Tki = Tki.cpu().numpy()
                self.Twi = self.Twk@Tki
                K44 = np.eye(4)
                K44[:3,:3] = self.K
                log_camera("camera",self.Twi, K44, kfd=True)
                # make the window follow camera
                log_window(f"{self.entity}",np.linalg.inv(self.Twi), K44)


        else:
            res = self.model.extract(token, cam_only=True)
            camera_pose = res['camera_poses']
            extrinsic = torch.inverse(camera_pose)
            if extrinsic.shape[1] > 1:
                extrinsic_ref=extrinsic.cpu()[0,-2]
                extrinsic = extrinsic.cpu()[0,-1]
                self.kft_extrinsic_ref = torch.eye(4)#extrinsic_ref

        dist = torch.sqrt(torch.sum((extrinsic[:3,3] - extrinsic_ref[:3,3])**2)) 
        isKF = dist > self.kf_th 

        print(dist)

        if isKF:
            self.extrins.append(extrinsic)
        return isKF

    def frontend(self, image):

        if self.vis:
            rr.log("image", rr.Image(image[:,:,::-1]))#,static=True)

        self.fid += 1
        print('Frame', self.fid)
        # run kf detector
        st = self.time
        enough_disparity = self.kf_detect(image)
        self.kf_time.append(self.time-st)
        if not enough_disparity:
            return False


        torch.cuda.empty_cache()
        # run T-frontend
        H_,W_,_ = image.shape
        frame = utils.load_image(image, target_size=self.target_size)
        self.H,self.W,_ = frame.shape
        frame_color = self.prepare_pointcloud_rgb(image, frame.shape[1:])
        st = self.time
        self.last_kf = frame.cuda()
        self.kf_frames.append(self.last_kf)
        self.last_kfid = self.fid
        self.frames.append(self.last_kf.clone())
        self.frame_colors.append(frame_color)
        self.kid += 1
        print("[italic purple] # KEYFRAME", self.kid)
        self.kf_timestamps.append(self.cur_timestamp)
        frame = frame.cuda()
        st = self.time

        if self.nkf == 1:
            pass
        elif self.nkf == 2:
            token = self.model.frontendT(torch.stack([self.kf_frames[0],frame]))
            self.map_add(token)
        else:
            token = self.model.frontendT(frame)
            print(self.time-st)

            self.map_add(token)

        self.kfids.append(self.fid)
        self.times.append(self.time-st)
        torch.cuda.empty_cache()

        # send signal to backend
        self.frontend_times += 1
        if self.frontend_times % self.backend_every == 0:
            self.signal_backend = True

        if self.vis and self.map is not None:
            st = time.time()
            map_before_bn = None
            if self.map_opt is None:
                map_before_bn = self.map
            else:
                S = self.map.shape[0]
                S_oldopt = self.map_opt.shape[0]

                map_before_bn = torch.cat([self.map_opt, self.map[S_oldopt:]],axis=0)
            if self.nkf == 2:
                ps,cs,confs,poses = self.extract(self.map)

            else:
                ps,cs,confs,poses = self.extract(self.map[-1:])


            self.vis_mem = [ps,cs,confs,poses]

            conf_threshold = np.percentile(confs, 15)
            msk = confs>=conf_threshold
            
            ps = ps[msk]
            cs = cs[msk]
            K44 = np.eye(4)
            K44[:3,:3] = self.K

            if self.nkf == 2:
                log_camera(f"{self.entity}/camera_kf/0",poses[0], K44)
                log_camera(f"{self.entity}/camera_kf/1",poses[1], K44)

                rr.log(f"{self.entity}/lines/0to1", rr.LineStrips3D([poses[:,:3,3].tolist()],colors=[0,0,255],radii=[0.005]))

                self.last_kf_pose = poses[1]
            else:
                log_camera(f"{self.entity}/camera_kf/{self.nkf-1}",poses.reshape(4,4), K44)
                rr.log(f"{self.entity}/lines/{self.nkf-2}to{self.nkf-1}", rr.LineStrips3D([np.stack([self.last_kf_pose[:3,3],poses[0,:3,3]]).tolist()],colors=[0,0,255],radii=[0.005]))

                self.last_kf_pose = poses[0]


            rr.log(
                    f"{self.entity}/pointclouds/{self.nkf}",
                    rr.Points3D(ps, colors=cs, radii=0.01),
                )

            print('log', time.time()-st)

            self.Twk = poses[-1].reshape(4,4)
        
    def backend(self, final=False):
        if not self.signal_backend:
            return

        torch.cuda.empty_cache()

        del self.model.fkv
        torch.cuda.empty_cache()
        print('Backending...', self.nkf, 'KFs')
        st = time.perf_counter()
        map_optimed = self.model.backendT(self.map.cuda())
        self.backend_time.append(time.perf_counter()-st)
        print('backend_take', time.perf_counter()-st)
        torch.cuda.empty_cache()

        if self.map_opt is not None:
            del self.map_opt
            torch.cuda.empty_cache()
        self.map_opt = map_optimed.cpu() 

        self.signal_backend = False
        torch.cuda.empty_cache()
        
        if self.vis:
            ps,cs,confs,poses = self.extract(self.map_opt)
            self.vis_mem = [ps,cs,confs,poses]
            conf_threshold = np.percentile(confs, 15)
            msk = confs>=conf_threshold
            
            ps = ps[msk]
            cs = cs[msk]
  

            for s in range(self.nkf+1):
                rr.log(f"{self.entity}/pointclouds/{s}", rr.Points3D(np.array([])))

            for s in range(self.nkf):
                K44 = np.eye(4)
                K44[:3,:3] = self.K
                log_camera(f"{self.entity}/camera_kf/{s}",poses[s].reshape(4,4), K44, update=True)

            for s in range(1, self.nkf):
                rr.log(f"{self.entity}/lines/{s-1}to{s}", rr.LineStrips3D([poses[s-1:s+1,:3,3].tolist()],colors=[0,0,255],radii=[0.005]))

            rr.log(
                    f"{self.entity}/pointclouds/{self.nkf}",
                    rr.Points3D(ps, colors=cs, radii=0.01),
                )
            self.last_kf_pose = poses[-1]

    def step(self, timestamp, image):
        if timestamp is None:
            self.cur_timestamp = self.fid+1
        else:
            self.cur_timestamp = timestamp
        
        self.frontend(image)

        self.backend()

    def map_add(self, token_kf):
        if self.map is None:
            self.map = token_kf.cpu() if self.save_gmem else token_kf #[tok.cpu() for tok in token_kf]
        else:
            if self.save_gmem:
                self.map = torch.cat([self.map, token_kf.cpu()],axis=0) # S,P,C
            else:
                self.map = torch.cat([self.map, token_kf],axis=0) # S,P,C

    @property
    def nkf(self):
        return self.kid+1

    @property
    def nf(self):
        return self.fid+1

    def prepare_pointcloud_rgb(self, image, target_hw):
        """Return RGB colors aligned to the model point grid."""
        target_h, target_w = target_hw
        if image.shape[-1] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(rgb).convert("RGB")
        width, height = img.size
        new_width = target_w
        new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        if new_height > target_h:
            start_y = (new_height - target_h) // 2
            img = img.crop((0, start_y, new_width, start_y + target_h))

        colors = np.asarray(img, dtype=np.float64) / 255.0
        assert colors.shape[:2] == (target_h, target_w), (
            f"RGB shape {colors.shape[:2]} does not match point grid {(target_h, target_w)}"
        )
        return colors

    def terminate(self):
        if self.nkf % self.backend_every != 0:
            self.signal_backend = True
            self.backend(final=True)

        print(self.kf_time)
        print(self.times)
        print(self.backend_time)
        print('frontend take', np.mean(self.times))
        print('KFT')
        print('total', np.sum(self.kf_time), 'FPS', float(len(self.kf_time))/np.sum(self.kf_time))
        print('FT')
        print('total', np.sum(self.times), 'FPS', float(len(self.times))/np.sum(self.times))
        print('BT')
        print('total', np.sum(self.backend_time), 'FPS', float(len(self.backend_time))/np.sum(self.backend_time))
        print('Summary')
        print('total', np.sum(self.kf_time)+np.sum(self.times)+np.sum(self.backend_time), 'FPS', float(len(self.kf_time))/(np.sum(self.kf_time)+np.sum(self.times)+np.sum(self.backend_time)))
        self.save_result(f'{self.outdir}/final', self.map_opt)

    def extract(self, map_all=None):
        result = self.model.extract(map_all.cuda())

        pts = result['points'].cpu().numpy() # 1,S,H,W,3
        local_pts = result['local_points'].cpu().numpy() # 1,S,H,W,3
        _,S,H,W,_ = pts.shape
        conf = result['conf'].cpu().numpy()
        point_clouds = [pts[0,s] for s in range(S)]
        #conf_threshold = np.percentile(conf, 15)
        #confs = [conf[0,s]>=conf_threshold for s in range(S)]
        assert len(self.frame_colors) >= S, "Not enough cached RGB keyframes for extracted point cloud"
        colors = np.stack(self.frame_colors[-S:], axis=0).reshape(-1,3) # S,H,W,C
        confs = conf.reshape(-1)


        camera_pose = result['camera_poses'].cpu().numpy()[0] # S,4,4
        pts = pts.reshape(-1,3)
        colors = colors.reshape(-1,3)

        # set depth for the last kf
        self.depth_last_kf = local_pts[0,-1,:,:,-1]

        return pts, colors, confs, camera_pose 

    def save_result(self, output_path = 'output/tmp', map_all=None, traj=True):
        '''
        if map_all is None:
            map_all = self.map
            '''
        print(self.kfids)

        if map_all is None:
            map_all = self.map_opt

        result = self.model.extract(map_all.cuda())
        pts = result['points'].cpu().numpy() # 1,S,H,W,3
        _,S,H,W,_ = pts.shape
        conf = result['conf'].cpu().numpy()
        point_clouds = [pts[0,s] for s in range(S)]
        conf_threshold = np.percentile(conf, 15)
        confs = [conf[0,s]>=conf_threshold for s in range(S)]

        assert len(self.frame_colors) >= S, "Not enough cached RGB keyframes for saved point cloud"
        colors = np.stack(self.frame_colors[-S:], axis=0).reshape(-1,3) # S,H,W,C
        msk = np.stack(confs).reshape(-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1,3).astype(np.float64)[msk])
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64)[msk])
        #downpcd = pcd.voxel_down_sample(voxel_size=0.005)
        o3d.io.write_point_cloud(f"{output_path}.ply", pcd)
        camera_pose = result['camera_poses'].cpu() #torch.Size([1, 14, 4,4])
        poses = camera_pose[0].numpy() 

        self.write_poses_to_file(f"{output_path}_traj.txt", poses, self.kf_timestamps)
        self.save_framewise_pointclouds(
            f"{output_path}_pc",
            point_clouds,
            self.kf_timestamps,
            confs,
            np.stack(self.frame_colors[-S:], axis=0),
        )

        return result

    def write_poses_to_file(self, filename, poses, frame_ids):

        with open(filename, "w") as f:
            assert len(poses) == len(frame_ids), "Number of provided poses and number of frame ids do not match"
            for frame_id, pose in zip(frame_ids, poses):
                x, y, z = pose[0:3, 3]
                rotation_matrix = pose[0:3, 0:3]
                quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                output = np.array([float(frame_id), x, y, z, *quaternion])
                f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def save_framewise_pointclouds(self, filename, pointclouds, frame_ids, conf_masks, colors):
        os.makedirs(filename, exist_ok=True)
        assert len(pointclouds) == len(colors), "Point clouds and RGB colors must have the same length"
        for frame_id, pointcloud, conf_masks, color in zip(frame_ids, pointclouds, conf_masks, colors):
            # save pcd as numpy array
            np.savez(f"{filename}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks, color=color)


def get_parser():
    parser = argparse.ArgumentParser(description="SLAM-Former demo")
    parser.add_argument("--ckpt_path", type=str, default="path/to/checkpoint.pth.model", help="Path to the checkpoint")
    parser.add_argument("--image_folder", type=str, default="path/to/image/folder", help="Path to folder containing images")
    parser.add_argument("--target_size", type=int, default=518, help="the target size of image(longer side)")
    parser.add_argument("--output_dir", type=str, default="outputs/tmp", help="Path to save the output")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for subsampling the input sequence")
    parser.add_argument("--kf_th", type=float, default=0.1, help="Keyframe selection threshold (minimum translation distance)")
    parser.add_argument("--retention_ratio", type=float, default=0.5, help="KV Pruning retention ratio")
    parser.add_argument("--bn_every", type=int, default=10, help="Run backend optimization every N keyframes")
    parser.add_argument("--vis", action="store_true", help="Enable real-time visualization with Rerun")
    parser.add_argument("--resize_rate", type=float, default=1, help="Resize rate for input images before processing")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    image_folder = args.image_folder
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    if 'tum' in args.image_folder:
        fx = 525.0  # focal length x
        fy = 525.0  # focal length y
        cx = 319.5  # optical center x
        cy = 239.5  # optical center y
        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy
    elif 'Replica' in args.image_folder:
        fx = 600.  # focal length x
        fy = 600.0  # focal length y
        cx = 599.5  # optical center x
        cy = 339.5  # optical center y
        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy
    else:
        K = None


    # Use the provided image folder path
    print(f"Loading images from {image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(image_folder, "*"))
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower()
               and "db" not in os.path.basename(f).lower()]
    image_names = utils.sort_images_by_number(image_names)

    frame_ids = []
    for path in image_names:
        filename = os.path.basename(path)
        match = re.search(r'\d+(?:\.\d+)?', filename)  # matches integers and decimals
        if match:
            frame_ids.append(float(match.group()))
        else:
            raise ValueError(f"No number found in image name: {filename}")

    print(f"Found {len(image_names)} images")

    print('resize image', args.resize_rate)

    slam = SLAM(
        outdir=outdir,
        kf_th=args.kf_th,
        bn_every=args.bn_every,
        vis=args.vis,
        ckpt_path=args.ckpt_path,
        target_size=args.target_size,
        retention_ratio=args.retention_ratio
    )
    
    slam.K = K
    for frame_id, image_name in zip(frame_ids[::args.stride], image_names[::args.stride]):
        img = cv2.imread(image_name)

        if args.resize_rate != 1:
            H,W,_ = img.shape
            img = cv2.resize(img, (int(W*args.resize_rate), int(H*args.resize_rate)), cv2.INTER_CUBIC)
        slam.step(frame_id, img)
    result = slam.terminate()
