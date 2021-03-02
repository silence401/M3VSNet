import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from datasets.data_io import *
import cv2
import math

class PCLDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(PCLDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        assert self.mode in ["train", "val", "test"]
        self.metas = self.gen_eth3d_path(self.datapath, mode)

    def gen_eth3d_path(self, eth3d_data_folder, mode='training'):
        """ generate data paths for eth3d dataset """
        #print("=======================")
       # print(eth3d_data_folder)
        sample_list = []

        data_names = []
        if mode == 'train':
            data_names = ['electro']
        elif mode == 'test':
            data_names = ['playground', 'terrains']

        for data_name in data_names:

            data_folder = os.path.join(eth3d_data_folder, data_name)

            image_folder = os.path.join(data_folder, 'images')
            depth_folder = os.path.join(data_folder, 'depths')
            cam_folder = os.path.join(data_folder, 'cams')

            # index to image name
            index2name = dict()
            dict_file = os.path.join(cam_folder,'index2prefix.txt')
            dict_file = open(dict_file,  'r')
            dict_list = dict_file.read().split()
            dict_size = int(dict_list[0])
            for i in range(0, dict_size):
                index = int(dict_list[2 * i + 1])
                name = str(dict_list[2 * i + 2])
                index2name[index] = name
            name2depth = dict()
            name2depth['images_rig_cam4_undistorted'] = 'images_rig_cam4'
            name2depth['images_rig_cam5_undistorted'] = 'images_rig_cam5'
            name2depth['images_rig_cam6_undistorted'] = 'images_rig_cam6'
            name2depth['images_rig_cam7_undistorted'] = 'images_rig_cam7'

            # cluster
            cluster_file = os.path.join(cam_folder,'pair.txt')
            cluster_file = open(cluster_file, 'r')
            cluster_list = cluster_file.read().split()
            for p in range(0, int(cluster_list[0])):
                paths = []
                ref_view_path = []
                # ref image
                ref_index = int(cluster_list[22 * p + 1])
                ref_image_name = index2name[ref_index]
                ref_image_path = os.path.join(image_folder, ref_image_name)
                ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % ref_index))
                
                # view images
                src_paths = []
                for view in range(self.nviews - 1):
                    view_index = int(cluster_list[22 * p + 2 * view + 3])
                    view_image_name = index2name[view_index]
                    view_image_path = os.path.join(image_folder, view_image_name)
                    view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' % view_index))
                    src_paths.append([view_image_path,view_cam_path])
                paths.append(src_paths)
                # depth path
                image_prefix = os.path.split(ref_image_name)[1]
                depth_sub_folder = name2depth[os.path.split(ref_image_name)[0]]
                ref_depth_name = os.path.join(depth_sub_folder, image_prefix)
                ref_depth_name = os.path.splitext(ref_depth_name)[0] + '.pfm'
                depth_image_path = os.path.join(depth_folder, ref_depth_name)
                #paths.append((depth_image_path)
                ref_view_path.append([ref_image_path, ref_cam_path, depth_image_path])
                sample_list.append((ref_view_path, src_paths))
        #print(sample_list[0])
        dict_file.close()
        cluster_file.close()
        return sample_list

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            pair_file = "./pair.txt"
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        img.convert('RGB')
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img
    
    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)
    
    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics
    
    def crop_mvs_input(self, images, cams, depth_image=None, max_w=0, max_h=0):
    #""" resize images and cameras to fit the network (can be divided by base image size) """
    # crop images and cameras
        h, w = images.shape[0:2]
        new_h = h
        new_w = w
        if new_h > max_h:
            new_h = max_h
        else:
            new_h = int(math.ceil(h / 32) * 32)
            print("????")
            print(new_h)
        if new_w > max_w:
            new_w = max_w
        else:
            new_w = int(math.ceil(w / 32) * 32)

        if max_w > 0:
            new_w = max_w
        if max_h > 0:
            new_h = max_h

        start_h = int(math.ceil((h - new_h) / 2))
        start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        images = images[start_h:finish_h, start_w:finish_w]
        cams[0][2] = cams[0][2] - start_w
        cams[1][2] = cams[1][2] - start_h

        # crop depth image
        # if not depth_image is None :
        #     depth_image = depth_image[start_h:finish_h, start_w:finish_w]

        if not depth_image is None:
            return images, cams, depth_image
        else:
            return images, cams
        
    def scale_camera(self, cam, scale=1):
        """ resize input in order to produce sampled depth map """
        new_cam = np.copy(cam)
        # focal:
        new_cam[0][0] = cam[0][0] * scale
        new_cam[1][1] = cam[1][1] * scale
        # principle point:
        new_cam[0][2] = cam[0][2] * scale
        new_cam[1][2] = cam[1][2] * scale
        return new_cam
    

    def scale_image(self, image, scale=1, interpolation='linear'):
        """ resize image using cv2 """
        if interpolation == 'linear':
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if interpolation == 'nearest':
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    
    def __getitem__(self, idx):
        meta = self.metas[idx]
        
        ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        #print([ref_view])
       # print("====================")
        #print(src_views[:self.nviews-1])
        view_ids = ref_view + src_views[:self.nviews - 1]
        #print(view_ids)

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        intrins=[]
        extrins=[]

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            # img_filename = os.path.join(self.datapath,
            #                             '{}/images/'.format(scan, vid + 1, light_idx))
            # mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            # depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            # proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            #print(vid)
            tmp_img = self.read_img(vid[0])
            #tmp_img = cv2.resize(tmp_img, (512, 768))
            #print(tmp_img.shape)
            #imgs.append(tmp_img)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(vid[1])
            """
            the pcl data have different size image 
            resize to same scale
            """
            #if max_h !=0 and max_w!=0:
            #tmp_img, intrinsics = scale_mvs_input(img, intrinsics, 640, 512)

            # imgs.append(tmp_img)
            # intrins.append(intrinsics)
            # extrins.append(extrinsics)

            # # multiply intrinsics and extrinsics to get projection matrix
            # proj_mat = extrinsics.copy()
            # proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            # proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                # depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                #                          dtype=np.float32)
               # mask = self.read_img(mask_filename)
                depth_values = np.arange(0, self.ndepths, dtype=np.float32)
                depth_values = depth_values*depth_interval+depth_min
                depth = self.read_depth(vid[2])
                print(tmp_img.shape)
                tmp_img, intrinsics, depth = self.crop_mvs_input(tmp_img, intrinsics, depth, 480, 320)
                print(tmp_img.shape)
                depth = self.scale_image(depth, 0.25)
                intrinsics = self.scale_camera(intrinsics, 0.25)

  
              #  depth = cv2.resize(depth, (512, 768))
                mask = np.array((depth > depth_min+depth_interval) & (depth < depth_min+(self.ndepths-2)*depth_interval), dtype=np.float32)
            else:
                tmp_img, intrinsics = self.crop_mvs_input(tmp_img, intrinsics, max_w = 480, max_h = 320)
                intrinsics = self.scale_camera(intrinsics, 0.25)
           # print(tmp_img.shape)
            imgs.append(tmp_img)
            intrins.append(intrinsics)
            extrins.append(extrinsics)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        intrins=np.stack(intrins)
        extrins=np.stack(extrins)
        # print(imgs.shape)
        # print(proj_matrices.shape)
        # print(depth.shape)
        # print(mask.shape)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth": depth,
                "depth_values": depth_values,
                "mask": mask,
                "intrinsics":intrins,
                "extrinsics":extrins}


if __name__ == "__main__":
    data = PCLDataset("/home/silence401/下载/dataset/mvsnet/eth3d", "../lists/dtu/train.txt", "train", 3, 128)
    item = data[10]