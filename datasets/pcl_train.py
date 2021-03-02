import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from datasets.data_io import *
import cv2
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
        self.metas = self.gen_pcl_path(self.datapath, mode)
    
    def gen_pcl_path(self, pcl_data_folder, mode='training'):

        sample_list = []

        data_names = []
        if mode == 'train':
            data_names = ['scene1']
        elif mode == 'test':
            data_names = []
        
        for data_name in data_names:
            data_folder = os.path.join(pcl_data_folder, data_name)
            print(data_folder)
            image_folder = os.path.join(data_folder, 'images')
            #no depth data?
            #depth_folder = os.path.join(data_folder, 'depths')
            cam_folder = os.path.join(data_folder, 'cams')

            cluster_file = os.path.join(data_folder, 'pair.txt')
            cluster_file = open(cluster_file, 'r')
            cluster_list = cluster_file.read().split()
            for ip in range(int(cluster_list[0])):
                paths = []

                #get ref image
                ref_view_path = []
                ref_index = int(cluster_list[22*ip+1])
                ref_image_path = os.path.join(image_folder, '%08d.jpg'%ref_index)
                ref_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' %ref_index))
                ref_view_path.append([ref_image_path, ref_cam_path, ref_image_path])
                
                #get src image
                src_path = []
                for view in range(self.nviews - 1):
                    view_index = int(cluster_list[22 * ip + 2 * view + 3])
                    view_image_path = os.path.join(image_folder, '%08d.jpg'%view_index)
                    view_cam_path = os.path.join(cam_folder, ('%08d_cam.txt' %view_index))
                    src_path.append([view_image_path, view_cam_path])
                
                sample_list.append((ref_view_path, src_path))
            
            cluster_file.close()
           # print(sample_list)
            return sample_list    
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
        img = Image.open(filename)
        img.convert('1')
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32)
        return np_img

        # read pfm depth file
        #return np.array(read_pfm(filename)[0], dtype=np.float32)
    
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

    
    def __getitem__(self, idx):
        print('here')
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
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(vid[1])
            tmp_img, intrinsics = self.scale_mvs_input(tmp_img, intrinsics, 640, 512)

            imgs.append(tmp_img)
            #print(tmp_img.shape)
            intrins.append(intrinsics)
            extrins.append(extrinsics)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                print(self.ndepths)
                depth_values = np.arange(0, self.ndepths, dtype=np.float32)
                depth_values = depth_values*depth_interval+depth_min
                #depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                #                        dtype=np.float64)
                #print(depth_min+self.ndepths*depth_interval)
                #print(depth_values)
               # mask = self.read_img(mask_filename)
                depth = self.read_depth(vid[0])
                print(depth.shape)
                #depth.convert('1')
                #print(depth.shape)
                depth = cv2.resize(depth, (tmp_img.shape[0], tmp_img.shape[1]))
                mask = np.array((depth > depth_min+depth_interval) & (depth < depth_min+(self.ndepths-2)*depth_interval), dtype=np.float32)

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


                    


                    



