'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


from .ModelNetDataLoader import pc_normalize, farthest_point_sample

category_ids = {
    '02691156': 0,
    '02747177': 1,
    '02773838': 2,
    '02801938': 3,
    '02808440': 4,
    '02818832': 5,
    '02828884': 6,
    '02843684': 7,
    '02871439': 8,
    '02876657': 9, 
    '02880940': 10,
    '02924116': 11,
    '02933112': 12,
    '02942699': 13,
    '02946921': 14,
    '02954340': 15,
    '02958343': 16,
    '02992529': 17,
    '03001627': 18,
    '03046257': 19,
    '03085013': 20,
    '03207941': 21,
    '03211117': 22,
    '03261776': 23,
    '03325088': 24,
    '03337140': 25,
    '03467517': 26,
    '03513137': 27,
    '03593526': 28,
    '03624134': 29,
    '03636649': 30,
    '03642806': 31,
    '03691459': 32,
    '03710193': 33,
    '03759954': 34,
    '03761084': 35,
    '03790512': 36,
    '03797390': 37,
    '03928116': 38,
    '03938244': 39,
    '03948459': 40,
    '03991062': 41,
    '04004475': 42,
    '04074963': 43,
    '04090263': 44,
    '04099429': 45,
    '04225987': 46,
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52,
    '04530566': 53,
    '04554684': 54,
}


class ShapeNet55ClsDataLoader(Dataset):
    def __init__(self, args, split='train', process_data=False):
        if False:
            self.root = '/cluster_HDD/umoja/jtang/ShapeNetV2_watertight/'
            self.root_split = '/cluster_HDD/umoja/jtang/ShapeNetV2_point/'
            self.npoints = 1024 # -2048 -4096 -8192
            self.process_data = True
            self.uniform = False
            self.use_normals = True
            self.num_category = 55
        else:
            self.root = args.root
            self.root_split = args.root_split
            self.npoints = args.num_point
            self.process_data = process_data
            self.uniform = args.use_uniform_sample
            self.use_normals = args.use_normals
            self.num_category = args.num_category
        self.classes = category_ids

        shape_ids = {}
        shape_ids['train'] = [ (cat, line.rstrip()[:-4]) for cat in self.classes for line in open(os.path.join(self.root_split, cat, 'train.lst')) if os.path.exists( os.path.join(self.root, cat, "4_pointcloud", line.rstrip()[:-4] + '.npz')) ]
        shape_ids['val']   = [ (cat, line.rstrip()[:-4]) for cat in self.classes for line in open(os.path.join(self.root_split, cat, 'val.lst')) if os.path.exists( os.path.join(self.root, cat, "4_pointcloud", line.rstrip()[:-4] + '.npz')) ]
        shape_ids['test']  = [ (cat, line.rstrip()[:-4]) for cat in self.classes for line in open(os.path.join(self.root_split, cat, 'test.lst')) if os.path.exists( os.path.join(self.root, cat, "4_pointcloud", line.rstrip()[:-4] + '.npz')) ]
        # merge val and test into test
        shape_ids['test']  = shape_ids['test'] + shape_ids['val']

        assert (split == 'train' or split == 'test')
        self.datapath = [(shape_ids[split][i][0], os.path.join(self.root, shape_ids[split][i][0], "4_pointcloud", shape_ids[split][i][1]) + '.npz') 
                        for i in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(self.root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root, 'shapenet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[fn[0]]
                    cls = np.array([cls]).astype(np.int32)
                    pc_data = np.load(fn[1])
                    points  = pc_data['points'].astype(np.float32)
                    normals = pc_data['normals'].astype(np.float32)
                    point_set = np.concatenate([points, normals], axis=1)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        #point_set = point_set[0:self.npoints, :]
                        ind = np.random.choice(point_set.shape[0], self.npoints, replace=False)
                        point_set = point_set[ind]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[fn[0]]
            label = np.array([cls]).astype(np.int32)
            pc_data = np.load(fn[1])
            points  = pc_data['points'].astype(np.float32)
            normals = pc_data['normals'].astype(np.float32)
            point_set = np.concatenate([points, normals], axis=1)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ShapeNet55ClsDataLoader('/cluster_HDD/umoja/jtang/ShapeNetV2_watertight/', '/cluster_HDD/umoja/jtang/ShapeNetV2_point/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
