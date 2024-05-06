import random
import numpy as np
import torch
import open3d as o3d
from torchvision import transforms as transforms
from ..utils.util import get_grid, grid_to_pointcloud
from .abstract import AbstractDataset
import os
from ..geotransformer.utils.pointcloud import \
    (random_sample_rotation,
    get_transform_from_rotation_translation,)
from ..visualize import save_ply ,adjust_intrinsic

###remove out###
remove_20_test = [32, 3727, 3728, 3858, 4831, 5069, 5380, 5381, 6032, 6284, 6285, 8112, 9894, 9895, 9947, 9948, 10297, 10915, 10944, 10945, 10959,10962, 10963, 11055, 13151, 13152, 13153, 14044, 14626, 14627, 14812, 14813, 15299, 15326, 15550, 15974, 16247, 16969, 16970,17616, 17617, 18008, 18009, 19116, 19513, 19514, 19540, 19541, 19558, 19559, 20113, 20117, 20406, 20407, 22432, 22433, 22814,22815, 23483, 23484, 23845, 23847, 23850, 23851, 23957, 23959, 24021, 24022, 25721, 25722, 25725, 25791]
remove_50_test = [3, 9, 10, 11, 29, 30, 31, 32, 33, 34, 35, 40, 41, 219, 538, 539, 540, 541, 564, 565, 609, 610, 611, 612, 629, 631, 832, 881, 897, 898, 899, 1002, 1004, 1006, 1007, 1011, 1016, 1017, 1018, 1042, 1043, 1044, 1067, 1068, 1072, 1073, 1111, 1112, 1124, 1133, 1134, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1197, 1198, 1201, 1212, 1213, 1382, 1385, 1559, 1560, 1561, 1569, 1713, 1714, 2008, 2009, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2037, 2038, 2049, 2064, 2065, 2066, 2067, 2087, 2115, 2116, 2117, 2118, 2125, 2148, 2162, 2163, 2164, 2237, 2305, 2306, 2312, 2331, 2336, 2337, 2338, 2410, 2513, 2579, 2581, 2746, 2764, 2778, 2779, 2780, 2989, 2997, 3055, 3056, 3065, 3066, 3081, 3082, 3083, 3084, 3091, 3223, 3224, 3225, 3226, 3227, 3228, 3259, 3260, 3262, 3291, 3466, 3467, 3470, 3594, 3699, 3700, 3760, 3774, 3775, 3776, 3777, 3801, 3813, 3820, 3821, 3822, 3823, 3824, 3825, 3832, 4062, 4063, 4064, 4065, 4066, 4067, 4071, 4072, 4073, 4081, 4082, 4214, 4215, 4217, 4269, 4407, 4419, 4423, 4424, 4448, 4449, 4457, 4458, 4459, 4472, 4476, 4477, 4478, 4479, 4483, 4484, 4489, 4490, 4491, 4492, 4497, 4498, 4499, 4500, 4501, 4504, 4505, 4506, 4507, 4512, 4513, 4524, 4531, 4701, 4720, 4722, 4735, 4736, 4776, 4777, 4778, 4779, 4910, 4933, 4959, 4985, 4986, 4987, 4988, 5006, 5008, 5012, 5014, 5015, 5040, 5074, 5075, 5146, 5254, 5255, 5256, 5366, 5384, 5440, 5486, 5490, 5497, 5498, 5503, 5505, 5546, 5547, 5550, 5558, 5559, 5564, 5565, 5566, 5567, 5568, 5569, 5596, 5607, 5608, 5617, 5618, 5688, 5694, 5695, 5697, 5700, 5701, 5702, 5703, 5707, 5712, 5713, 5714, 5715, 5716, 5724, 5725, 5726, 5727, 5728, 5729, 5730, 5731, 5732, 5733, 5734, 5735, 5736, 5737, 5738, 5741, 5744, 5745, 5746, 5747, 5748, 5749, 5750, 5753, 5754, 5755, 5756, 5759, 5760, 5761, 5762, 5763, 5764, 5765, 5766, 5768, 5769, 5770, 5774, 5806, 5807, 5808, 5809, 5810, 5811, 5812, 5813, 5814, 5818, 5819, 5828, 5829, 5849, 5850, 5851, 5852, 5853, 5854, 5855, 5856, 5866, 5867, 5868, 5869, 5870, 5872, 5873, 5880, 5881, 5895, 5896, 5938, 5939, 5940, 6047, 6075, 6076, 6077, 6078, 6079, 6084, 6085, 6086, 6088, 6096, 6097, 6098, 6104, 6105, 6124, 6130, 6134, 6135, 6136, 6137, 6138, 6139, 6140, 6141, 6142, 6147, 6153, 6154, 6156, 6292, 6293, 6294, 6296, 6297, 6308, 6328, 6329, 6330, 6331, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6339, 6397, 6399, 6463, 6464, 6470, 6506, 6507, 6508, 6509, 6510, 6511, 6571, 6572, 6573, 6648, 6754, 6907, 6908, 6956, 7035, 7094, 7562, 7568, 7575, 7576, 7704, 7742, 7743, 7744, 7745, 7746, 7886, 7888, 8378, 8398, 8399, 8401, 8461, 8738, 8741, 8742, 8850, 8851, 8852, 8918, 9112, 9113, 9191, 9223, 9225, 9226, 9227, 9228, 9229, 9230, 9232, 9233, 9251, 9253, 9262, 9263, 9264, 9268, 9278, 9279, 9283, 9284, 9286, 9287, 9335, 9336, 9507, 9508, 9588, 9597, 9602, 9603, 9604, 9605, 9607, 9608, 9609, 9717, 9721, 9727, 9744, 9745, 9750, 9751, 9752, 9753, 9757, 9764, 9771, 9772, 9773, 9778, 9779, 9788, 9790, 9864, 9865, 9906, 9908, 9909, 9910, 9914, 9915, 9916, 9918, 9919, 9920, 9921, 9922, 9923, 9932, 9933, 9934, 9935, 9936, 9941, 9942, 9943, 9944, 9945, 9946, 9947, 10151, 10152, 10153, 10155, 10156, 10157, 10158, 10159, 10160, 10161, 10162, 10163, 10167, 10168, 10169, 10170, 10172, 10173, 10174, 10175, 10176, 10177, 10178, 10179, 10180, 10181, 10182, 10183, 10184, 10186, 10188, 10190, 10191, 10192, 10193, 10194, 10195, 10197, 10198, 10200, 10201, 10202, 10203, 10205, 10206, 10207, 10208, 10209, 10210, 10211, 10212, 10213, 10214, 10215, 10216, 10217, 10218, 10219, 10222, 10303, 10320, 10321, 10392, 10393, 10398, 10399, 10400, 10401, 10406, 10407, 10428, 10429, 10430, 10457, 10458, 10459, 10460, 10471, 10472, 10551, 10552, 10553, 10563, 10564, 10566, 10567, 10568, 10569, 10570, 10705, 10772, 10778, 10941, 11050, 11051, 11226, 11259, 11260, 11261, 11262, 11263, 11268, 11269, 11270, 11344, 11365, 11372, 11373, 11417, 11418, 11419, 11420, 11421, 11422, 11455, 11457, 11525, 11527, 11528, 11529, 11530, 11537, 11589, 11590, 11591, 11592, 11593, 11594, 11595, 11596, 11597, 11598, 11646, 11647, 11656, 11673, 11674, 11675, 11676, 11679, 11815, 11861, 11864, 11874, 11875, 11876, 11877, 11878, 11879, 11880, 11916, 11917, 11935, 11938, 11940, 11941, 11942, 11943, 11947, 11949, 12014, 12015, 12026, 12027, 12028, 12029, 12030, 12223, 12224, 12225, 12226, 12228, 12229, 12230, 12231, 12232, 12259, 12260, 12261, 12265, 12266, 12267, 12268, 12269, 12270, 12273, 12274, 12275, 12367, 12368, 12369, 12370, 12543, 12696, 12697, 12698, 12739, 12752, 12754, 12865, 12926, 13047, 13085, 13097, 13201, 13251, 13252, 13253, 13254, 13306, 13307, 13511, 13512, 13514, 13515, 13527, 13528, 13529, 13534, 13535, 13536, 13541, 13542, 13591, 13592, 13637, 13638, 13639, 13640, 13641, 13642, 13643, 13745, 13767, 13768, 13773, 13774, 13775, 13776, 13778, 13779, 13780, 13785, 13786, 13799, 13803, 13810, 13814, 13815, 13816, 13824, 13829, 13831, 13928, 13932, 13934, 13935, 13947, 13962, 14229, 14256, 14273, 14274, 14275, 14291, 14313, 14320, 14321, 14322, 14323, 14331, 14332, 14333, 14334, 14423, 14424, 14443, 14444, 14453, 14457, 14458, 14471, 14492, 14497, 14498, 14499, 14500, 14501, 14504, 14512, 14513, 14519, 14520, 14521, 14522, 14523, 14528, 14529, 14530, 14531, 14532, 14533, 14539, 14540, 14541, 14542, 14543, 14736, 14737, 14738, 14739, 14759, 14778, 14785, 14786, 14818, 14819, 14861, 14890, 14891, 14893, 14894, 14895, 14896, 14897, 14898, 14899, 14979, 15000, 15041, 15078, 15083, 15096, 15097, 15098, 15099, 15100, 15101, 15102, 15111, 15112, 15121, 15122, 15123, 15124, 15129, 15132, 15133, 15146, 15147, 15148, 15149, 15150, 15154, 15155, 15156, 15166, 15171, 15289, 15290, 15291, 15308, 15310, 15311, 15312, 15317, 15387, 15388, 15389, 15413, 15440, 15441, 15442, 15542, 15577, 15697, 15748, 15783, 15784, 15785, 15786, 15787, 15792, 15793, 15794, 15795, 15796, 15797, 15802, 15803, 15804, 15805, 15806, 15809, 15810, 15813, 15816, 15817, 15818, 15819, 15820, 15821, 15822, 15823, 15824, 15825, 15826, 16014, 16015, 16027, 16029, 16030, 16060, 16061, 16062, 16134, 16281, 16282, 16283, 16284, 16306, 16307, 16309, 16310, 16311, 16312, 16313, 16357, 16358, 16360, 16361, 16362, 16363, 16364, 16365, 16366, 16368, 16369, 16370, 16371, 16372, 16373, 16374, 16375, 16376, 16377, 16378, 16383, 16384, 16385, 16386, 16387, 16389, 16390, 16391, 16392, 16396, 16397, 16401, 16402, 16414, 16415, 16416, 16417, 16419, 16420, 16421, 16422, 16425, 16485, 16497, 16500, 16513, 16529, 16530, 16531, 16538, 16540, 16541, 16542, 16545, 16569, 16570, 16571, 16579, 16604, 16605, 16606, 16635, 16657, 16734, 16775, 16776, 16777, 16787, 16788, 16789, 16790, 16791, 16825, 16826, 16827, 16828, 16840, 16841, 16843, 17289, 17290, 17291, 17300, 17303, 17305, 17558, 17560, 17561, 17590, 17591, 17592, 17594, 17595, 17603, 17612, 17613, 17798, 17799, 17800, 17801, 17805, 17815, 17817, 17819, 17820, 18163, 18164, 18171, 18173, 18177, 18178, 18179, 18184, 18185, 18186, 18187, 18188, 18189, 18211, 18212, 18213, 18214, 18222, 18223, 18228, 18229, 18230, 18231, 18232, 18233, 18234, 18263, 18269, 18287, 18288, 18289, 18290, 18292, 18299, 18300, 18302, 18303, 18316, 18317, 18318, 18319, 18321, 18322, 18323, 18324, 18325, 18326, 18327, 18405, 18527, 18560, 18561, 18562, 18563, 18567, 18568, 18569, 18570, 18586, 18587, 18588, 18589, 18590, 18591, 18593, 18619, 18634, 18635, 18636, 18637, 18639, 18721, 18723, 18742, 18743, 18747, 18748, 18749, 18750, 18771, 18772, 18773, 18774, 18775, 18776, 18784, 18836, 18873, 19004, 19005, 19006, 19007, 19008, 19009, 19010, 19014, 19015, 19016, 19017, 19018, 19022, 19023, 19049, 19066, 19458, 19459, 19505, 19513, 19515, 19516, 19674, 19675, 19676, 19677, 19678, 19679, 19680, 19681, 19683, 19684, 19703, 19704, 19715, 19851, 19855, 19881, 19884, 20035, 20036, 20044, 20045, 20046, 20047, 20049, 20091, 20097, 20124, 20242, 20243, 20249, 20250, 20274, 20276, 20417, 20422, 20571, 20607, 20608, 20628, 20629, 20630, 20631, 20783, 20797, 20800, 20801, 20802, 20833, 20834, 20835, 20837, 20853, 20854, 20868, 20882, 20892, 20893, 20895, 20896, 20897, 20899, 20900, 20901, 20902, 20903, 20905, 20906, 20907, 20908, 20909, 20910, 20911, 20912, 20913, 20914, 20916, 20917, 20918, 20919, 20941, 20942, 20950, 20951, 21026, 21090, 21109, 21110, 21196, 21201, 21202, 21203, 21204, 21205, 21206, 21207, 21208, 21209, 21210, 21211, 21219, 21220, 21221, 21222, 21223, 21224, 21229, 21230, 21231, 21232, 21234, 21235, 21236, 21237, 21238, 21239, 21248, 21249, 21250, 21251, 21252, 21253, 21254, 21255, 21256, 21259, 21260, 21261, 21262, 21272, 21273, 21274, 21276, 21277, 21279, 21280, 21281, 21282, 21283, 21284, 21345, 21346, 21500, 21563, 21564, 21565, 21566, 21567, 21857, 21870, 21872, 21873, 21878, 21879, 21880, 21881, 21882, 21885, 21902, 21903, 21904, 21929, 21930, 21940, 22024, 22031, 22032, 22033, 22057, 22058, 22059, 22060, 22062, 22065, 22139, 22140, 22141, 22195, 22200, 22201, 22202, 22216, 22217, 22218, 22219, 22220, 22222, 22223, 22224, 22226, 22230, 22231, 22240, 22255, 22256, 22257, 22315, 22316, 22317, 22322, 22323, 22324, 22325, 22326, 22328, 22329, 22371, 22372, 22373, 22380, 22381, 22382, 22439, 22440, 22445, 22548, 22561, 22662, 22685, 22694, 22702, 22703, 22705, 22708, 22709, 22710, 22711, 22712, 22718, 22719, 22720, 22721, 22759, 22776, 22777, 22778, 22779, 22895, 22896, 23073, 23075, 23076, 23200, 23201, 23295, 23296, 23297, 23298, 23301, 23302, 23303, 23304, 23305, 23306, 23311, 23312, 23314, 23315, 23378, 23379, 23380, 23394, 23475, 23504, 23534, 23542, 23543, 23552, 23856, 23886, 23887, 23889, 23890, 23891, 23892, 23893, 23894, 23895, 23899, 23900, 23901, 23902, 23903, 23904, 23905, 23910, 23911, 23912, 23913, 23914, 23915, 23916, 23917, 23921, 23922, 23923, 23961, 23962, 23963, 23964, 23966, 23967, 23968, 23969, 23970, 23971, 23972, 23973, 23979, 23980, 23981, 23982, 24023, 24029, 24031, 24032, 24033, 24060, 24070, 24072, 24074, 24077, 24078, 24079, 24219, 24229, 24266, 24267, 24553, 24634, 24952, 24953, 24981, 25029, 25039, 25053, 25067, 25068, 25069, 25073, 25074, 25075, 25079, 25080, 25081, 25085, 25086, 25090, 25091, 25092, 25099, 25100, 25101, 25105, 25106, 25107, 25111, 25135, 25139, 25140, 25141, 25167, 25249, 25250, 25251, 25313, 25334, 25393, 25444, 25445, 25446, 25447, 25453, 25454, 25455, 25456, 25457, 25458, 25459, 25462, 25463, 25475]

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        r"""Get rigid transform matrix from rotation matrix and translation vector.
        Args:
            rotation (array): (3, 3)
            translation (array): (3,)

        Returns:
            transform: (4, 4)
        """
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

class VideoDatasetScannet(AbstractDataset):
    def __init__(self, cfg, root_path, data_dict, split):
        name = "RGBD_dmatch"
        super(VideoDatasetScannet, self).__init__(name, split, root_path)
        self.cfg = cfg
        self.root_path = root_path
        self.split = split
        self.num_views = cfg['num_views']
        self.view_spacing = cfg['view_spacing']
        self.image_dim = cfg['img_dim']
        self.match_pairs = f'{cfg["match_pairs_root"]}/Scannet_matchs_{split}_{self.view_spacing}0/matches/'
        # self.match_pairs = f'{cfg["match_pairs_root"]}/lightglue/lightglue_ScanNet_pairs_{split}_{self.view_spacing}/matches/'
        self.img_scale = cfg['img_scale']
        self.window_size = cfg['window_size']
        self.use_augmentation = False
        self.URR_Superglue = True
        self.aug_rotation = 1.0
        self.aug_noise = 0.005
        self.data_dict = data_dict
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(self.image_dim),
                transforms.CenterCrop(self.image_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
            ]
        )

        # strided:      (0, 10), (10, 20), (20, 30), etc
        # non-strided:  (0, 10), ( 1, 11), ( 2, 12), etc
        strided = split in ["valid", "test"]
        self.instances = self.dict_to_instances(self.data_dict, strided=strided)
        # if self.view_spacing==2:
        #     remove_20_test.sort(reverse=True)
        #     for index in remove_20_test:
        #         self.instances.pop(index)
        #
        # elif self.view_spacing==5:
        #     remove_50_test.sort(reverse=True)
        #     for index in remove_50_test:
        #         self.instances.pop(index)

        self.read_superglue = True
        self.read_lightglue = False
        print("================================")
        print(f"Stats for {self.name} - {split}")
        print(f"Numer of instances {len(self.instances)}")
        print("Configs:")
        print(cfg)
        print("================================")

    def __len__(self):
        return len(self.instances)

    def dep_transform(self, dep):
        # h, w = dep.shape
        # left = int(round((w - h) / 2.0))
        # right = left + h
        # dep = dep[:, left:right]
        dep = torch.Tensor(dep[None,None,:,:]).float()
        dep = torch.nn.functional.interpolate(dep, (self.image_dim[0], self.image_dim[1]))[0]
        return dep

    def __getitem__(self, index):
        cls_id, s_id, f_ids = self.instances[index]
        s_instance = self.data_dict[cls_id][s_id]["instances"]
        output = {"uid": index, "class_id": cls_id, "sequence_id": s_id}
        output['index'] = index
        output['scene_name'] = s_id.split('_seq')[0]
        output['ref_frame'] = f_ids[1]
        output['src_frame'] = f_ids[0]

        # read superglue matches

        if self.read_superglue:
            match_path = os.path.join(self.match_pairs, f"{s_id}_{f_ids[0]}_{f_ids[1]}_matches.npz")
            npz = np.load(match_path)
            kpts0 = npz['keypoints0']
            kpts1= npz['keypoints1']
            conf = npz['match_confidence']
            matches = npz['matches']
            valid = matches > -1
            if self.img_scale is not None:
                mkpts0 = kpts0 // self.img_scale
                mkpts1 = kpts1 // self.img_scale
            else:
                mkpts0 = kpts0
                mkpts1 = kpts1

            mkpts0_corres = mkpts0[np.where(conf>0.4)]
            mkpts1_corres = mkpts1[matches[np.where(conf>0.4)]]
        elif self.read_lightglue:
            match_path = os.path.join(self.match_pairs, f"{s_id}_{f_ids[0]}_{s_id}_{f_ids[1]}_matches.npz")
            npz = np.load(match_path)
            kpts0 = npz['matchs0']
            kpts1 = npz['matches1']
            if self.img_scale is not None:
                mkpts0 = kpts0 // self.img_scale
                mkpts1 = kpts1 // self.img_scale
            else:
                mkpts0 = kpts0
                mkpts1 = kpts1
            mkpts0_corres = mkpts0
            mkpts1_corres = mkpts1
        # get mask from superglue
        img0_mask_map = torch.zeros((self.image_dim[1], self.image_dim[0]))
        img1_mask_map = torch.zeros((self.image_dim[1], self.image_dim[0]))
        meshgrid = torch.meshgrid(torch.tensor(range(self.image_dim[1])), torch.tensor(range(self.image_dim[0])))
        id_coords = torch.stack(meshgrid, dim=0).permute(2,1,0)
        if self.read_superglue:
            if self.window_size is None:
                img0_mask_map[mkpts0[valid][:, 0], mkpts0[valid][:, 1]] = 1
                img1_mask_map[mkpts1[matches[valid]][:,0],mkpts1[matches[valid]][:,1]] = 1
            else:
                w = self.window_size
                for i in range(len(mkpts0[valid])):
                    # 640
                    left_up_0 = max(int(mkpts0[valid][i][0] - w), 0)
                    right_up_0 = min(int(mkpts0[valid][i][0] + w), img0_mask_map.shape[0])

                    left_up_1 = max(int(mkpts1[matches[valid]][i][0] - w), 0)
                    right_up_1 = min(int(mkpts1[matches[valid]][i][0] + w), img1_mask_map.shape[0])

                    # 480
                    left_down_0 = max(int(mkpts0[valid][i][1] - w), 0)
                    right_down_0 = min(int(mkpts0[valid][i][1] + w), img0_mask_map.shape[1])

                    left_down_1 = max(int(mkpts1[matches[valid]][i][1] - w), 0)
                    right_down_1 = min(int(mkpts1[matches[valid]][i][1] + w), img1_mask_map.shape[1])

                    box_coords_src = id_coords[left_up_0:right_up_0, left_down_0:right_down_0, :]
                    box_coords_tgt = id_coords[left_up_1:right_up_1, left_down_1:right_down_1, :]

                    if len(box_coords_src) > 0 and len(box_coords_tgt) > 0:
                        img0_mask_map[left_up_0:right_up_0, left_down_0:right_down_0] = 1
                        img1_mask_map[left_up_1:right_up_1, left_down_1:right_down_1] = 1
        elif self.read_lightglue:
            if self.window_size is None:
                img0_mask_map[mkpts0[:, 0], mkpts0[:, 1]] = 1
                img1_mask_map[mkpts1[:, 0], mkpts1[:, 1]] = 1
            else:
                w = self.window_size
                for i in range(len(mkpts0)):
                    # 640
                    left_up_0 = max(int(mkpts0[i][0] - w), 0)
                    right_up_0 = min(int(mkpts0[i][0] + w), img0_mask_map.shape[0])

                    left_up_1 = max(int(mkpts1[i][0] - w), 0)
                    right_up_1 = min(int(mkpts1[i][0] + w), img1_mask_map.shape[0])

                    # 480
                    left_down_0 = max(int(mkpts0[i][1] - w), 0)
                    right_down_0 = min(int(mkpts0[i][1] + w), img0_mask_map.shape[1])

                    left_down_1 = max(int(mkpts1[i][1] - w), 0)
                    right_down_1 = min(int(mkpts1[i][1] + w), img1_mask_map.shape[1])

                    box_coords_src = id_coords[left_up_0:right_up_0, left_down_0:right_down_0, :]
                    box_coords_tgt = id_coords[left_up_1:right_up_1, left_down_1:right_down_1, :]

                    if len(box_coords_src) > 0 and len(box_coords_tgt) > 0:
                        img0_mask_map[left_up_0:right_up_0, left_down_0:right_down_0] = 1
                        img1_mask_map[left_up_1:right_up_1, left_down_1:right_down_1] = 1

        ######## get ref points ############
        ref_rgb = self.get_rgb(s_instance[f_ids[1]]["rgb_path"])
        # -- Transform K to handle image resize and crop
        # K = s_instance[f_ids[1]]["intrinsic"][:3, :3].copy()
        # K = adjust_intrinsic(K, [640, 480], [self.image_dim[-1], self.image_dim[0]])
        # K = torch.tensor(K,dtype=torch.float32)
        # output["K"] = K
        # transform and save rgb
        ref_rgb = self.rgb_transform(ref_rgb)
        # gray_rgb = self.rgb_transform(gray_img)
        output["ref_rgb_path"] = os.path.join(self.root, s_instance[f_ids[1]]["rgb_path"])
        # output[f"gray_rgb_{i}"] = torch.from_numpy(gray_img)
        # Resize depth and scale to meters according to ScanNet Docs

        K = s_instance[f_ids[1]]["intrinsic"][:3, :3].copy()
        output['K'] = K
        K = adjust_intrinsic(K, [640, 480], [self.image_dim[-1], self.image_dim[0]])
        K = torch.tensor(K, dtype=torch.float32)
        ##load K adjust##
        dep_ref = self.get_img(s_instance[f_ids[1]]["dep_path"])
        dep_ref = self.dep_transform(dep_ref)
        dep_ref = dep_ref / 1000.0
        output["ref_dep_path"] = os.path.join(self.root, s_instance[f_ids[1]]["dep_path"])
        # generate pointcloud
        ref_pc = grid_to_pointcloud(K.inverse(), dep_ref)
        ref_pc_valid = ref_pc[(ref_pc[:, 2] > 0.) & (ref_pc[:, 2] < 65.)]
        ref_valid_inds2d = torch.stack(
            [torch.where((dep_ref[0] > 0) & (dep_ref[0] < 65.))[1],
             torch.where((dep_ref[0] > 0) & (dep_ref[0] < 65.))[0]], dim=-1)
        ref_pc_piror_inds = torch.where(img1_mask_map[ref_valid_inds2d[:, 0], ref_valid_inds2d[:, 1]] == 1)[0]

        ref_pc_rgb = (ref_rgb * 0.5 + 0.5).view(3, -1).transpose(0, 1)
        ref_pc_rgb = ref_pc_rgb[ref_pc[:, 2] != 0]
        output["ref_points_rgb"] = ref_pc_rgb

        output["ref_points"] = ref_pc_valid
        output["ref_pc_piror_inds"] = ref_pc_piror_inds
        output["ref_points_feat"] = torch.ones((ref_pc_valid.shape[0], 1), dtype=torch.float32)
        output[f"ref_path"] = os.path.join(self.root_path, s_instance[f_ids[1]]["rgb_path"])

        ####### get src points #######
        src_rgb = self.get_rgb(s_instance[f_ids[0]]["rgb_path"])
        # -- Transform K to handle image resize and crop
        # K = s_instance[f_ids[0]]["intrinsic"][:3, :3].copy()
        # K = adjust_intrinsic(K, [640, 480], [self.image_dim[-1], self.image_dim[0]])
        # K = torch.tensor(K).float()
        # output["K"] = K
        # transform and save rgb
        src_rgb = self.rgb_transform(src_rgb)
        output["src_rgb_path"] = os.path.join(self.root, s_instance[f_ids[0]]["rgb_path"])

        # Resize depth and scale to meters according to ScanNet Docs
        # http://kaldir.vc.in.tum.de/scannet_benchmark/documentation
        dep_src = self.get_img(s_instance[f_ids[0]]["dep_path"])
        dep_src = self.dep_transform(dep_src)
        dep_src = dep_src / 1000.0
        output["src_dep_path"] = os.path.join(self.root, s_instance[f_ids[0]]["dep_path"])
        # generate pointcloud
        P_src = s_instance[f_ids[0]]["extrinsic"].copy()
        P_src = torch.tensor(P_src, dtype=torch.float32)
        P_ref = s_instance[f_ids[1]]["extrinsic"].copy()
        P_ref = torch.tensor(P_ref, dtype=torch.float32)
        P_relative = torch.matmul(torch.linalg.inv(P_ref), P_src)
        output["transform"] = P_relative[:3, :]

        src_pc = grid_to_pointcloud(K.inverse(), dep_src)
        src_pc_valid = src_pc[(src_pc[:, 2] > 0.) & (src_pc[:, 2] < 65.)]
        src_valid_inds2d = torch.stack(
            [torch.where((dep_src[0] > 0) & (dep_src[0] < 65.))[1],
             torch.where((dep_src[0] > 0) & (dep_src[0] < 65.))[0]], dim=-1)
        ##random downsample##
        # n = np.random.choice(len(src_pc_valid), len(src_pc_valid) // 8, replace=False)
        # src_pc_valid = src_pc_valid[n]
        # src_valid_inds2d = src_valid_inds2d[n]
        src_pc_piror_inds = torch.where(img0_mask_map[src_valid_inds2d[:, 0], src_valid_inds2d[:, 1]] == 1)[0]

        src_pc_rgb = (src_rgb * 0.5 + 0.5).view(3, -1).transpose(0, 1)
        src_pc_rgb = src_pc_rgb[src_pc[:, 2] != 0]
        output["src_points_rgb"] = src_pc_rgb
        output["src_points"] = src_pc_valid  # valid points
        output["src_pc_piror_inds"] = src_pc_piror_inds  # valid points prior index
        output["src_points_feat"] = torch.ones((src_pc_valid.shape[0], 1), dtype=torch.float32)
        # Some rotation conversions -- absolute -> relative reference
        # Read in separate instances
        # Set identities for xyz and quat
        output["src_path"] = os.path.join(self.root_path, s_instance[f_ids[0]][f"rgb_path"])

        if self.use_augmentation:
            rotation = P_relative[:3, :3]
            translation = P_relative[:3, 3]
            aug_src = np.random.rand(1)[0]
            ref_points, src_points, rotation, translation, rot_ab = self._augment_point_cloud(
                ref_pc_valid, src_pc_valid, rotation, translation, aug_src
            )
            transform = get_transform_from_rotation_translation(rotation, translation)
            output["transform"] = torch.from_numpy(transform[:3, :])
            output["src_points"] = src_points
            output["ref_points"] = ref_points
        output["transform"] = torch.tensor(output["transform"], dtype=torch.float32)

        # superglue correspondences 3dinds from superglue
        if self.URR_Superglue:
            ref_pc_3dins = torch.where([(ref_pc[:, 2] > 0.) & (ref_pc[:, 2] < 65.)][0] == True)[0]
            src_pc_3dins = torch.where([(src_pc[:, 2] > 0.) & (src_pc[:, 2] < 65.)][0] == True)[0]
            corres = []
            for arr1, arr0 in zip(torch.tensor(mkpts1_corres), torch.tensor(mkpts0_corres)):
                ids1 = torch.where(ref_pc_3dins == arr1[1] * self.image_dim[-1] + arr1[0])[0]
                ids0 = torch.where(src_pc_3dins == arr0[1] * self.image_dim[-1] + arr0[0])[0]
                if ids1.shape[0] != 0 and ids0.shape[0] != 0:
                    corres.append([ids1, ids0])
            corres = torch.tensor(corres)

            # ransac_corres_path = os.path.join(f'/home/levi/Data_work/workplace/Geobyoc_overlap_embedding_superglue_ransac/rasac_3dmatch_20/{self.split}',
            #     f"{output['scene_name']}-{output['ref_frame']}-{output['src_frame']}.npz")
            # ransac_npz = np.load(ransac_corres_path)
            # ransac_corres = ransac_npz['correspondence_set']

            # result,_ = registration_with_ransac_from_correspondences(src_points[corres[:,1]],
            #     ref_points[corres[:,0]],
            #     distance_threshold=0.05,
            #     ransac_n=3,
            #     num_iterations=1000)
            # ransac_corres = np.array(result.correspondence_set)
            # output['3d_corres_est'] = corres[ransac_corres[:,0]]

            output['3d_corres_est'] = corres
        return output


    def _augment_point_cloud(self, ref_points, src_points, rotation, translation, aug_src):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        aug_rotation = np.array(aug_rotation, dtype=np.float32)
        aug_rotation = torch.from_numpy(aug_rotation)

        if aug_src > 0.5:
            src_points = torch.mm(src_points, aug_rotation)
            rotation = torch.mm(rotation,aug_rotation)
        else:
            ref_points = torch.matmul(ref_points, aug_rotation)
            rotation = torch.matmul(torch.linalg.inv(aug_rotation), rotation)
            translation = np.matmul(translation, aug_rotation)

        ref_points += (torch.rand(ref_points.shape[0], 3) - 0.5) * torch.tensor(self.aug_noise, dtype=torch.float32)
        src_points += (torch.rand(src_points.shape[0], 3) - 0.5) * torch.tensor(self.aug_noise, dtype=torch.float32)

        return ref_points, src_points, rotation, translation, aug_rotation

    def transform_points_Rt(self,
            points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
    ):
        H, W = viewpoint.shape
        assert H == 3 and W == 4, "Rt is B x 3 x 4 "
        t = viewpoint[:, 3]
        r = viewpoint[:, 0:3]

        # transpose r to handle the fact that P in num_points x 3
        # yT = (RX) + t = (XT @ RT) + t
        # r = r.transpose(0,1).contiguous()

        # invert if needed
        if inverse:
            points = points - t
            # points = points.mm(r.inverse())
            points = points.mm(r)
        else:
            points = points.mm(r.inverse())
            points = points + t

        return points

    def dict_to_instances(self, data_dict, strided):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # populate dictionary
        for cls_id in data_dict:
            for s_id in data_dict[cls_id]:
                frames = list(data_dict[cls_id][s_id]["instances"].keys())
                frames.sort()

                if strided:
                    frames = frames[:: self.view_spacing]
                    stride = 1
                else:
                    stride = self.view_spacing

                num_frames = len(frames)

                for i in range(num_frames - self.num_views * stride):
                    f_ids = []
                    for v in range(self.num_views):
                        f_ids.append(frames[i + v * stride])

                    # Hacky way of getting source to be in the middle for triplets
                    mid = self.num_views // 2
                    f_ids = f_ids[mid:] + f_ids[:mid]
                    instances.append([cls_id, s_id, tuple(f_ids)])

        return instances


