import numpy as np
import torch.nn as nn
from sklearn.cluster import DBSCAN
from torch.autograd import Variable
import torch
import copy
import cv2
from tools.lanenet_postprocess import LaneNetPostProcessor

def Vectorization(S, C, D):
    """
    segmentation S, cluster embedding C, direction D
    return Vectorized HDmap
    """
    # S:bs*4*200*200 C:bs*200*200

    device = C.device
    bs, H, W = C.shape
    processor = LaneNetPostProcessor()
    # C_2 = copy.deepcopy(C_0)
    # mask_list = []
    # for i in range(C_2.max()+1):
    #     mask_list.append((C_2==i).int())
    # C = torch.stack(mask_list)
    # print("C",C.shape)
    # n_feature, bs, H, W = C.shape
    # S = S.permute(0, 2, 3, 1).argmax(3).int().cpu().numpy()

    # C = C.permute(1, 2, 3, 0).cpu().numpy()  # bs, H, W, n_feature
    # print("C 2 ",C.shape)
    # C = C[S[:, :, :, None].expand(-1, -1, -1, n_feature) != 3]
    # print(C.shape)
    # for i in S[0]:
    #     print(i)
    # print("S[0]", torch.count_nonzero(S[0]))
    # C = C * S[0]
    # print("C",C.shape)
    # lines_cloud_list = []
    # vectorlized_lines = []
    # for i in range(bs):
    #     embedding = np.stack([C[i][:, :, j][S[i] != 3] for j in range(n_feature)], axis=1)
    #     if embedding.shape[0] == 0:
    #         print('cluster=0')
    #         embedding = np.zeros([H*W, n_feature])
    #     else:
    #         print('cluster>0')
    #         print('embedding',embedding.shape)
    #     line_cloud = DBSCAN(eps=3, min_samples=4).fit_predict(embedding)
    #     print("line_cloud", line_cloud.shape)
    #     lines_cloud_list.append(line_cloud)
    # lines_cloud = np.stack(lines_cloud_list, axis=0)
    # # print("lines_cloud", lines_cloud.shape)
    #
    # for line_cloud in lines_cloud:
    for i in range(bs):
        # print("C[i]",C[i].shape)
        binary_mask = torch.argmax(S[i], dim=0).cpu().numpy()
        binary_mask[np.where(binary_mask != 0)] = 1
        # print("binary_mask", binary_mask.shape)
        # for i in binary_mask:
        #     print(i)
        mask_image, lane_coords, line_clouds = processor.postprocess(binary_mask, C[i].cpu().numpy(), S[i].cpu().numpy())
        # print("mask_image", mask_image.shape, "lane_coords", len(lane_coords))  # lane_coords车道线各像素点坐标

        # cv2.imwrite('save/mask/'
        #             + str(i)
        #             + ".jpg", mask_image)
        for i in range(len(lane_coords)):
            # print("line_cloud", line_cloud.shape) # 100*100
            sparse_line = Directional_NMS(torch.tensor(line_clouds[i]).to(device).reshape(1,1,H,W), lane_coords[i])
            vector_line = Connect_Line(sparse_line, D)
            vectorlized_lines.append(vector_line)
    return vectorlized_lines


def Directional_NMS(L, coordinates):
    """
    line point cloud L with confidence
    return sparse line point cloud L_sparse
    """
    # print("L",L.shape)
    # for i in L[0,0]:
    #     print(i)
    pool_index = 5
    mp_vertival = nn.MaxPool2d((1, pool_index))
    mp_horizontal = nn.MaxPool2d((pool_index, 1))
    ap_vertical = nn.AvgPool2d((5, 10))
    ap_horizontal = nn.AvgPool2d((10, 5))

    L_float = L.float()
    _, _, H, W = L.shape
    L_sparse = torch.zeros((H, W),dtype=int)

    # print(H,W)
    # print("ap_vertical", ap_vertical(L_float).shape, "ap_horizontal", ap_horizontal(L_float).shape)
    # print("mp_horizontal(L_float)",mp_horizontal(L_float))
    for index_x in range(H):
        for index_y in range(W):
            if L[0, 0, index_x, index_y] == 0:
                continue
            # print("point", L[0, 0, index_x, index_y], index_x, index_y)
            # print(ap_vertical(L_float)[0, 0, index_x//5, index_y//9], ap_horizontal(L_float)[0, 0, index_x//9, index_y//5])
            if ap_vertical(L_float)[0, 0, index_x//5, index_y//10] > ap_horizontal(L_float)[0, 0, index_x//10, index_y//5]:
                if mp_horizontal(L_float)[0, 0, index_x//pool_index, index_y] == L_float[0, 0, index_x, index_y]:
                    # print(mp_horizontal(L_float)[0, 0, index_x//5, index_y])
                    L_sparse[index_x, index_y] = 1
                # else:
                    # print("miss")
            else:
                if mp_vertival(L_float)[0, 0, index_x, index_y//pool_index] == L_float[0, 0, index_x, index_y]:
                    # print("hit2")
                    L_sparse[index_x, index_y] = 1
                # else:
                    # print("miss")
    # print("L_sparse",L_sparse.shape)
    # for i in L_sparse:
    #     print(i)
    # L_sparse= L_sparse.numpy()
    # L_sparse = np.where(L_sparse == 1, 255.0, 0)
    # cv2.imwrite('save/sparse/image'
    #             + ".jpg", L_sparse)
    return L_sparse


def Connect_Line(L, D):
    """
    line point cloud L, direction D
    return vector line L_vector
    """
    # print("L",L.shape,"D",D.shape)
    # for i in L:
    #     print(i)
    idx, idy = np.where(L == 1)
    # print("idx.shape",idx.shape)
    index_list = np.random.choice(np.linspace(start=0, stop=idx.shape[0]-1, num=idx.shape[0]).astype(int), size=idx.shape[0]//10, replace=False)
    index_list.sort()
    print("idx",idx[index_list])
    print("idy",idy[index_list])
    line_1 = Connect_One_Direction(idx[index_list], idy[index_list], L, D)
    line_2 = Connect_One_Direction(idx[index_list][::-1], idy[index_list][::-1], L, D)
    L_vector = np.concatenate((line_1, reverse(line_2)))
    return L_vector


def reverse(line):
    """
    reverse line direction
    e.g. direction defines as clockwise from the front, line[1,0,0,0,0,0,0,0] means front
         reverse(line) returns [0,0,0,0,1,0,0,0]
    """
    half_index = len(line)
    tmp_line = line[:half_index].copy()
    line[:half_index] = line[half_index:]
    line[half_index:] = tmp_line
    return line


def Connect_One_Direction(index_x, index_y, L, D):
    """
    start_point p (x,y), line point cloud L (n*2), direction D (200*200)
    return one direction connected vector line L_vector
    """
    idx, idy = index_x, index_y
    step = 1
    threshold = 5
    L_vector = []
    begin_idx, begin_idy  = idx[0], idy[0]
    while idx.shape[0] > 0:
        # # Direction pd of p is not taken
        # if D[p] != 0:
        #     D[p] = 0
        # else:
        #     break
        direction = D[:, begin_idx, begin_idy]
        np.tan(direction*10)
        p_target = p + D[p] * step
        dist1 = np.sqrt(((L - p) ** 2).sum(1))
        dist2 = np.sqrt(((L - p_target) ** 2).sum(1))
        p_next = L[0]
        next_dist = np.sqrt(((p_next - p_target) ** 2).sum())
        del_list = []
        for i, j in enumerate(dist1):
            if j < step - 1:
                del_list.append(i)
            else:
                if dist2[i] < next_dist:
                    p_next = L[i]
        L = np.delete(L, del_list)
        if np.sqrt(((p_next - p) ** 2).sum()):
            break
        L_vector.append(p_next)
        p = p_next
    return L_vector


# Instance embedding
class DiscriminativeLoss(nn.Module):

    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=True):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target, n_clusters):
        target.requires_grad = False
        return self._discriminative_loss(input, target, n_clusters)

    def _discriminative_loss(self, input, target, n_clusters):
        bs, n_features, height, width = input.size()
        max_n_clusters = target.size(1)

        input = input.contiguous().view(bs, n_features, height * width)
        target = target.contiguous().view(bs, max_n_clusters, height * width)

        c_means = self._cluster_means(input, target, n_clusters)
        l_var = self._variance_term(input, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means, n_clusters)
        l_reg = self._regularization_term(c_means, n_clusters)
        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg
        return loss

    def _cluster_means(self, input, target, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, 1, max_n_clusters, n_loc
        target = target.unsqueeze(1)
        # bs, n_features, max_n_clusters, n_loc
        input = input * target

        means = []
        for i in range(bs):
            # n_features, n_clusters, n_loc
            input_sample = input[i, :, :n_clusters[i]]
            # 1, n_clusters, n_loc,
            target_sample = target[i, :, :n_clusters[i]]
            # n_features, n_cluster
            mean_sample = input_sample.sum(2) / (target_sample.sum(2) + 1e-8)

            # padding
            n_pad_clusters = max_n_clusters - n_clusters[i]
            assert n_pad_clusters >= 0
            if n_pad_clusters > 0:
                pad_sample = torch.zeros(n_features, n_pad_clusters).to(mean_sample.device)
                pad_sample = Variable(pad_sample)
                # if self.usegpu:
                #     pad_sample = pad_sample.cuda()
                mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        # bs, n_features, max_n_clusters
        means = torch.stack(means)

        return means

    def _variance_term(self, input, target, c_means, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, max_n_clusters, n_loc
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 1) -
                           self.delta_var, min=0) ** 2) * target

        var_term = 0
        for i in range(bs):
            if n_clusters[i] == 0:
                continue
            # n_clusters, n_loc
            var_sample = var[i, :n_clusters[i]]
            # n_clusters, n_loc
            target_sample = target[i, :n_clusters[i]]

            # n_clusters
            c_var = var_sample.sum(1) / (target_sample.sum(1) + 1e-8)
            var_term += c_var.sum() / (n_clusters[i] + 1e-8)
        var_term /= bs

        return var_term

    def _distance_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        dist_term = 0
        for i in range(bs):
            if n_clusters[i] <= 1:
                continue

            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = mean_sample.unsqueeze(2).expand(n_features, n_clusters[i], n_clusters[i])
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters[i].item())).to(diff.device)
            margin = Variable(margin)
            # if self.usegpu:
            #     margin = margin.cuda()
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (2 * n_clusters[i] * (n_clusters[i] - 1) + 1e-8)
        dist_term /= bs

        return dist_term

    def _regularization_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]
            reg_term += torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term /= bs

        return reg_term
