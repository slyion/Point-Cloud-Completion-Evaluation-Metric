import torch
from Chamfer3D_L1L2.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def get_loss(pcds_pred, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
    else:
        CD = chamfer

    P3 = pcds_pred
    cd3 = CD(P3, gt)  ##


    return  cd3* 1e3


