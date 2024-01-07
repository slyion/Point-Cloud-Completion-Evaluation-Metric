
import torch
from Chamfer3D_L1L2.loss_utils import get_loss  as get_lossl1l2
from utils_F.metrics import Metrics
import os
import sys
sys.path.append("./emd/")
import emd_module as emd

EMD = emd.emdModule()






def  getGPU():

    GPUlist=[0,1,2,3]
    if GPUlist[0] >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        strTemp = 'cuda:' + str(GPUlist[0])
        strTemp = 'cuda:0'
        device = torch.device(strTemp) if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')

    return device

def creattensor():
    # 创建一个大小为2x3的张量（Tensor），数据类型为float
    gt = torch.Tensor([
        [[1, 2, 3], [4, 5, 6],[1, 2, 3], [4, 5, 6]],
        [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]
                     ])
    pre_end = torch.Tensor([
        [[1, 2, 3], [4, 5, 6],[1, 2, 3], [4, 5, 6]],
        [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]]
                 ])
    return gt,pre_end


if __name__ == '__main__':

    # # 获取数据  方法1
    # gt,pre_end = creattensor()
    # device = getGPU()
    # gt = gt.to(device)
    # pre_end = pre_end.to(device)

    # 获取数据  方法2
    gt = torch.rand(4, 8192, 3).cuda() # please normalize your point cloud to [0, 1]
    pre_end = torch.rand(4, 8192, 3).cuda()
    print(gt.shape,pre_end.shape)
    # ##############################################
    # 主 L1  L2
    xxxxxx = get_lossl1l2(pre_end, gt, sqrt=True)  # False True
    xxgt_false = get_lossl1l2(pre_end, gt, sqrt=False)  # False True

    print("L1 ,L2",xxxxxx,xxgt_false)

    ##################################################
    ###  F_score
    for batch_i in range(pre_end.shape[0]):
        _metrics = Metrics.get((pre_end.detach())[batch_i], (gt.detach())[batch_i])
        print("F_score",_metrics)
    ############################
    # ###  EMD
    dist, _ = EMD(pre_end, gt, 0.002, 10000)
    # dist, _ = EMD(pre_gt, gt, 0.005, 50)
    emd1 = torch.sqrt(dist).mean()

    print("emd1",emd1)



