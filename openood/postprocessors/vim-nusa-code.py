from typing import Any
import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv, qr
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
from .base_postprocessor import BasePostprocessor

class VIMNuSAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False
        
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()
            with torch.no_grad():
                self.w, self.b = net.get_fc()
                print('Extracting id training feature')
                feature_id_train = []
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()
                    _, feature = net(data, return_feature=True)
                    feature_id_train.append(feature.cpu().numpy())
                feature_id_train = np.concatenate(feature_id_train, axis=0)
                logit_id_train = feature_id_train @ self.w.T + self.b
            
            # 计算中心点 (同 VIM 方法)
            self.u = -np.matmul(pinv(self.w), self.b)
            
            # 为了更好的数值稳定性，使用QR分解计算列空间的正交基
            # 这与原始 NuSA 论文中的方法一致
            q, r = qr(self.w.T)  # q 包含列空间的正交基
            self.column_space = q
            
            # 计算训练集的 NuSA 分数
            # NuSA 分数是特征向量在列空间上的投影范数
            nusa_scores_id_train = norm(np.matmul(feature_id_train - self.u, self.column_space), axis=-1)
            
            # 计算缩放因子 α，用于将 NuSA 分数转换为虚拟 logit
            # α 确保虚拟 logit 与真实 logit 在量级上是可比的
            self.alpha = logit_id_train.max(axis=-1).mean() / nusa_scores_id_train.mean()
            print(f'{self.alpha=:.4f}')
            
            self.setup_flag = True
        else:
            pass
            
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = net.forward(data, return_feature=True)
        feature_ood = feature_ood.cpu().numpy()
        logit_ood = torch.tensor(feature_ood @ self.w.T + self.b)
        _, pred = torch.max(logit_ood, dim=1)
        
        # 计算 NuSA 分数（特征在列空间上的投影）
        nusa_score = norm(np.matmul(feature_ood - self.u, self.column_space), axis=-1)
        
        # 将 NuSA 分数转换为虚拟 logit
        virtual_logit = nusa_score * self.alpha
        
        # 将虚拟 logit 与原始 logit 拼接
        # 这里我们需要将 virtual_logit 转换为与 logit_ood 相同的形状
        virtual_logit = torch.tensor(virtual_logit).unsqueeze(1)
        extended_logits = torch.cat([logit_ood, virtual_logit], dim=1)
        
        # 计算扩展后的 softmax
        softmax_probs = torch.softmax(extended_logits, dim=1)
        
        # VIM-NuSA 分数是虚拟类的 softmax 概率
        # 较高的分数表示样本更可能是 OOD
        vim_nusa_score = softmax_probs[:, -1].numpy()
        
        # 为保持与原始 VIM 分数方向一致，取负值
        # 这样较高的分数表示分布内样本，较低的分数表示分布外样本
        score_ood = -vim_nusa_score
        
        return pred, torch.from_numpy(score_ood)
        
    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]
        
    def get_hyperparam(self):
        return self.dim
