"""
稀疏图注意力网络模型定义
用于出租车司机异常行为检测的图神经网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax
from torch_geometric.utils import add_self_loops


class SparseGraphAttentionLayer(nn.Module):
    """
    稀疏图注意力层
    
    实现基于注意力机制的图神经网络层，用于处理稀疏图结构
    支持多头注意力机制，能够学习节点间的重要性权重
    """
    
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """
        初始化稀疏图注意力层
        
        参数:
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            dropout (float): Dropout概率
            alpha (float): LeakyReLU负斜率参数
            concat (bool): 是否在多头注意力中拼接输出
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 权重矩阵和注意力向量
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W)  # Xavier初始化
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 节点特征矩阵 (N, in_features)
            edge_index (torch.Tensor): 边索引 (2, E)
        
        返回:
            torch.Tensor: 更新后的节点特征 (N, out_features)
        """
        # 添加自环边
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 线性变换
        Wh = x @ self.W  # (N, out_features)

        # 准备边索引
        src, dst = edge_index  # 源节点和目标节点索引
        Wh_src = Wh[src]  # 源节点特征 (E, out_features)
        Wh_dst = Wh[dst]  # 目标节点特征 (E, out_features)

        # 计算注意力系数
        a_input = torch.cat([Wh_src, Wh_dst], dim=1)  # 拼接源节点和目标节点特征
        e = self.leakyrelu((a_input @ self.a).squeeze(-1))  # 注意力分数

        # 对每个目标节点的注意力进行softmax归一化
        alpha = scatter_softmax(e, dst)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # 消息传递：邻居特征的加权求和
        m = Wh_src * alpha.unsqueeze(-1)  # 加权消息
        h_prime = scatter_add(m, dst, dim=0, dim_size=x.size(0))  # 聚合消息

        return F.elu(h_prime) if self.concat else h_prime


class SparseGAT(nn.Module):
    """
    稀疏图注意力网络
    
    基于多头注意力机制的图神经网络，用于节点分类任务
    支持稀疏图结构，能够处理大规模图数据
    """
    
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """
        初始化稀疏图注意力网络
        
        参数:
            nfeat (int): 输入特征维度
            nhid (int): 隐藏层维度
            nclass (int): 输出类别数
            dropout (float): Dropout概率
            alpha (float): LeakyReLU负斜率参数
            nheads (int): 注意力头数量
        """
        super().__init__()
        self.dropout = dropout

        # 多头稀疏注意力层
        self.attentions = nn.ModuleList([
            SparseGraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        # 输出层
        self.out_att = SparseGraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, concat=False)

    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 节点特征矩阵 (N, nfeat)
            edge_index (torch.Tensor): 边索引 (2, E)
        
        返回:
            torch.Tensor: 节点预测结果 (N,)
        """
        x = F.dropout(x, self.dropout, training=self.training)
        # 拼接多头注意力输出
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.out_att(x, edge_index)
        return x.squeeze()
