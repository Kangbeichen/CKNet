import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parameter import Parameter
import math

    # paper------------''When Radiology Report Generation Meets Knowledge Graph ''
    # 获取graph中各个节点的初始化特征---------------Node Feature Initialization
class ClsAttention(nn.Module):  # spatial attention

    def __init__(self, feat_size, num_classes):  #2048,30
        super().__init__()
        self.feat_size = feat_size
        self.num_classes = num_classes  #30
        self.channel_w = nn.Conv2d(feat_size, num_classes, 1, bias=False)


    def forward(self, feats):   #cnn_feats:[8, 1024, 16, 16]-->[batch_size, 2048, 7, 7]
        # feats: batch size x feat size x H x W
        batch_size, feat_size, H, W = feats.size()

        # att_maps: batch_size * num_classes(out_channels) * h * w
        # 因为卷积核的size为1, 所以att_maps输出只有channel维度改变
        att_maps = self.channel_w(feats) #（batch_size,30，16，16）--->batch_size,30，7，7）

        # 对应原文:we employ a spatial attention module (node attention module in Figure 2)upon the output activation.
        att_maps = torch.softmax(att_maps.view(batch_size, self.num_classes, -1), dim=2)  #（8,30，256)--->（16,30，49)

        feats_t = feats.view(batch_size, feat_size, H * W).permute(0, 2, 1)   #(8,256,1024)--->(16,49,2048)

        # 对应原文: Then, the initial feature of a node in the graph is obtained as the attention-weighted sum of the activation,
        # where attention weights come from the corresponding channel.
        cls_feats = torch.bmm(att_maps, feats_t)       #H * W^T,线性变换
        return cls_feats     #([8, 30, 1024]) ---->(16,30,2048)    node的数目 *node feature维度


class GCLayer(nn.Module):

    def __init__(self, in_size, state_size):
        super().__init__()
        self.condense = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.condense_norm = nn.BatchNorm1d(state_size)
        self.fw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.fw_norm = nn.BatchNorm1d(state_size)
        self.bw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.bw_norm = nn.BatchNorm1d(state_size)
        self.update = nn.Conv1d(3 * state_size, in_size, 1, bias=False)
        self.update_norm = nn.BatchNorm1d(in_size)
        self.relu = nn.ReLU(inplace=True)
        # v2:
        self.dropout = nn.Dropout(0.5)

    def forward(self, states, fw_A, bw_A):
        # states: batch size x feat size x nodes
        condensed = self.relu(self.condense_norm(self.condense(states)))
        fw_msg = self.relu(self.fw_norm(self.fw_trans(states).bmm(fw_A)))
        bw_msg = self.relu(self.bw_norm(self.bw_trans(states).bmm(bw_A)))
        updated = self.update_norm(self.update(torch.cat((condensed, fw_msg, bw_msg), dim=1)))
        updated = self.relu(self.dropout(updated) + states)
        return updated

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):   #1024,256
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(8, in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters_xavier()

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):  #([2, 1024, 31]),[2, 31, 31]

        x = x.permute(0,2,1)  #[2，31，1024]
        self.weight = Parameter(torch.FloatTensor(x.size()[0], self.in_features, self.out_features).to('cpu'))  #([2, 1024, 256])
        self.reset_parameters_xavier()
        support = torch.bmm(x, self.weight)#[2,31,256]
        output = torch.bmm(adj, support)#[2,31,256]
        if self.bias is not None:
            return  (output + self.bias).permute(0,2,1)  #[2,256,31]
        else:
            return  output.permute(0,2,1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionLayer(nn.Module):

    def __init__(self,in_size, state_size):    #1024,256
        super(GraphConvolutionLayer, self).__init__()
        self.in_size = in_size
        self.state_size = state_size

        self.condense = nn.Conv1d(in_size, state_size, 1)
        self.condense_norm = nn.BatchNorm1d(state_size)

#这里是咋回事
        self.gcn_forward = GraphConvolution(in_size, state_size)
        self.gcn_backward = GraphConvolution(in_size, state_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.conv1d = nn.Conv1d(3*state_size, in_size, 1, bias=False)
        self.norm = nn.BatchNorm1d(in_size)

        self.test_conv = nn.Conv1d(state_size, in_size, 1, bias=False)


    def forward(self, x, fw_A, bw_A):#输入:torch.Size([2, 1024, 31]),[2, 31, 31],[2, 31, 31]

        states = x
        condensed_message = self.relu(self.condense_norm(self.condense(x)))  #([2, 256, 31])
        fw_message = self.relu(self.gcn_forward(x, fw_A))#输入:torch.Size([2, 1024, 31]),[2, 31, 31],输出：([2, 256, 31])
        bw_message = self.relu(self.gcn_backward(x, bw_A))

        update = torch.cat((condensed_message, fw_message, bw_message),dim=1)#([2, 768, 31]，接受邻居信息更新节点
        x = self.norm(self.conv1d(update))  #([2, 1024, 31])
        x = self.relu(x+states)  #([2, 1024, 31])

        return x

class GCN(nn.Module):

    def __init__(self, in_size, state_size):  #1024,256
        super(GCN, self).__init__()

        # in_size:1024, state_size:256
        self.gcn1 = GraphConvolutionLayer(in_size, state_size)
        self.gcn2 = GraphConvolutionLayer(in_size, state_size)
        self.gcn3 = GraphConvolutionLayer(in_size, state_size)

    def forward(self, states, fw_A, bw_A): #[2, 31, 1024],[2, 31, 31],[2, 31, 31]
        # states: batch_size * feature_size(in_size) * number_classes
        states = states.permute(0,2,1)   #[2,1024,31]

        # states: batch_size * number_classes * feature_size(in_size)，fw_A为边的连接关系
        states = self.gcn1(states, fw_A, bw_A)  #输入:torch.Size([2, 1024, 31]),[2, 31, 31],[2, 31, 31]，输出：([2, 1024, 31])
        states = self.gcn2(states, fw_A, bw_A)
        states = self.gcn3(states, fw_A, bw_A)

        return states.permute(0,2,1) #[2, 31, 1024]

class GCNClassifier(nn.Module):

    def __init__(self, num_classes, fw_adj, bw_adj): #30,31,31
        super().__init__()
        self.num_classes = num_classes
        # self.densenet121 = models.densenet121(pretrained=True)
        # feat_size = self.densenet121.classifier.in_features      #1024
        feat_size = 2048
        # self.densenet121.classifier = nn.Linear(feat_size, num_classes)
        self.cls_atten = ClsAttention(feat_size, num_classes)

        self.gcn = GCN(feat_size, 256)

        self.fc2 = nn.Linear(feat_size, num_classes)  #1024--30

        #度矩阵D：把第i行所有元素加起来
        fw_D = torch.diag_embed(fw_adj.sum(dim=1)) #fw_D_(31,31)
        bw_D = torch.diag_embed(bw_adj.sum(dim=1))
        inv_sqrt_fw_D = fw_D.pow(-0.5)     #D^(-1/2)
        inv_sqrt_fw_D[torch.isinf(inv_sqrt_fw_D)] = 0
        inv_sqrt_bw_D = bw_D.pow(-0.5)
        inv_sqrt_bw_D[torch.isinf(inv_sqrt_bw_D)] = 0
        
        self.fw_A = inv_sqrt_fw_D.mm(fw_adj).mm(inv_sqrt_fw_D) #(31,31)  D^(-1/2) * A * D^(-1/2)，将A经过归一化，不改变特征矩阵H原本的分布
        self.bw_A = inv_sqrt_bw_D.mm(bw_adj).mm(inv_sqrt_bw_D)

    def forward(self, att_feats_0, att_feats_1):    #[8, 3, 512, 512]--->[batch_size, 2048, 7, 7]
        batch_size = att_feats_0.size(0)      #16
        device = att_feats_0.device   #新增----解决arguments are located on different GPUs
        fw_A = self.fw_A.repeat(batch_size, 1, 1).to(device)  #fw_A:torch.Size([16, 31, 31])
        bw_A = self.bw_A.repeat(batch_size, 1, 1).to(device)

        # cnn_feats1 = self.densenet121.features(img1) #cnn_feats1:torch.Size([8, 1024, 16, 16] )batch size x feat size x H x W
        # cnn_feats2 = self.densenet121.features(img2)

        #batch_size * num_classes(out_channel)
        # 对应原文: The feature of the global node is initialized with the output of global average pooling
        global_feats1 = att_feats_0.mean(dim=(2, 3)) #global_feats1:torch.Size([8, 1024])，--->(16,2048)
        global_feats2 = att_feats_1.mean(dim=(2, 3))

        # H * W^T,线性变换， graph中各节点的初始特征
        cls_feats1 = self.cls_atten(att_feats_0)
        cls_feats2 = self.cls_atten(att_feats_1) #([8, 30, 1024])--->(16,30,2048)

        #([8, 1024])---unsqueeze(1):torch.Size([8, 1, 1024])
        node_feats1 = torch.cat((global_feats1.unsqueeze(1), cls_feats1), dim=1) #([8, 31, 1024])，节点特征矩阵--->(16,31,2048)
        node_feats2 = torch.cat((global_feats2.unsqueeze(1), cls_feats2), dim=1)

        node_feats1 = node_feats1.contiguous()
        node_feats2 = node_feats2.contiguous()

        node_states1 = self.gcn(node_feats1, fw_A, bw_A) #输入：[8, 31, 1024],[8, 31, 31],[8, 31, 31]，node_states1：([8, 31, 1024])--->(16,31,2048)
        node_states2 = self.gcn(node_feats2, fw_A, bw_A)

        # global average pooling was applied to obtain a graph level feature
        # global_states = node_states1.mean(dim=1) + node_states2.mean(dim=1)  #[8,1024]
        # global_states = node_states1.mean(dim=0) + node_states2.mean(dim=0)  #[31,1024]
        global_states = node_states1 + node_states2  #[8,31,2048]   universal features of each abnormality

        # fully connected layer with Sigmoid activation was used to predict probabilities for each finding as a multi-label classification task
        logits = self.fc2(global_states)[:,:20]   #(8,20)
        # print(logits)
        return global_states
