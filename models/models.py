import numpy as np
import torch
import torch.nn as nn

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor
from modules.mlclassifier import GCNClassifier


class BaseCMNModel(nn.Module):
    def __init__(self, args, fw_adj, bw_adj, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.fw_adj = fw_adj
        self.bw_adj = bw_adj
        self.visual_extractor = VisualExtractor(args)    #extracted by pre-trained convolutional neural networks (CNN), such as VGG  or ResNet
        self.encoder_decoder = BaseCMN(args, tokenizer) #---------------------->
        self.gcn = GCNClassifier(args.num_classes, fw_adj, bw_adj)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        #patch_feats, avg_feats:([batch_size, 49, 2048]),([batch_size, 2048])
        #----patch_feats(att_feats_0):[batch_size, 2048, 7, 7]
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])  #取所有行的第0个数据
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])  #取所有行的第1个数据

        #Graph
        global_states = self.gcn(att_feats_0, att_feats_1)   #(16,31,2048)

        # the features are expanded into a sequence by concatenating them from each row of the patches on the image.
        batch_size = att_feats_0.size(0)
        patch_feats_0 = att_feats_0.reshape(batch_size, 2048, -1).permute(0, 2, 1)   #(16,49,2048)
        patch_feats_1 = att_feats_1.reshape(batch_size, 2048, -1).permute(0, 2, 1)
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        # att_feats：([batch_size, 98, 2048])
        att_feats = torch.cat((patch_feats_0, patch_feats_1), dim=1)


        if mode == 'train':
            output = self.encoder_decoder(global_states, att_feats, targets, mode='forward')  #--->_forward
            #output:[batch_size,不固定/序列长度，761】
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(global_states, fc_feats, att_feats, mode='sample', update_opts=update_opts)  #att_model:_sample-->_prepare_feature
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError
