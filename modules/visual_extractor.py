import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor   #resnet101
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]  #使用除了最后2层的前面所有层,去掉了AdaptiveAvgPool2d(output_size=(1, 1)), Linear(in_features=2048, out_features=1000, bias=True)]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images): #images:[batch_size, 3, 224, 224] / 32 --->
         #patch_feats:[batch_size, 2048, 7, 7]
        patch_feats = self.model(images)
         # avg_feats:[batch_size, 2048]---没用到
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))

        batch_size, feat_size, _, _ = patch_feats.shape    #8,2048
         #patch_feats:[batch_size, 49, 2048]
        # patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # ----》patch_feats改成了[batch_size, 2048, 7, 7]
        return patch_feats, avg_feats        #[batch_size, 49, 2048],[batch_size, 2048]  global visual representation with 2048 dimensions

# if __name__ == '__main__':
#     model = getattr(models,'resnet101')(pretrained=True)
#     modules = list(model.children())[:-2]
#     print(module s[:-2])
#     model = nn.Sequential(*modules)

    # images = torch.randn(16,3,512,512)
    # patch_feats = model(images)
    # print(patch_feats.shape)

    # avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    # avg_feats = avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
    # print(avg_feats.size())
    # batch_size, feat_size, _, _ = patch_feats.shape
    # patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
    # print(patch_feats.size(), avg_feats.size())

