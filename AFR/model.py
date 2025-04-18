import torch
from torch import nn
from torch.nn import functional as F
from backbones import ConvTransformerBackbone, FPNIdentity, ClsHead
import math
import random
import sys
import importlib
from utils import setup_dataset
# from torchinfo import summary

featuredim = {
    "rgb": 1024,
    "flow": 1024,
    "both": 2048,
}


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class FPNIdentityWithCBAM(FPNIdentity):
    def __init__(self, *args, **kwargs):
        super(FPNIdentityWithCBAM, self).__init__(*args, **kwargs)
        self.cbam = CBAM(kwargs['out_channel'])

    def forward(self, feats, masks):
        feats, masks = super().forward(feats, masks)
        feats = [self.cbam(f.unsqueeze(2)).squeeze(2) for f in feats]
        return feats, masks


class VADTransformer(nn.Module):
    """
    Transformer based model for video anomaly detection
    """

    def __init__(self, args):
        super(VADTransformer, self).__init__()
        self.bs = args.batch_size
        self.n_layers = args.arch[2]
        self.mha_win_size = [args.n_mha_win_size] * self.n_layers
        self.max_seq_len = args.max_seq_len
        self.backbone = ConvTransformerBackbone(
            n_in=featuredim[args.modality],
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_embd_ks=3,
            max_len=args.max_seq_len,
            arch=(args.arch[0], args.arch[1], args.arch[2] - 1),
            mha_win_size=self.mha_win_size,
            scale_factor=args.scale_factor,
            with_ln=True,
        )

        self.neck = FPNIdentityWithCBAM(
            in_channels=[args.n_embd] * self.n_layers,
            out_channel=args.n_embd,
            scale_factor=args.scale_factor,
            start_level=0,
            with_ln=True,
            drop_rate=args.dropout,
            se_enhance=args.se_ratio,
        )

        self.cls_head = ClsHead(
            input_dim=args.n_embd,
            feat_dim=args.n_embd,
            num_classes=1,
            kernel_size=3,
            with_ln=True,
        )

    def forward(self, inputs, is_training=False):
        feats = inputs["feats"].permute(0, 2, 1)
        masks = inputs["masks"].unsqueeze(1)
        feats, masks = self.backbone(feats, masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        logits = self.cls_head(fpn_feats, fpn_masks)

        logits = [x.squeeze() for x in logits]
        fpn_masks = [x.squeeze() for x in fpn_masks]
        scores = [x.sigmoid() * y for x, y in zip(logits, fpn_masks)]

        if is_training:
            contrast_pairs = None
            pseudo_label = inputs["pseudo_label"]
            fpn_feats = [f.permute(0, 2, 1) for f in fpn_feats]

            ABN_EMB_PRED, ABN_EMB_PSE, N_EMB = tuple(), tuple(), tuple()
            for i, mask in enumerate(fpn_masks):
                k_normal = math.ceil(mask[0: self.bs].sum(-1).float().mean().item() * random.uniform(0.2, 0.4))
                N_EMB += (self.select_topk_embeddings(scores[i][0: self.bs], fpn_feats[i][0: self.bs], k_normal),)
                k_abnormal = math.ceil(mask[self.bs:].sum(-1).float().mean().item() * random.uniform(0.1, 0.3))
                ABN_EMB_PRED += (self.select_topk_embeddings(scores[i][self.bs:], fpn_feats[i][self.bs:], k_abnormal),)
                pse_label_i = torch.max_pool1d(pseudo_label[self.bs:], kernel_size=2 ** i).float()
                ABN_EMB_PSE += (self.select_topk_embeddings(pse_label_i, fpn_feats[i][self.bs:], k_abnormal),)
            contrast_pairs = {"ABN_EMB_PRED": ABN_EMB_PRED, "ABN_EMB_PSE": ABN_EMB_PSE, "N_EMB": N_EMB}
            return scores, logits, fpn_masks, contrast_pairs
        return scores

    def select_topk_embeddings(self, scores, embeddings, k):
        if k == 0:
            return torch.zeros_like(embeddings[:, :1, :])  # 处理 k=0 的情况，防止索引错误
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings


if __name__ == "__main__":
    config_module = importlib.import_module("config.ucfcrime_cfg")
    parser = config_module.parse_args()
    args = parser.parse_args()
    model = VADTransformer(args)
    print(model)
    # summary(model, input_data=(torch.randn(1, 384, 1024), torch.ones(1, 384)))



#  只有通道注意力
# import torch
# from torch import nn
# from torch.nn import functional as F
# from backbones import ConvTransformerBackbone, FPNIdentity, ClsHead
# import math
# import random
# import sys
# import importlib
# from utils import setup_dataset
# # from torchinfo import summary
#
# featuredim = {
#     "rgb": 1024,
#     "flow": 1024,
#     "both": 2048,
# }
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(in_planes, ratio)
#
#     def forward(self, x):
#         out = x * self.ca(x)
#         return out
#
#
# class FPNIdentityWithCBAM(FPNIdentity):
#     def __init__(self, *args, **kwargs):
#         super(FPNIdentityWithCBAM, self).__init__(*args, **kwargs)
#         self.cbam = CBAM(kwargs['out_channel'])
#
#     def forward(self, feats, masks):
#         feats, masks = super().forward(feats, masks)
#         feats = [self.cbam(f.unsqueeze(2)).squeeze(2) for f in feats]
#         return feats, masks
#
#
# class VADTransformer(nn.Module):
#     """
#     Transformer based model for video anomaly detection
#     """
#
#     def __init__(self, args):
#         super(VADTransformer, self).__init__()
#         self.bs = args.batch_size
#         self.n_layers = args.arch[2]
#         self.mha_win_size = [args.n_mha_win_size] * self.n_layers
#         self.max_seq_len = args.max_seq_len
#         self.backbone = ConvTransformerBackbone(
#             n_in=featuredim[args.modality],
#             n_embd=args.n_embd,
#             n_head=args.n_head,
#             n_embd_ks=3,
#             max_len=args.max_seq_len,
#             arch=(args.arch[0], args.arch[1], args.arch[2] - 1),
#             mha_win_size=self.mha_win_size,
#             scale_factor=args.scale_factor,
#             with_ln=True,
#         )
#
#         self.neck = FPNIdentityWithCBAM(
#             in_channels=[args.n_embd] * self.n_layers,
#             out_channel=args.n_embd,
#             scale_factor=args.scale_factor,
#             start_level=0,
#             with_ln=True,
#             drop_rate=args.dropout,
#             se_enhance=args.se_ratio,
#         )
#
#         self.cls_head = ClsHead(
#             input_dim=args.n_embd,
#             feat_dim=args.n_embd,
#             num_classes=1,
#             kernel_size=3,
#             with_ln=True,
#         )
#
#     def forward(self, inputs, is_training=False):
#         feats = inputs["feats"].permute(0, 2, 1)
#         masks = inputs["masks"].unsqueeze(1)
#         feats, masks = self.backbone(feats, masks)
#         fpn_feats, fpn_masks = self.neck(feats, masks)
#         logits = self.cls_head(fpn_feats, fpn_masks)
#
#         logits = [x.squeeze() for x in logits]
#         fpn_masks = [x.squeeze() for x in fpn_masks]
#         scores = [x.sigmoid() * y for x, y in zip(logits, fpn_masks)]
#
#         if is_training:
#             contrast_pairs = None
#             pseudo_label = inputs["pseudo_label"]
#             fpn_feats = [f.permute(0, 2, 1) for f in fpn_feats]
#
#             ABN_EMB_PRED, ABN_EMB_PSE, N_EMB = tuple(), tuple(), tuple()
#             for i, mask in enumerate(fpn_masks):
#                 k_normal = math.ceil(mask[0: self.bs].sum(-1).float().mean().item() * random.uniform(0.2, 0.4))
#                 N_EMB += (self.select_topk_embeddings(scores[i][0: self.bs], fpn_feats[i][0: self.bs], k_normal),)
#                 k_abnormal = math.ceil(mask[self.bs:].sum(-1).float().mean().item() * random.uniform(0.1, 0.3))
#                 ABN_EMB_PRED += (self.select_topk_embeddings(scores[i][self.bs:], fpn_feats[i][self.bs:], k_abnormal),)
#                 pse_label_i = torch.max_pool1d(pseudo_label[self.bs:], kernel_size=2 ** i).float()
#                 ABN_EMB_PSE += (self.select_topk_embeddings(pse_label_i, fpn_feats[i][self.bs:], k_abnormal),)
#             contrast_pairs = {"ABN_EMB_PRED": ABN_EMB_PRED, "ABN_EMB_PSE": ABN_EMB_PSE, "N_EMB": N_EMB}
#             return scores, logits, fpn_masks, contrast_pairs
#         return scores
#
#     def select_topk_embeddings(self, scores, embeddings, k):
#         if k == 0:
#             return torch.zeros_like(embeddings[:, :1, :])  # 处理 k=0 的情况，防止索引错误
#         _, idx_DESC = scores.sort(descending=True, dim=1)
#         idx_topk = idx_DESC[:, :k]
#         idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
#         selected_embeddings = torch.gather(embeddings, 1, idx_topk)
#         return selected_embeddings
#
#
# if __name__ == "__main__":
#     config_module = importlib.import_module("config.ucfcrime_cfg")
#     parser = config_module.parse_args()
#     args = parser.parse_args()
#     model = VADTransformer(args)
#     print(model)
#     # summary(model, input_data=(torch.randn(1, 384, 1024), torch.ones(1, 384)))


# 只有空间注意力
# import torch
# from torch import nn
# from torch.nn import functional as F
# from backbones import ConvTransformerBackbone, FPNIdentity, ClsHead
# import math
# import random
# import sys
# import importlib
# from utils import setup_dataset
# # from torchinfo import summary
#
# featuredim = {
#     "rgb": 1024,
#     "flow": 1024,
#     "both": 2048,
# }
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# class CBAM(nn.Module):
#     def __init__(self, in_planes, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.sa = SpatialAttention(kernel_size)
#
#     def forward(self, x):
#         result = x * self.sa(x)
#         return result
#
#
# class FPNIdentityWithCBAM(FPNIdentity):
#     def __init__(self, *args, **kwargs):
#         super(FPNIdentityWithCBAM, self).__init__(*args, **kwargs)
#         self.cbam = CBAM(kwargs['out_channel'])
#
#     def forward(self, feats, masks):
#         feats, masks = super().forward(feats, masks)
#         feats = [self.cbam(f.unsqueeze(2)).squeeze(2) for f in feats]
#         return feats, masks
#
#
# class VADTransformer(nn.Module):
#     """
#     Transformer based model for video anomaly detection
#     """
#
#     def __init__(self, args):
#         super(VADTransformer, self).__init__()
#         self.bs = args.batch_size
#         self.n_layers = args.arch[2]
#         self.mha_win_size = [args.n_mha_win_size] * self.n_layers
#         self.max_seq_len = args.max_seq_len
#         self.backbone = ConvTransformerBackbone(
#             n_in=featuredim[args.modality],
#             n_embd=args.n_embd,
#             n_head=args.n_head,
#             n_embd_ks=3,
#             max_len=args.max_seq_len,
#             arch=(args.arch[0], args.arch[1], args.arch[2] - 1),
#             mha_win_size=self.mha_win_size,
#             scale_factor=args.scale_factor,
#             with_ln=True,
#         )
#
#         self.neck = FPNIdentityWithCBAM(
#             in_channels=[args.n_embd] * self.n_layers,
#             out_channel=args.n_embd,
#             scale_factor=args.scale_factor,
#             start_level=0,
#             with_ln=True,
#             drop_rate=args.dropout,
#             se_enhance=args.se_ratio,
#         )
#
#         self.cls_head = ClsHead(
#             input_dim=args.n_embd,
#             feat_dim=args.n_embd,
#             num_classes=1,
#             kernel_size=3,
#             with_ln=True,
#         )
#
#     def forward(self, inputs, is_training=False):
#         feats = inputs["feats"].permute(0, 2, 1)
#         masks = inputs["masks"].unsqueeze(1)
#         feats, masks = self.backbone(feats, masks)
#         fpn_feats, fpn_masks = self.neck(feats, masks)
#         logits = self.cls_head(fpn_feats, fpn_masks)
#
#         logits = [x.squeeze() for x in logits]
#         fpn_masks = [x.squeeze() for x in fpn_masks]
#         scores = [x.sigmoid() * y for x, y in zip(logits, fpn_masks)]
#
#         if is_training:
#             contrast_pairs = None
#             pseudo_label = inputs["pseudo_label"]
#             fpn_feats = [f.permute(0, 2, 1) for f in fpn_feats]
#
#             ABN_EMB_PRED, ABN_EMB_PSE, N_EMB = tuple(), tuple(), tuple()
#             for i, mask in enumerate(fpn_masks):
#                 k_normal = math.ceil(mask[0: self.bs].sum(-1).float().mean().item() * random.uniform(0.2, 0.4))
#                 N_EMB += (self.select_topk_embeddings(scores[i][0: self.bs], fpn_feats[i][0: self.bs], k_normal),)
#                 k_abnormal = math.ceil(mask[self.bs:].sum(-1).float().mean().item() * random.uniform(0.1, 0.3))
#                 ABN_EMB_PRED += (self.select_topk_embeddings(scores[i][self.bs:], fpn_feats[i][self.bs:], k_abnormal),)
#                 pse_label_i = torch.max_pool1d(pseudo_label[self.bs:], kernel_size=2 ** i).float()
#                 ABN_EMB_PSE += (self.select_topk_embeddings(pse_label_i, fpn_feats[i][self.bs:], k_abnormal),)
#             contrast_pairs = {"ABN_EMB_PRED": ABN_EMB_PRED, "ABN_EMB_PSE": ABN_EMB_PSE, "N_EMB": N_EMB}
#             return scores, logits, fpn_masks, contrast_pairs
#         return scores
#
#     def select_topk_embeddings(self, scores, embeddings, k):
#         if k == 0:
#             return torch.zeros_like(embeddings[:, :1, :])  # 处理 k=0 的情况，防止索引错误
#         _, idx_DESC = scores.sort(descending=True, dim=1)
#         idx_topk = idx_DESC[:, :k]
#         idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
#         selected_embeddings = torch.gather(embeddings, 1, idx_topk)
#         return selected_embeddings
#
#
# if __name__ == "__main__":
#     config_module = importlib.import_module("config.ucfcrime_cfg")
#     parser = config_module.parse_args()
#     args = parser.parse_args()
#     model = VADTransformer(args)
#     print(model)
#     # summary(model, input_data=(torch.randn(1, 384, 1024), torch.ones(1, 384)))



# import torch
# from torch import nn
# from torch.nn import functional as F
# from backbones import ConvTransformerBackbone, FPNIdentity, ClsHead
# import math
# import random
# import sys
# import importlib
# from utils import setup_dataset
# # from torchinfo import summary
#
# featuredim = {
#     "rgb": 1024,
#     "flow": 1024,
#     "both": 2048,
# }
#
# class NonLocalBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(NonLocalBlock, self).__init__()
#         self.in_channels = in_channels
#         self.inter_channels = in_channels // 2  # 统一所有输出通道
#
#         self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
#         self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
#         self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
#         self.W = nn.Conv2d(self.inter_channels, in_channels, 1)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         batch_size, C, H, W = x.size()
#
#         g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # [B, H*W, C']
#         theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # [B, H*W, C']
#         phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C', H*W]
#
#         # print(f"g_x shape: {g_x.shape}")
#         # print(f"theta_x shape: {theta_x.shape}")
#         # print(f"phi_x shape: {phi_x.shape}")
#
#         f = torch.matmul(theta_x, phi_x)  # [B, H*W, H*W]
#         f = self.softmax(f)
#
#         y = torch.matmul(f, g_x)  # [B, H*W, C']
#         # print(f"y shape: {y.shape}")
#
#         y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, H, W)
#         W_y = self.W(y)  # [B, C, H, W]
#
#         return W_y + x
#
#
#
#
# class FPNIdentityWithNonLocal(FPNIdentity):
#     def __init__(self, *args, **kwargs):
#         super(FPNIdentityWithNonLocal, self).__init__(*args, **kwargs)
#         self.non_local = NonLocalBlock(kwargs['out_channel'])  # 加入Non-local Attention
#
#     def forward(self, feats, masks):
#         feats, masks = super().forward(feats, masks)
#         feats = [self.non_local(f.unsqueeze(2)).squeeze(2) for f in feats]  # 应用Non-local Attention
#         return feats, masks
#
#
# class VADTransformer(nn.Module):
#     """
#     Transformer based model for video anomaly detection
#     """
#
#     def __init__(self, args):
#         super(VADTransformer, self).__init__()
#         self.bs = args.batch_size
#         self.n_layers = args.arch[2]
#         self.mha_win_size = [args.n_mha_win_size] * self.n_layers
#         self.max_seq_len = args.max_seq_len
#         self.backbone = ConvTransformerBackbone(
#             n_in=featuredim[args.modality],
#             n_embd=args.n_embd,
#             n_head=args.n_head,
#             n_embd_ks=3,
#             max_len=args.max_seq_len,
#             arch=(args.arch[0], args.arch[1], args.arch[2] - 1),
#             mha_win_size=self.mha_win_size,
#             scale_factor=args.scale_factor,
#             with_ln=True,
#         )
#
#         # 使用FPNIdentityWithNonLocal替代FPNIdentity
#         self.neck = FPNIdentityWithNonLocal(
#             in_channels=[args.n_embd] * self.n_layers,
#             out_channel=args.n_embd,
#             scale_factor=args.scale_factor,
#             start_level=0,
#             with_ln=True,
#             drop_rate=args.dropout,
#             se_enhance=args.se_ratio,
#         )
#
#         self.cls_head = ClsHead(
#             input_dim=args.n_embd,
#             feat_dim=args.n_embd,
#             num_classes=1,
#             kernel_size=3,
#             with_ln=True,
#         )
#
#     def forward(self, inputs, is_training=False):
#         # forward the network (backbone -> neck -> heads)
#         feats = inputs["feats"].permute(0, 2, 1)  # (B, C, T)
#         masks = inputs["masks"].unsqueeze(1)  # (B, 1, T)
#         feats, masks = self.backbone(feats, masks)
#         fpn_feats, fpn_masks = self.neck(feats, masks)
#         logits = self.cls_head(fpn_feats, fpn_masks)  # (B, cls, T)
#
#         # output
#         logits = [x.squeeze() for x in logits]  # (B, 1, T) -> (B, T)
#         fpn_masks = [x.squeeze() for x in fpn_masks]  # (B, 1, T) -> (B, T)
#         scores = [x.sigmoid() * y for x, y in zip(logits, fpn_masks)]  # (B, T)
#
#         if is_training:
#             contrast_pairs = None
#             pseudo_label = inputs["pseudo_label"]
#             fpn_feats = [f.permute(0, 2, 1) for f in fpn_feats]  # (B, T, C)
#
#             ABN_EMB_PRED, ABN_EMB_PSE, N_EMB = tuple(), tuple(), tuple()
#             for i, mask in enumerate(fpn_masks):
#                 # select representative normal feature
#                 k_normal = math.ceil(mask[0 : self.bs].sum(-1).float().mean().item() * random.uniform(0.2, 0.4))
#                 N_EMB += (self.select_topk_embeddings(scores[i][0 : self.bs], fpn_feats[i][0 : self.bs], k_normal),)
#
#                 # select top 10% representative abnormal feature from predict
#                 k_abnormal = math.ceil(mask[self.bs :].sum(-1).float().mean().item() * random.uniform(0.1, 0.3))
#                 ABN_EMB_PRED += (self.select_topk_embeddings(scores[i][self.bs :], fpn_feats[i][self.bs :], k_abnormal),)
#                 # select top 10% representative abnormal feature from pseudo label
#                 pse_label_i = torch.max_pool1d(pseudo_label[self.bs :], kernel_size=2**i).float()
#                 ABN_EMB_PSE += (self.select_topk_embeddings(pse_label_i, fpn_feats[i][self.bs :], k_abnormal),)
#
#             contrast_pairs = {"ABN_EMB_PRED": ABN_EMB_PRED, "ABN_EMB_PSE": ABN_EMB_PSE, "N_EMB": N_EMB}
#             return scores, logits, fpn_masks, contrast_pairs
#         return scores
#
#     def select_topk_embeddings(self, scores, embeddings, k):
#         _, idx_DESC = scores.sort(descending=True, dim=1)
#         idx_topk = idx_DESC[:, :k]
#         idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
#         selected_embeddings = torch.gather(embeddings, 1, idx_topk)
#         return selected_embeddings
#
# if __name__ == "__main__":
#     config_module = importlib.import_module("config.ucfcrime_cfg")
#     parser = config_module.parse_args()
#     args = parser.parse_args()
#     model = VADTransformer(args)
#     print(model)
#     # summary(model, input_data=(torch.randn(1, 384, 1024), torch.ones(1, 384)))
