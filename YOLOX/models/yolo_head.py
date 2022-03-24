# Decoupled head of YOLOx
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 width: float = 1.0,
                 strides: List[int] = (8, 16, 32),
                 in_channels: List[int] = (256, 512, 1024),
                 act: str = "silu",
                 depthwise: bool = False):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        assert len(strides) == len(in_channels), "Length of strides and in_channels should be same!"
        self.num_classes = num_classes
        self.decode_in_inference = True

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act
                        ),
                        Conv(in_channels=int(256 * width),
                             out_channels=int(256 * width),
                             ksize=3,
                             stride=1,
                             act=act)
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act
                        ),
                        Conv(in_channels=int(256 * width),
                             out_channels=int(256 * width),
                             ksize=3,
                             stride=1,
                             act=act)
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, feats, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, feat) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, feats)
        ):
            x = self.stems[k](feat)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, feats[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(feats[0])
                )
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
            outputs.append(output)

        if self.training:
            pass
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, n_classes + 5]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=feats[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        """
        output.shape: (batch_size, height, width, n_classes + 5)

        Make output flatten. Then make output[:4]'s value to BoundingBox using 'grids' and 'strides'
        """
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid
        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        """
        outputs.shape: (batch_size, total_detections(Sigma of (w x h) for all feature maps), n_classes + 5)

        Make outputs[:4]'s value to BoundingBox using 'grids' and 'strides'
        """
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs



if __name__ == "__main__":
    # (736, 1280)
    input_sample = [torch.randn((8, 256, 736 // 8, 1280 // 8)),
                    torch.randn((8, 512, 736 // 16, 1280 // 16)),
                    torch.randn((8, 1024, 736 // 32, 1280 // 32))]
    model = YOLOXHead(num_classes=5)
    #model = model.train()
    model = model.eval()
    outputss = model(input_sample)
