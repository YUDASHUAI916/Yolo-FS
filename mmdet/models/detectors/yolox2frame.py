# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info
import torch.nn as nn
from ...utils import log_img_scale
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import HEADS, build_loss


@DETECTORS.register_module()
class YOLOX2Frame(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Note: Considering the trade-off between training speed and accuracy,
    multi-scale training is temporarily kept. More elegant implementation
    will be adopted in the future.

    Args:
        backbone (nn.Module): The backbone module.
        neck (nn.Module): The neck module.
        bbox_head (nn.Module): The bbox head module.
        train_cfg (obj:`ConfigDict`, optional): The training config
            of YOLOX. Default: None.
        test_cfg (obj:`ConfigDict`, optional): The testing config
            of YOLOX. Default: None.
        pretrained (str, optional): model pretrained path.
            Default: None.
        input_size (tuple): The model default input image size. The shape
            order should be (height, width). Default: (640, 640).
        size_multiplier (int): Image size multiplication factor.
            Default: 32.
        random_size_range (tuple): The multi-scale random range during
            multi-scale training. The real training image size will
            be multiplied by size_multiplier. Default: (15, 25).
        random_size_interval (int): The iter interval of change
            image size. Default: 10.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,

                 loss_motion_aware_module = dict(                               # ! add new loss func
                     type='CrossEntropyLoss', 
                     loss_weight=1.0, use_sigmoid=False, use_mask=True,
                 ),
                 loss_feature_denoising_module = dict(
                     type='CrossEntropyLoss', 
                     loss_weight=1.0, use_sigmoid=False, use_mask=True,
                 ),
                 
                 init_cfg=None):
        super(YOLOX2Frame, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        log_img_scale(input_size, skip_square=True)
        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self._input_size = input_size
        self._random_size_range = random_size_range
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0

        #  ! add new module        2022/6/11 ShawnYu.
        self.Motion_Aware = True
        self.motion_aware_module = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(),

            nn.Conv2d(3, 1, 3, 1, 1, bias=True),
        )

        #  ! add new module        2022/6/11 ShawnYu.
        self.Feature_Denoising = True
        self.feature_denoising_module_1 = nn.Sequential(        # ! 80 -> 80
            nn.Conv2d(128, 64, 3, 1, 1),        
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1),
        )
        self.feature_denoising_module_2 = nn.Sequential(        # ! 40 -> 80
            torch.nn.UpsamplingBilinear2d((80, 80)),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1),
        )
        self.feature_denoising_module_3 = nn.Sequential(        # ! 20 -> 40 -> 80
            torch.nn.UpsamplingBilinear2d((40, 40)),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            torch.nn.UpsamplingBilinear2d((80, 80)),
            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1),
        )

        #  ! add new loss func     2022/6/11 ShawnYu.
        self.loss_motion_aware = build_loss(loss_motion_aware_module)
        #  ! add new loss func     2022/6/11 ShawnYu.
        self.loss_feature_denoising = build_loss(loss_feature_denoising_module)

    def forward_train(self,
                      img,
                      img_2,
                      img_2_gt,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Multi-scale training
        img, gt_bboxes = self._preprocess(img, gt_bboxes)

        losses = super(YOLOX2Frame, self).forward_trainYolox2Frame(img, img_2, img_2_gt, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize()
        self._progress_in_iter += 1

        return losses

    def _preprocess(self, img, gt_bboxes):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(
                img,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes

    def _random_resize(self):
        tensor = torch.LongTensor(2).cuda()

        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            aspect_ratio = float(
                self._default_input_size[1]) / self._default_input_size[0]
            size = (self._size_multiplier * size,
                    self._size_multiplier * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if self.world_size > 1:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    @auto_fp16(apply_to=('img', 'img_2'))
    def forward(self, img, img_2, img_metas, gt_bboxes=None, gt_labels=None, img_2_gt=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_trainYolox2Frame(img, img_2, img_2_gt, img_metas, gt_bboxes, gt_labels, **kwargs)
        else:
            return self.forward_testYolox(img, img_2, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
