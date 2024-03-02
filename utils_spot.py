''' Based on SLATE and OCLF libraries:
https://github.com/singhgautam/slate/blob/master/utils.py
https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/utils/masking.py
'''
import math
import random
import warnings
import argparse
import numpy as np
from typing import Optional
from PIL import ImageFilter
from collections import OrderedDict
from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import generate_binary_structure

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

binary_structure = generate_binary_structure(2,2)

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080','#C56932',
'#b7a58c', '#3a627d', '#9abc15', '#54810c', '#a7389c', '#687253', '#61f584', '#9a17d4', '#52b0c1', '#21f5b4', '#a2856c', '#9b1c34', '#4b1062', '#7cf406', '#0b1f63']

def gumbel_max(logits, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = logits + gumbels

    return gumbels.argmax(dim)


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau

    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def log_prob_gaussian(value, mean, std):

    var = std ** 2
    if isinstance(var, float):
        return -0.5 * (((value - mean) ** 2) / var + math.log(var) + math.log(2 * math.pi))
    else:
        return -0.5 * (((value - mean) ** 2) / var + var.log() + math.log(2 * math.pi))


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):

    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=False, weight_init='kaiming')
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))


    def forward(self, x):

        x = self.m(x)
        return F.relu(F.group_norm(x, 1, self.weight, self.bias))


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):

    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gru_cell(input_size, hidden_size, bias=True):

    m = nn.GRUCell(input_size, hidden_size, bias)

    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)

    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

    return m

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

inv_normalize = transforms.Compose([transforms.Normalize((0., 0., 0.), (1/0.229, 1/0.224, 1/0.225)),
                                    transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1))])


def pairwise_IoU(pred_mask, gt_mask):
    pred_mask = repeat(pred_mask, "... n c -> ... 1 n c")
    gt_mask = repeat(gt_mask, "... n c -> ... n 1 c")
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    union_sum = torch.clamp(union.sum(-1).float(), min=0.000001) # to avoid division by zero.
    iou = intersection.sum(-1) / union_sum
    return iou


def pairwise_IoU_efficient(pred_mask, gt_mask):
    intersection = torch.einsum("bkj,bij->bki", gt_mask, pred_mask)
    union = gt_mask.sum(dim=2).unsqueeze(dim=2) + pred_mask.sum(dim=2).unsqueeze(dim=1) - intersection
    iou = intersection / torch.clamp(union, min=0.000001) # to avoid division by zero.
    return iou


def compute_IoU(pred_mask, gt_mask):
    # assumes shape: batch_size, set_size, channels
    is_padding = (gt_mask == 0).all(-1)

    # discretized 2d mask for hungarian matching
    pred_mask_id = torch.argmax(pred_mask, -2)
    num_slots = pred_mask.size(1)
    if num_slots < gt_mask.size(1): # essentially padding the pred_mask if num_slots < gt_mask.size(1)
        num_slots = gt_mask.size(1)
    pred_mask_disc = rearrange(
        F.one_hot(pred_mask_id, num_slots).to(torch.float32), "b c n -> b n c"
    )

    # treat as if no padding in gt_mask
    pIoU = pairwise_IoU_efficient(pred_mask_disc.float(), gt_mask.float())
    #pIoU = pairwise_IoU(pred_mask_disc.bool(), gt_mask.bool())
    pIoU_inv = 1 - pIoU
    pIoU_inv[is_padding] = 1e3
    pIoU_inv_ = pIoU_inv.detach().cpu().numpy()

    # hungarian matching
    indices = np.array([linear_sum_assignment(p) for p in pIoU_inv_])
    indices_ = pred_mask.size(1) * indices[:, 0] + indices[:, 1]
    indices_ = torch.from_numpy(indices_).to(device=pred_mask.device)
    IoU = torch.gather(rearrange(pIoU, "b n m -> b (n m)"), 1, indices_)
    mIoU = (IoU * ~is_padding).sum(-1) / torch.clamp((~is_padding).sum(-1), min=0.000001) # to avoid division by zero.
    return mIoU

def att_matching(attention_1, attention_2):

    batch_size, slots, height, width = attention_1.shape
    
    mask_1 = torch.nn.functional.one_hot(attention_1.argmax(1).reshape(batch_size,-1), num_classes=slots).to(torch.float32).permute(0,2,1)
    mask_2 = torch.nn.functional.one_hot(attention_2.argmax(1).reshape(batch_size,-1), num_classes=slots).to(torch.float32).permute(0,2,1)
    
    # assumes shape: batch_size, set_size, channels
    is_padding = (mask_2 == 0).all(-1)

    # discretized 2d mask for hungarian matching
    pred_mask_1_id = torch.argmax(mask_1, -2)
    pred_mask_1_disc = rearrange(
        F.one_hot(pred_mask_1_id, mask_1.size(1)).to(torch.float32), "b c n -> b n c"
    )

    # treat as if no padding in mask_2
    pIoU = pairwise_IoU_efficient(pred_mask_1_disc.float(), mask_2.float())
    pIoU_inv = 1 - pIoU
    pIoU_inv[is_padding] = 1e3
    pIoU_inv_ = pIoU_inv.detach().cpu().numpy()

    # hungarian matching
    indices = np.array([linear_sum_assignment(p)[1] for p in pIoU_inv_])
    #attention_2_permuted = torch.stack([x[indices[n]] for n, x in enumerate(attention_2)],dim=0)

    pIoU = pIoU.detach().cpu().numpy()
    matching_scores = np.array([[pIoU[b][i,j] for i,j in enumerate(indices[b])] for b in range(batch_size)])
    return indices, matching_scores

def trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
        
#Copied from https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/utils/masking.py
"""Utilities related to masking."""

class CreateSlotMask(nn.Module):
    """Module intended to create a mask that marks empty slots.
    Module takes a tensor holding the number of slots per batch entry, and returns a binary mask of
    shape (batch_size, max_slots) where entries exceeding the number of slots are masked out.
    """

    def __init__(self, max_slots: int):
        super().__init__()
        self.max_slots = max_slots

    def forward(self, n_slots: torch.Tensor) -> torch.Tensor:
        (batch_size,) = n_slots.shape

        # Create mask of shape B x K where the first n_slots entries per-row are false, the rest true
        indices = torch.arange(self.max_slots, device=n_slots.device)
        masks = indices.unsqueeze(0).expand(batch_size, -1) >= n_slots.unsqueeze(1)

        return masks

#Copied from https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/utils/masking.py
class CreateRandomMaskPatterns(nn.Module):
    """Create random masks.
    Useful for showcasing behavior of metrics.
    """

    def __init__(self, pattern: str, n_slots: Optional[int] = None, n_cols: int = 2):
        super().__init__()
        if pattern not in ("random", "blocks"):
            raise ValueError(f"Unknown pattern {pattern}")
        self.pattern = pattern
        self.n_slots = n_slots
        self.n_cols = n_cols

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if self.pattern == "random":
            rand_mask = torch.rand_like(masks)
            return rand_mask / rand_mask.sum(1, keepdim=True)
        elif self.pattern == "blocks":
            n_slots = masks.shape[1] if self.n_slots is None else self.n_slots
            height, width = masks.shape[-2:]
            new_masks = torch.zeros(
                len(masks), n_slots, height, width, device=masks.device, dtype=masks.dtype
            )
            blocks_per_col = int(n_slots // self.n_cols)
            remainder = n_slots - (blocks_per_col * self.n_cols)
            slot = 0
            for col in range(self.n_cols):
                rows = blocks_per_col if col < self.n_cols - 1 else blocks_per_col + remainder
                for row in range(rows):
                    block_width = math.ceil(width / self.n_cols)
                    block_height = math.ceil(height / rows)
                    x = col * block_width
                    y = row * block_height
                    new_masks[:, slot, y : y + block_height, x : x + block_width] = 1
                    slot += 1
            assert torch.allclose(new_masks.sum(1), torch.ones_like(masks[:, 0]))
            return new_masks
        
def spiral_pattern(A, how = 'left_top'):
    
    out = []
    
    if how == 'left_top':
        A = np.array(A)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'top_left':
        A = np.rot90(np.fliplr(np.array(A)), k=1)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'right_top':
        A = np.fliplr(np.array(A))
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'top_right':
        A = np.rot90(np.array(A), k=1)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'right_bottom':
        A = np.rot90(np.array(A), k=2)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'bottom_right':
        A = np.fliplr(np.rot90(np.array(A), k=1))
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'left_bottom':
        A = np.rot90(np.fliplr(np.array(A)), k=2)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'bottom_left':
        A = np.rot90(np.array(A), k=3)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)


def visualize(image, true_mask, pred_dec_mask, rgb_dec_attns, pred_default_mask, rgb_default_attns, N=8):
    _, _, H, W = image.shape
    
    rgb_pred_dec_mask = (torch.stack([draw_segmentation_masks((image[idx]*255).to(torch.uint8).cpu(), 
                                        masks= torch.nn.functional.one_hot(pred_dec_mask[idx]).permute(2,0,1).to(torch.bool).cpu(), 
                                        alpha=.5,
                                        colors = colors) for idx in range(image.shape[0])])/255.)
    
    rgb_pred_default_mask = (torch.stack([draw_segmentation_masks((image[idx]*255).to(torch.uint8).cpu(), 
                                    masks= torch.nn.functional.one_hot(pred_default_mask[idx]).permute(2,0,1).to(torch.bool).cpu(), 
                                    alpha=.5,
                                    colors = colors) for idx in range(image.shape[0])])/255.)

    _, true_mask_unique = torch.unique(true_mask,return_inverse=True)
    rgb_true_mask = (torch.stack([draw_segmentation_masks((image[idx]*255).to(torch.uint8).cpu(), 
                                masks= torch.nn.functional.one_hot(true_mask_unique[idx]).permute(2,0,1).to(torch.bool).cpu(), 
                                alpha=.5,
                                colors = colors) for idx in range(image.shape[0])])/255.)
    
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1).cpu()
    rgb_default_attns = rgb_default_attns[:N].expand(-1, -1, 3, H, W).cpu()
    rgb_dec_attns = rgb_dec_attns[:N].expand(-1, -1, 3, H, W).cpu()

    rgb_true_mask = rgb_true_mask.unsqueeze(dim=1).cpu()
    rgb_pred_default_mask = rgb_pred_default_mask.unsqueeze(dim=1).cpu()
    rgb_pred_dec_mask = rgb_pred_dec_mask.unsqueeze(dim=1).cpu()

    return torch.cat((image, rgb_true_mask, rgb_pred_dec_mask, rgb_dec_attns, rgb_pred_default_mask, rgb_default_attns), dim=1).view(-1, 3, H, W)


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def load_pretrained_encoder(model, pretrained_weights, prefix=None):
    if pretrained_weights:
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrained_weights)
        if "state_dict" in checkpoint:
            checkpoint_model = checkpoint["state_dict"]
        elif 'target_encoder' in checkpoint:
            checkpoint_model = checkpoint["target_encoder"]
        elif 'model' in checkpoint:
            checkpoint_model = checkpoint["model"]

        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}

        if prefix is not None:
            checkpoint = checkpoint_model
            checkpoint_model = OrderedDict()
            # Keep only the parameters/buffers of the ViT encoder.
            all_keys = list(checkpoint.keys())
            counter = 0
            for key in all_keys:
                if key.startswith(prefix):
                    counter += 1
                    new_key = key[len(prefix):]
                    print(f"\t #{counter}: {key} ==> {new_key}")
                    checkpoint_model[new_key] = checkpoint[key]

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        print("Model = %s" % str(model))
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        assert len(set(msg.missing_keys)) == 0
