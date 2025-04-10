# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        if len(x.shape) == 3:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        else:
            mask_keep = m
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)

import torch

def apply_masks_inverse(x, masks, concat=True):
    """
    Removes elements from x specified in masks.

    :param x: Tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: List of tensors of shape [B, K] containing indices of K patches in [N] to remove
    :param concat: If True, concatenates the output tensors along the batch dimension
    :return: Tensor with specified elements removed
    """
    all_x = []
    B, N = x.shape[:2]
    device = x.device

    idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)  # Shape: [B, N]

    for m in masks:
        # Create a boolean mask where True indicates positions to remove
        remove_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        remove_mask.scatter_(dim=1, index=m, value=True)

        # Indices to keep are where remove_mask is False
        indices_to_keep = idx[~remove_mask].view(B, -1)

        if x.dim() == 3:
            indices_to_keep_expanded = indices_to_keep.unsqueeze(-1).expand(-1, -1, x.size(-1))
        else:
            indices_to_keep_expanded = indices_to_keep

        # Gather elements not in the mask
        x_kept = torch.gather(x, dim=1, index=indices_to_keep_expanded)
        all_x.append(x_kept)

    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)


def graph_custom_collate_fn(batch):
    num_clips = len(batch[0][0])
    
    video = [torch.stack([b[0][clipidx] for b in batch]) for clipidx in range(num_clips)]
    masks = torch.stack([b[1] for b in batch])
    graphs_list = [b[2] for b in batch]
    labels = torch.tensor([b[3] for b in batch])
    indices = [torch.stack([torch.tensor(b[4][clipidx]) for b in batch]) for clipidx in range(num_clips)]
    
    # Find the maximum M (first dimension) in the batch
    max_M = max([tensor.shape[0] for tensor in graphs_list]) 
    T, batch_size, dtype = graphs_list[0].shape[1], len(graphs_list), graphs_list[0].dtype
    padded_graphs = torch.zeros(batch_size, max_M, T, dtype=dtype)
    # Copy tensors into the padded_batch
    for i, tensor in enumerate(graphs_list):
        padded_graphs[i, :len(tensor), :] = tensor  # Copy the tensor without for-loop padding
        
    return video, masks, padded_graphs, labels, indices

