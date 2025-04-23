# Copyright (c) Charlotte-CharMLab at University of North Carolina at Charlotte.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# https://github.com/Charlotte-CharMLab/Fibottention
# --------------------------------------------------------

import torch
import math
import random

def get_mask_attn_wythoff(B, H, N, is_modified, is_shuffled, depth_id, 
                                     add_class_token=True, device=torch.device('cpu'), 
                                     dtype=torch.float32):
    """
    Create a masked attention pattern based on Wythoff's sequence using only the tensor dimensions.
    
    Args:
        B (int): Batch size.
        H (int): Number of attention heads.
        N (int): Number of tokens (excluding the special [CLS] token, if applicable).
        is_modified (bool): Whether to use the modified version of the sequence.
        is_shuffled (bool): Whether to shuffle the indices across heads using depth_id as seed.
        depth_id (int): Seed value used for shuffling.
        add_class_token (bool): Whether to add a special token (e.g., [CLS]) at the beginning.
        device (torch.device): Device where tensors are allocated.
        dtype (torch.dtype): Data type for the tensors.
    
    Returns:
        torch.Tensor: The resulting attention mask of shape (B, H, N, N) or (B, H, N+1, N+1) if add_class_token is True.
    """
    headindices = generate_head_indices(N=N, h=H, wmin=5, is_modified=is_modified)
    mask = torch.zeros((B, H, N, N), device=device, dtype=dtype)
    
    # Optionally shuffle head indices
    if is_shuffled:
        headindices = shuffle(depth_id, headindices)
    
    # Build the mask using Wythoff-based diagonal patterns
    for h in range(H):
        fib_indices = headindices[h]
        for i in fib_indices:
            # Create diagonal masks (positive offsets)
            indices = torch.arange(max(-i, 0), min(N, N - i), device=device)
            mask[:, h, indices, indices + i] = 1
            # Create diagonal masks (negative offsets)
            indices = torch.arange(max(i, 0), min(N, N + i), device=device)
            mask[:, h, indices, indices - i] = 1

    # Optionally extend the mask to include a class token in the first row and column
    if add_class_token:
        mask_extended = torch.ones((B, H, N + 1, N + 1), device=device, dtype=dtype)
        mask_extended[:, :, 1:, 1:] = mask
        return mask_extended
    
    return mask

def generate_head_indices(N, h, wmin, is_modified):
    """
    Generate head indices based on the Wythoff's sequence.
    
    Args:
        N (int): Number of tokens.
        h (int): Number of attention heads.
        wmin (int): Minimum window size.
        is_modified (bool): Whether to use a modified sequence.
        
    Returns:
        list: A list (one per head) with tensors of indices.
    """
    wmax = N // 3
    headindices = [[] for _ in range(h)]
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    for i in range(1, h + 1):
        a = int(math.floor(math.floor(i * phi) * phi))
        b = int(math.floor(math.floor(i * phi) * (phi ** 2)))
        w = wmin + int((wmax - wmin) / (h - 1) * (i - 1))
        if is_modified:
            b_Wyt_m = b - a
            a_Wyt_m = a - b_Wyt_m
            headindices[i - 1] = get_fibonacci(a_Wyt_m, b_Wyt_m, w)
        else:
            headindices[i - 1] = get_fibonacci(a, b, w)
    
    headindices = [torch.tensor(seq, dtype=torch.int64) for seq in headindices]
    return headindices

def get_fibonacci(a, b, w):
    """
    Generate a Fibonacci-like sequence until the last term exceeds the window w.
    
    Args:
        a (int): First term.
        b (int): Second term.
        w (int): Maximum allowed value for the sequence (exclusive).
    
    Returns:
        list: Fibonacci sequence (adjusted) not exceeding w.
    """
    fib_seq = [a, b]
    while fib_seq[-1] <= w:
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    return fib_seq[:-1]

def shuffle(seed, array_of_sets):
    """
    Shuffle the array of index sets using the provided seed.
    
    Args:
        seed (int): The seed value for randomness.
        array_of_sets (list): List of index sets (each a tensor or list).
    
    Returns:
        list: Shuffled array of index sets.
    """
    random.seed(seed)
    shuffled_array = array_of_sets[:]
    random.shuffle(shuffled_array)
    return shuffled_array
