import torch
import torch.nn.functional as F

def split_shuffle_reassemble(x, a=3, b=2):
    bs, c, h, w = x.shape
    patch_size_h, patch_size_w = h // a, w // b
    output = torch.zeros((bs, c, 224, 224), device=x.device)
    
    for i in range(bs):
        # split
        patches = []
        for ay in range(a):
            for bx in range(b):
                start_y = ay * patch_size_h
                start_x = bx * patch_size_w
                patch = x[i, :, start_y:start_y+patch_size_h, start_x:start_x+patch_size_w]
                patches.append(patch)
        
        # random shuffle
        random_indices = torch.randperm(a * b)
        shuffled_patches = [patches[idx] for idx in random_indices]
        
        # reassemble
        for idx, patch in enumerate(shuffled_patches):
            ay, bx = divmod(idx, b)
            start_y = ay * (224 // a)
            start_x = bx * (224 // b)
            end_y = start_y + (224 // a)
            end_x = start_x + (224 // b)
            # adjust the origin size
            patch_resized = F.interpolate(patch.unsqueeze(0), size=(end_y - start_y, end_x - start_x), mode='bilinear', align_corners=False)
            output[i, :, start_y:end_y, start_x:end_x] = patch_resized.squeeze(0)
    
    return output
