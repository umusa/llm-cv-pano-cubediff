# import torch
# import torch.nn.functional as F

# def seam_loss(latents_pred, latents_gt, border=1):
#     """
#     MSE on a ‘ring’ of ±border pixels around each cube-face edge.
#     Args
#     ----
#     latents_pred , latents_gt : [B,6,C,H,W]  – predicted vs target noise
#     border                    : int          – width (pixels) of ring
#     """
#     # slices for the four vertical seams (L/R)
#     seam_pred = torch.cat([
#         latents_pred[:, 0, :, :, -border:],   # front-right
#         latents_pred[:, 1, :, :, -border:],   # right-back
#         latents_pred[:, 2, :, :, -border:],   # back-left
#         latents_pred[:, 3, :, :, -border:],   # left-front
#     ], dim=-1)
#     seam_gt   = torch.cat([
#         latents_gt[:, 0, :, :, -border:],
#         latents_gt[:, 1, :, :, -border:],
#         latents_gt[:, 2, :, :, -border:],
#         latents_gt[:, 3, :, :, -border:],
#     ], dim=-1)

#     # top & bottom rings
#     top_pred    = latents_pred[:, 4, :, :border, :]
#     bottom_pred = latents_pred[:, 5, :, -border:, :]
#     top_gt      = latents_gt[:, 4, :, :border, :]
#     bottom_gt   = latents_gt[:, 5, :, -border:, :]

#     seam_pred = torch.cat([seam_pred, top_pred, bottom_pred], dim=-1)
#     seam_gt   = torch.cat([seam_gt,   top_gt,   bottom_gt],   dim=-1)

#     return F.mse_loss(seam_pred, seam_gt, reduction="mean")



# def seam_loss(pred, target, border=1):
#     """
#     MSE between matching vertical borders of the four side faces.
#     pred , target : [B,6,C,H,W]
#     """
#     # face-index pairs (left slice, right slice)
#     # pairs = [(0,1),  # front → right
#     #          (1,2),  # right → back
#     #          (2,3),  # back  → left
#     #          (3,0)]  # left  → front

#     # loss = 0.0
#     # for a, b in pairs:
#     #     # rightmost 'border' columns of face a  vs  leftmost of face b
#     #     la = pred[:, a, :, :, -border:]
#     #     lb = target[:, b, :, :, :border]
#     #     loss += F.mse_loss(la, lb, reduction="mean")

#     # return loss / len(pairs)

#     def seam_loss(pred, border=8):
#     """
#     CubeDiff overlap-crop loss along the four vertical seams.
#     pred : [B,6,C,H,W]  – ε̂ at step t   (target ε is *not* needed)
#     """
#     seams = [(0,1), (1,2), (2,3), (3,0)]      # front→right→back→left
#     loss  = 0.
#     for a, b in seams:
#         # right border of a  vs  left border of b  (two directions)
#         loss += F.mse_loss(pred[:,a,:,:,-border:], pred[:,b,:,:,:border])
#         loss += F.mse_loss(pred[:,a,:,:,:border],  pred[:,b,:,:,-border:])
#     return loss / (2 * len(seams))

import torch.nn.functional as F

def seam_loss(pred, border=8):
    """
    CubeDiff overlap-crop loss along the four vertical seams.
    pred : [B,6,C,H,W]  – ε̂ at step t   (target ε is *not* needed)
    """
    seams = [(0,1), (1,2), (2,3), (3,0)]      # front→right→back→left
    loss  = 0.
    for a, b in seams:
        # right border of a  vs  left border of b  (two directions)
        loss += F.mse_loss(pred[:,a,:,:,-border:], pred[:,b,:,:,:border])
        loss += F.mse_loss(pred[:,a,:,:,:border],  pred[:,b,:,:,-border:])
    return loss / (2 * len(seams))


