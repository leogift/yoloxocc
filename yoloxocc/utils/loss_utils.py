import torch
import torch.nn.functional as F

__all__ = [
    "get_occ_iou",
]

@torch.no_grad()
def get_occ_iou(
    occ_pred,
    occ_centermask_target,
):
    # occ_pred B,Y,Z,X
    # occ_centermask_target B,Y,Z,X

    assert occ_pred.shape[0] == occ_centermask_target.shape[0] \
        and occ_pred.shape[1] == occ_centermask_target.shape[1], \
        f"expect {occ_pred.shape[:2]} == {occ_centermask_target.shape[:2]}"

    # 对齐
    Zp,Xp = occ_pred.shape[2:]
    Z,X = occ_centermask_target.shape[2:]
    if Zp!=Z or Xp!=X:
        occ_pred = F.adaptive_max_pool2d(occ_pred, (Z,X))

    # occ_pred logits
    i = (occ_pred.sigmoid().round() * occ_centermask_target.round()).sum()
    u = torch.clamp((occ_pred.sigmoid().round() + occ_centermask_target.round()).sum() - i, 1.5e-5)
    iou = i / u

    return iou
