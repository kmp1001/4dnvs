import cv2
import numpy as np
import torch
import os
from depth_anything_3.api import DepthAnything3


def load_da3_model(model_path, device):
    model = DepthAnything3.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    return model


def write_video_tensor_to_images(video, out_dir):
    paths = []
    for i in range(video.shape[0]):
        frame = video[i].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        path = os.path.join(out_dir, f"{i:04d}.jpg")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def run_da3_depth_with_calibration(
    model,
    frame_paths,
    K_seq,
    T_world_seq,
    output_size,
    keep_conf_percent=0.95,
    return_conf_mask=False,
):
    K = K_seq.detach().cpu().numpy().astype(np.float32)
    T = T_world_seq.detach().cpu().numpy().astype(np.float32)

    # DA3 wants world-to-camera extrinsics.
    pred = model.inference(
        frame_paths,
        intrinsics=K,
        extrinsics=np.linalg.inv(T),
    )
    depth = pred.depth.astype(np.float32)  # [F, H0, W0]

    # DA3 confidence. Different versions may use different field names.
    if hasattr(pred, "conf"):
        conf = pred.conf.astype(np.float32)
    elif hasattr(pred, "confidence"):
        conf = pred.confidence.astype(np.float32)
    else:
        raise AttributeError(
            "DA3 prediction does not contain pred.conf or pred.confidence."
        )

    out_h, out_w = output_size

    # Resize depth to output size.
    depth_t = torch.from_numpy(depth).unsqueeze(1).float()
    depth_t = torch.nn.functional.interpolate(
        depth_t,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    depth_resized = depth_t.squeeze(1).cpu().numpy().astype(np.float32)

    # Resize confidence to output size.
    conf_t = torch.from_numpy(conf).unsqueeze(1).float()
    conf_t = torch.nn.functional.interpolate(
        conf_t,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    conf_resized = conf_t.squeeze(1).cpu().numpy().astype(np.float32)

    # Keep top keep_conf_percent confidence pixels per frame.
    F = conf_resized.shape[0]
    conf_mask = np.zeros_like(conf_resized, dtype=bool)

    for i in range(F):
        conf_i = conf_resized[i]
        valid = np.isfinite(conf_i)

        if valid.sum() == 0:
            continue

        thresh = np.quantile(
            conf_i[valid],
            1.0 - float(keep_conf_percent),
        )

        conf_mask[i] = valid & (conf_i >= thresh)

    # Remove low-confidence depth.
    depth_resized_filtered = depth_resized.copy()
    depth_resized_filtered[~conf_mask] = 0.0

    if return_conf_mask:
        return depth_resized_filtered, conf_mask, conf_resized

    return depth_resized_filtered
