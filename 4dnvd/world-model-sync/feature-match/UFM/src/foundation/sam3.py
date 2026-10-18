import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor

from utils.utils import cfg_get


def load_sam3_video_predictor(checkpoint_path):
    return build_sam3_video_predictor(checkpoint_path=checkpoint_path)


def get_sam3_prompts(config):
    prompts = cfg_get(config, "sam3.prompts", None)

    if prompts is None:
        # Backward compatibility with old config: "sam3.prompt": "vehicle"
        prompts = cfg_get(config, "sam3.prompt", "vehicle")

    if isinstance(prompts, str):
        prompts = [prompts]

    prompts = [str(p).strip() for p in prompts if str(p).strip()]
    assert len(prompts) > 0, "No SAM3 prompts provided."

    return prompts


def get_sam3_sky_prompts(config):
    prompts = cfg_get(config, "sam3.sky_prompts", None)

    if prompts is None:
        # Backward compatibility / simple override: "sam3.sky_prompt": "sky"
        prompts = cfg_get(config, "sam3.sky_prompt", "sky")

    if prompts is None:
        return []

    if isinstance(prompts, str):
        prompts = [prompts]

    prompts = [str(p).strip() for p in prompts if str(p).strip()]
    return prompts


def dilate_bool_mask(mask, radius):
    if radius <= 0:
        return mask.astype(bool)

    k = 2 * int(radius) + 1
    return cv2.dilate(
        mask.astype(np.uint8),
        np.ones((k, k), dtype=np.uint8),
        iterations=1,
    ).astype(bool)


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def force_weight_dtype_inputs(model):
    def pre_hook(module, inputs):
        weight = getattr(module, "weight", None)
        if weight is None or not torch.is_tensor(weight):
            return inputs

        target_dtype = weight.dtype
        target_device = weight.device

        fixed = []
        changed = False

        for x in inputs:
            if torch.is_tensor(x) and x.is_floating_point():
                if x.dtype != target_dtype or x.device != target_device:
                    x = x.to(device=target_device, dtype=target_dtype)
                    changed = True
            fixed.append(x)

        return tuple(fixed) if changed else inputs

    for module in model.modules():
        if hasattr(module, "weight"):
            module.register_forward_pre_hook(pre_hook)

    return model


def _as_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resize_mask_if_needed(mask, height, width):
    mask = np.squeeze(mask).astype(bool)
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def _run_sam3_video_propagation_once(
    video_predictor,
    session_id,
    num_frames,
    height,
    width,
    score_thres,
    start_frame_index,
    propagation_direction,
):
    dynamic_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    seg_frames = [[] for _ in range(num_frames)]

    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": propagation_direction,
        "start_frame_index": int(start_frame_index),
        "max_frame_num_to_track": int(num_frames),
    }

    for result in video_predictor.handle_stream_request(request):
        frame_idx = int(result["frame_index"])
        if frame_idx < 0 or frame_idx >= num_frames:
            continue

        outputs = result.get("outputs", None)
        if outputs is None:
            continue

        obj_ids = _as_numpy(outputs.get("out_obj_ids"))
        probs = _as_numpy(outputs.get("out_probs"))
        boxes_xywh = _as_numpy(outputs.get("out_boxes_xywh"))
        masks = _as_numpy(outputs.get("out_binary_masks"))

        if obj_ids is None or masks is None or len(obj_ids) == 0:
            continue

        probs = (
            np.ones((len(obj_ids),), dtype=np.float32)
            if probs is None
            else probs.reshape(-1).astype(np.float32)
        )
        keep = probs >= float(score_thres)
        if keep.sum() == 0:
            continue

        obj_ids = obj_ids[keep]
        probs = probs[keep]
        masks = masks[keep]
        if boxes_xywh is not None:
            boxes_xywh = boxes_xywh[keep]

        frame_union = np.zeros((height, width), dtype=np.bool_)
        for j in range(len(obj_ids)):
            mask = _resize_mask_if_needed(masks[j], height, width)
            frame_union |= mask

            box_xyxy = None
            if boxes_xywh is not None:
                x, y, w, h = boxes_xywh[j].astype(np.float32).tolist()
                box_xyxy = np.array(
                    [x * width, y * height, (x + w) * width, (y + h) * height],
                    dtype=np.float32,
                )

            seg_frames[frame_idx].append(
                {
                    "id": int(obj_ids[j]),
                    "score": float(probs[j]),
                    "box_xyxy": box_xyxy,
                    "mask": mask,
                }
            )

        dynamic_mask[frame_idx] |= frame_union

    return dynamic_mask, seg_frames


def run_sam3_vehicle_masks_video_based(
    video_predictor,
    left_video,  # torch [F, 3, H, W], RGB in [0, 1]
    prompts=("vehicle",),
    score_thres=0.5,
    dilate_radius=3,
    prompt_frame_index=0,
    propagation_direction="forward",
):
    """
    Run official SAM3 video predictor with one or more text prompts.

    Each prompt is run in its own SAM3 video session. The final dynamic mask is the
    union over all prompts. Instance ids are made globally unique across prompts.

    Returns:
        dynamic_mask: torch.bool [F, H, W]
            Union mask of all accepted SAM3 instances from all prompts.

        seg_frames: list[list[dict]]
            Per-frame instance metadata. Each dict includes prompt, score, raw_sam3_id,
            globally unique id, box_xyxy, and mask.

        instance_id_maps: torch.long [F, H, W]
            0 = background.
            >0 = globally unique instance id across all prompts.
    """
    assert left_video.ndim == 4, f"Expected [F, 3, H, W], got {left_video.shape}"
    assert left_video.shape[1] == 3, (
        f"Expected RGB channel dim=3, got {left_video.shape}"
    )

    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = [str(p).strip() for p in prompts if str(p).strip()]
    assert len(prompts) > 0, "No SAM3 prompts provided."

    num_frames = left_video.shape[0]
    height = left_video.shape[2]
    width = left_video.shape[3]

    # Convert torch video to list[PIL.Image] once, then reuse it across prompt sessions.
    video_np = left_video.detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    video_pil = [
        Image.fromarray((video_np[i] * 255.0).astype(np.uint8))
        for i in range(num_frames)
    ]

    dynamic_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    instance_id_maps = np.zeros((num_frames, height, width), dtype=np.int32)
    seg_frames = [[] for _ in range(num_frames)]

    next_global_instance_id = 1

    print(
        f"Running SAM3 video masks: prompts={prompts}, "
        f"prompt_frame_index={prompt_frame_index}, "
        f"direction={propagation_direction}"
    )

    for prompt_idx, prompt in enumerate(prompts):
        response = video_predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": video_pil,
            }
        )
        session_id = response["session_id"]

        # Raw SAM3 object ids may restart for each prompt/session, so remap them to
        # globally unique ids for visualization and metadata.
        raw_to_global_id = {}

        try:
            video_predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": int(prompt_frame_index),
                    "text": prompt,
                }
            )

            stream_request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": propagation_direction,
                "start_frame_index": int(prompt_frame_index),
                "max_frame_num_to_track": int(num_frames),
            }

            for result in video_predictor.handle_stream_request(stream_request):
                frame_idx = int(result["frame_index"])

                if frame_idx < 0 or frame_idx >= num_frames:
                    continue

                outputs = result.get("outputs", None)
                if outputs is None:
                    continue

                out_obj_ids = _as_numpy(outputs.get("out_obj_ids"))
                out_probs = _as_numpy(outputs.get("out_probs"))
                out_boxes_xywh = _as_numpy(outputs.get("out_boxes_xywh"))
                out_binary_masks = _as_numpy(outputs.get("out_binary_masks"))

                if (
                    out_obj_ids is None
                    or out_binary_masks is None
                    or len(out_obj_ids) == 0
                ):
                    continue

                out_obj_ids = np.asarray(out_obj_ids).reshape(-1)
                if out_probs is None:
                    out_probs = np.ones((len(out_obj_ids),), dtype=np.float32)
                else:
                    out_probs = np.asarray(out_probs).reshape(-1).astype(np.float32)

                out_binary_masks = np.asarray(out_binary_masks)
                if out_binary_masks.ndim == 4:
                    # Common shapes: [N, 1, H, W] or [N, H, W, 1].
                    out_binary_masks = np.squeeze(out_binary_masks)

                # After squeeze, a single instance can become [H, W]. Restore [N, H, W].
                if out_binary_masks.ndim == 2:
                    out_binary_masks = out_binary_masks[None]

                if out_boxes_xywh is not None:
                    out_boxes_xywh = np.asarray(out_boxes_xywh)

                for j in range(len(out_obj_ids)):
                    score = float(out_probs[j])
                    if score < float(score_thres):
                        continue

                    raw_id = int(out_obj_ids[j])
                    if raw_id not in raw_to_global_id:
                        raw_to_global_id[raw_id] = next_global_instance_id
                        next_global_instance_id += 1

                    global_id = int(raw_to_global_id[raw_id])

                    mask = _resize_mask_if_needed(out_binary_masks[j], height, width)

                    if dilate_radius > 0:
                        mask = dilate_bool_mask(mask, int(dilate_radius))

                    if mask.sum() == 0:
                        continue

                    dynamic_mask[frame_idx] |= mask

                    # In overlaps, later prompts overwrite earlier prompts in the instance map.
                    # The union dynamic mask is unaffected.
                    instance_id_maps[frame_idx][mask] = global_id

                    if out_boxes_xywh is not None and len(out_boxes_xywh) > j:
                        box_xywh = out_boxes_xywh[j].astype(np.float32)
                        x, y, w, h = box_xywh.tolist()

                        # Official SAM3 typically returns normalized xywh. If the values look
                        # already pixel-space, keep them as pixels.
                        if max(abs(x), abs(y), abs(w), abs(h)) <= 2.0:
                            box_xyxy = np.array(
                                [
                                    x * width,
                                    y * height,
                                    (x + w) * width,
                                    (y + h) * height,
                                ],
                                dtype=np.float32,
                            )
                        else:
                            box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
                    else:
                        ys, xs = np.where(mask)
                        box_xyxy = np.array(
                            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                            dtype=np.float32,
                        )

                    seg_frames[frame_idx].append(
                        {
                            "id": global_id,
                            "raw_sam3_id": raw_id,
                            "prompt_index": int(prompt_idx),
                            "prompt": prompt,
                            "keyword": prompt,
                            "score": score,
                            "box_xyxy": box_xyxy,
                            "mask": mask,
                        }
                    )

        finally:
            try:
                video_predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                    }
                )
            except Exception:
                pass

    dynamic_mask_t = torch.from_numpy(dynamic_mask).bool()
    instance_id_maps_t = torch.from_numpy(instance_id_maps).long()

    return dynamic_mask_t, seg_frames, instance_id_maps_t
