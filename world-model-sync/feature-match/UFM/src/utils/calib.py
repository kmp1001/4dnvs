import numpy as np
import torch
from py123d.datatypes.sensors.base_camera import Camera, CameraModel


def make_se3_from_se2_array(se2):
    x, y, yaw = float(se2[0]), float(se2[1]), float(se2[2])
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4, dtype=np.float32)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = x
    T[1, 3] = y
    return T


def quat_wxyz_to_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    qw, qx, qy, qz = q
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def transform_7d_numpy_to_matrix4x4(tq):
    tq = np.asarray(tq, dtype=np.float32).reshape(-1)
    if tq.shape[0] != 7:
        raise ValueError(f"Expected 7D transform, got {tq.shape}")
    tx, ty, tz, qw, qx, qy, qz = tq
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_wxyz_to_matrix(qw, qx, qy, qz)
    T[:3, 3] = [tx, ty, tz]
    return T


def ensure_4x4_transform(T_like):
    if hasattr(T_like, "matrix"):
        T_like = T_like.matrix
    elif hasattr(T_like, "array"):
        T_like = T_like.array
    T = np.asarray(T_like, dtype=np.float32)
    if T.shape == (4, 4):
        return T
    if T.size == 7:
        return transform_7d_numpy_to_matrix4x4(T)
    raise ValueError(f"Unexpected transform shape: {T.shape}")


def extract_camera_to_ego(cam):
    if hasattr(cam.metadata, "camera_to_imu_se3"):
        return ensure_4x4_transform(cam.metadata.camera_to_imu_se3)
    return np.eye(4, dtype=np.float32)


def extract_camera_intrinsics(cam: Camera, old_w: int, old_h: int):
    model = cam.metadata.camera_model
    if model == CameraModel.PINHOLE:
        return np.array(cam.metadata.intrinsics.camera_matrix, dtype=np.float32).copy()

    if model == CameraModel.FISHEYE_MEI:
        proj = cam.metadata.projection
        g1 = proj.gamma1 if proj is not None else 1.0
        g2 = proj.gamma2 if proj is not None else 1.0
        u0 = proj.u0 if proj is not None else old_w / 2.0
        v0 = proj.v0 if proj is not None else old_h / 2.0
        return np.array([[g1, 0, u0], [0, g2, v0], [0, 0, 1]], dtype=np.float32)

    if model == CameraModel.FTHETA:
        intr = cam.metadata.intrinsics
        poly = intr.fw_poly if intr is not None else [0, 500]
        f = poly[1] if len(poly) > 1 else 500.0
        cx = intr.cx if intr is not None else old_w / 2.0
        cy = intr.cy if intr is not None else old_h / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

    return np.array(
        [[500, 0, old_w / 2], [0, 500, old_h / 2], [0, 0, 1]], dtype=np.float32
    )


def rescale_intrinsics(K, old_w, old_h, new_w, new_h):
    K = K.copy()
    K[0, 0] *= new_w / old_w
    K[0, 2] *= new_w / old_w
    K[1, 1] *= new_h / old_h
    K[1, 2] *= new_h / old_h
    return K.astype(np.float32)


def apply_render_pose_convention(cam_c2w, enabled=True):
    if not enabled:
        return cam_c2w

    # Match the working point-cloud renderer convention.
    fix = torch.eye(4, dtype=cam_c2w.dtype, device=cam_c2w.device)
    fix[0, 0] = -1.0
    fix[1, 1] = -1.0
    return fix.unsqueeze(0) @ cam_c2w


def shift_camera_poses_local(cam_c2w, offset_xyz_m):
    """Translate camera-to-world poses in the camera's local coordinate frame.

    The renderer uses the OpenCV camera convention: +x points right, +y points
    down, and +z points forward. Therefore a virtual camera two metres left of
    the real camera uses ``(-2, 0, 0)``, while two metres up uses ``(0, -2, 0)``.

    Args:
        cam_c2w: Camera-to-world pose tensor with shape ``[..., 4, 4]``.
        offset_xyz_m: Local camera-frame translation ``(x, y, z)`` in metres.

    Returns:
        A shifted pose tensor with the same shape, dtype, and device as
        ``cam_c2w``. The input tensor is not modified.
    """
    if not isinstance(cam_c2w, torch.Tensor):
        raise TypeError("cam_c2w must be a torch.Tensor")
    if cam_c2w.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected cam_c2w shape [..., 4, 4], got {tuple(cam_c2w.shape)}"
        )

    offset = torch.as_tensor(
        offset_xyz_m,
        dtype=cam_c2w.dtype,
        device=cam_c2w.device,
    )
    if offset.numel() != 3:
        raise ValueError(
            "offset_xyz_m must contain exactly three values (x, y, z), "
            f"got shape {tuple(offset.shape)}"
        )

    delta = torch.eye(4, dtype=cam_c2w.dtype, device=cam_c2w.device)
    delta[:3, 3] = offset.reshape(3)
    return cam_c2w @ delta
