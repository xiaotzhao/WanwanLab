#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : rotation.py
@Path    : envs/common/rotation.py
@Desc    : Pose & rotation utility module.
            Implements common rotation operations including Euler angles,
            quaternions, rotation matrices and coordinate transformations.
            Designed for simulation environments, robot pose calculation
            and posture conversion tasks.
@Author  : xiaotong Zhao
@Project : WanwanLab
@Date    : 2026-06-15
"""


import numpy as np


def np_quat_conjugate_batched(q: np.ndarray) -> np.ndarray:
    """
    Compute the conjugate of quaternions.
    Quaternion format: [w, x, y, z] (w-first).
    Supports arbitrary batch dimensions with shape (..., 4).
    For unit quaternion, conjugate equals inverse.

    Args:
        q: Input quaternion array, last dimension must be 4.

    Returns:
        Conjugated quaternion array with the same shape as input.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {q.shape}")
    conj = np.array(q, copy=True)
    conj[..., 1:] *= -1
    return conj


def np_quat_mul_batched(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternion arrays with broadcast support.
    Quaternion format: [w, x, y, z] (w-first).
    Supports arbitrary leading dimensions, last dimension is fixed to 4.

    Args:
        q1: First quaternion array, shape (..., 4)
        q2: Second quaternion array, shape (..., 4)

    Returns:
        Quaternion product after broadcasting, shape (..., 4)
    """
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got {q1.shape} and {q2.shape}")

    lead_shape = np.broadcast_shapes(q1.shape[:-1], q2.shape[:-1])
    q1 = np.broadcast_to(q1, (*lead_shape, 4))
    q2 = np.broadcast_to(q2, (*lead_shape, 4))

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def np_quat_canonicalize(q: np.ndarray) -> np.ndarray:
    """
    Canonicalize quaternion sign: enforce real part w >= 0.
    Note: q and -q represent the identical rotation.
    Standardize representation to eliminate sign ambiguity.

    Args:
        q: Input quaternion, shape (4,) or (N, 4), w-first format.

    Returns:
        Canonicalized quaternion with non-negative w component.
    """
    q_was_1d = q.ndim == 1
    if q_was_1d:
        q = q[None, :]

    sign = np.where(q[:, 0:1] < 0.0, -1.0, 1.0)
    result = q * sign
    canonical: np.ndarray = result[0] if q_was_1d else result
    return canonical


def np_quat_ensure_continuity(q: np.ndarray) -> np.ndarray:
    """
    Ensure continuity for quaternion time sequence (T, 4).
    Flip sign of current quaternion if dot product with previous frame < 0.
    Avoid sudden jumps caused by quaternion sign ambiguity in trajectory.

    Args:
        q: Quaternion time sequence, shape (T, 4), T = number of frames.

    Returns:
        Continuous quaternion sequence with smooth frame transition.
    """
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternion sequence with shape (T, 4), got {q.shape}")

    result = np.array(q, copy=True)
    for i in range(1, result.shape[0]):
        if float(np.dot(result[i - 1], result[i])) < 0.0:
            result[i] *= -1.0
    return result



def np_quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion batch (N, 4), w-first, to axis-angle vectors (N, 3).

    Adapted from PyTorch3D. Uses atan2 + Taylor expansion for numerical
    stability near zero rotation.

    Args:
        q: Unit quaternion array with shape (N, 4), order [w, x, y, z].

    Returns:
        Axis-angle representation (axis * angle), shape (N, 3).
    """
    q = np_quat_canonicalize(q)
    xyz = q[:, 1:]  # (N, 3) imaginary part
    w = q[:, 0:1]  # (N, 1) real part
    norms = np.linalg.norm(xyz, axis=-1, keepdims=True)  # (N, 1)
    half_angle = np.arctan2(norms, w)  # (N, 1)
    angle = 2.0 * half_angle  # (N, 1)
    small = np.abs(angle) < 1e-6  # (N, 1)
    safe_angle = np.where(small, 1.0, angle)
    sin_half_over_angle = np.where(
        small,
        0.5 - angle**2 / 48.0,
        np.sin(half_angle) / safe_angle,
    )
    axis_angle: np.ndarray = xyz / sin_half_over_angle
    return axis_angle


def np_quat_angular_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """Estimate angular velocity from a quaternion time sequence using shortest-arc diffs.

     Args:
         q: Quaternion time sequence with shape (T, 4), w-first unit quaternion.
         dt: Time interval between adjacent frames, must be positive.

     Returns:
         Angular velocity sequence (T, 3), compact axis-angle format (rad/s).
     """
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternion sequence with shape (T, 4), got {q.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    rotations = np_quat_ensure_continuity(q)
    num_frames = rotations.shape[0]
    omega = np.zeros((num_frames, 3), dtype=rotations.dtype)
    if num_frames <= 1:
        return omega

    if num_frames == 2:
        q_rel = np_quat_mul_batched(rotations[1], np_quat_conjugate_batched(rotations[0]))
        q_rel = np_quat_canonicalize(q_rel)
        angvel = np_quat_to_axis_angle(q_rel[None, :])[0] / dt
        omega[:] = angvel
        return omega

    q_prev = rotations[:-2]
    q_next = rotations[2:]
    q_rel = np_quat_mul_batched(q_next, np_quat_conjugate_batched(q_prev))
    q_rel = np_quat_canonicalize(q_rel)
    omega[1:-1] = np_quat_to_axis_angle(q_rel) / (2.0 * dt)
    omega[0] = omega[1]
    omega[-1] = omega[-2]
    return omega
