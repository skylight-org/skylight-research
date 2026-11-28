"""Independent NumPy Algorithms 1/2 oracle; not released author code."""

from types import SimpleNamespace

import numpy as np
import torch


def array(tensor):
    return tensor.detach().to(dtype=torch.float64, device="cpu").numpy()


def unpack(packed, bits, count):
    """Independent Python integer bitstream decoding."""
    data = int.from_bytes(bytes(packed.cpu().tolist()), "little")
    return np.array(
        [(data >> (index * bits)) & (2**bits - 1) for index in range(count)],
        dtype=np.int64,
    )


def encode(x, state, *, variant="mse", bits=3):
    """Given explicit matrices/table, evaluate the paper formulas in FP64."""
    data = array(x)
    rho = np.sqrt(np.sum(data * data, axis=-1, keepdims=True))
    unit = data / np.where(rho > 0, rho, 1.0)
    rotation, levels = array(state["rotation"]), array(state["codebook"])
    coarse_bits = bits - (variant == "prod")
    if coarse_bits:
        rotated = unit @ rotation
        edges = (levels[:-1] + levels[1:]) / 2
        indices = np.searchsorted(edges, rotated, side="left")
        indices = np.where(rho == 0, 0, indices)
        coarse = levels[indices] @ rotation.T
        coarse = np.where(rho == 0, 0, coarse)
    else:
        indices = np.empty(0, dtype=np.int64)
        coarse = np.zeros_like(unit)
    gamma, sign_bits = None, None
    estimate = coarse
    if variant == "prod":
        projection = array(state["projection"])
        residual = unit - coarse
        gamma = np.sqrt(np.sum(residual * residual, axis=-1, keepdims=True))
        sign_bits = (residual @ projection.T) >= 0
        estimate = coarse + (
            ((2 * sign_bits.astype(np.float64) - 1) @ projection)
            * gamma
            * (np.sqrt(np.pi / 2) / data.shape[-1])
        )
    return SimpleNamespace(
        indices=indices,
        norms=rho,
        coarse=coarse,
        residual_norms=gamma,
        sign_bits=sign_bits,
        reconstructed=estimate * rho,
    )


def decode(record, state):
    """Reference reconstruction using actual stored codes/norms and explicit state."""
    rotation, levels = array(state["rotation"]), array(state["codebook"])
    coarse_bits = record.metadata["coarse_bits"]
    if coarse_bits:
        indices = unpack(record.codes, coarse_bits, record.metadata["code_count"])
        coarse = levels[indices.reshape(record.shape)] @ rotation.T
        coarse = np.where(array(record.norms) == 0, 0, coarse)
    else:
        coarse = np.zeros(record.shape, dtype=np.float64)
    if record.qjl_codes is not None:
        signs = unpack(record.qjl_codes, 1, record.metadata["qjl_code_count"])
        signs = (2 * signs - 1).reshape(record.shape)
        coarse += (
            (signs @ array(state["projection"]))
            * array(record.residual_norms)
            * (np.sqrt(np.pi / 2) / record.shape[-1])
        )
    return coarse * array(record.norms)


def conditional_sphere_mean(low, high, dimension):
    """Independent analytic moment and SciPy incomplete-Beta probability mass."""
    from scipy.special import betainc, gammaln

    alpha = (dimension - 1) / 2
    a, b = (low + 1) / 2, (high + 1) / 2
    if low >= 0:
        # Symmetry supplies the upper-tail CDF without subtracting from one.
        mass = betainc(alpha, alpha, (1 - low) / 2) - betainc(
            alpha, alpha, (1 - high) / 2
        )
    else:
        mass = betainc(alpha, alpha, b) - betainc(alpha, alpha, a)
    factor = np.exp(
        gammaln(dimension / 2) - 0.5 * np.log(np.pi) - gammaln((dimension - 1) / 2)
    )
    moment = (
        factor
        * (max(0.0, 1 - low * low) ** alpha - max(0.0, 1 - high * high) ** alpha)
        / (dimension - 1)
    )
    return moment / mass
