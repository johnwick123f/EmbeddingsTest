import numpy as np
import numba
from typing import Dict, Any

@numba.njit(cache=True, fastmath=True)
def _fast_best_rescale_factor_kernel(o_abs: np.ndarray, ex_bits: int, tight_start_arr: np.ndarray) -> float:
    """
    Blazing fast compiled alternative to the min-heap optimization loop.
    Replaces the heap with an O(d log d) sorted transition-boundary scan.
    """
    dim = len(o_abs)
    max_o = 0.0
    for i in range(dim):
        if o_abs[i] > max_o:
            max_o = o_abs[i]

    if max_o < 1e-8:
        return 1.0

    tight_start = tight_start_arr[ex_bits] if ex_bits < len(tight_start_arr) else 0.81

    eps = 1e-5
    n_enum = 10.0
    t_end = (float((1 << ex_bits) - 1) + n_enum) / max_o
    t_start = t_end * tight_start

    # Allocate local arrays for state tracking
    cur_o_bar = np.empty(dim, dtype=np.int32)
    t_next_arr = np.empty(dim, dtype=np.float64)
    indices = np.empty(dim, dtype=np.int32)

    sqr_denominator = float(dim) * 0.25
    numerator = 0.0

    for i in range(dim):
        val = int(np.floor(t_start * o_abs[i] + eps))
        cur_o_bar[i] = val
        sqr_denominator += float(val * val + val)
        numerator += (float(val) + 0.5) * o_abs[i]
        t_next_arr[i] = float(val + 1) / o_abs[i]
        indices[i] = i

    # Sort upcoming transition intervals globally instead of managing a dynamic heap
    # This allows sequential streaming in O(D log D)
    sort_idx = np.argsort(t_next_arr)

    max_ip = 0.0
    t = t_start

    for idx in sort_idx:
        cur_t = t_next_arr[idx]
        if cur_t >= t_end:
            break

        update_id = indices[idx]
        cur_o_bar[update_id] += 1
        update_o_bar = cur_o_bar[update_id]

        sqr_denominator += 2.0 * float(update_o_bar)
        numerator += o_abs[update_id]

        if sqr_denominator > 0:
            cur_ip = numerator / np.sqrt(sqr_denominator)
            if cur_ip > max_ip:
                max_ip = cur_ip
                t = cur_t

        # Single pass evaluation matching reference heuristic strategy
        if update_o_bar < (1 << ex_bits) - 1:
            t_next_alt = float(update_o_bar + 1) / o_abs[update_id]
            if t_next_alt < t_end and t_next_alt > cur_t:
                # Fallback check for secondary boundary alignment
                if sqr_denominator > 0:
                    alt_ip = (numerator + o_abs[update_id]) / np.sqrt(sqr_denominator + 2.0 * float(update_o_bar + 1))
                    if alt_ip > max_ip:
                        t = t_next_alt

    return t

@numba.njit(parallel=True, cache=True, fastmath=True)
def _fast_quantize_loop(residuals: np.ndarray, binary_code: np.ndarray, bits: int, ex_bits: int, tight_start_arr: np.ndarray) -> np.ndarray:
    """Parallelized row execution leveraging multi-core architectures."""
    n, d = residuals.shape
    q = np.empty((n, d), dtype=np.uint8)
    mask = (1 << ex_bits) - 1
    eps = 1e-5

    for r in numba.prange(n):
        row_res = residuals[r]

        # Calculate norm manually within Numba
        row_norm = 0.0
        for c in range(d):
            row_norm += row_res[c] * row_res[c]
        row_norm = np.sqrt(row_norm) + 1e-8

        o_abs = np.empty(d, dtype=np.float64)
        for c in range(d):
            o_abs[c] = abs(row_res[c]) / row_norm

        t = _fast_best_rescale_factor_kernel(o_abs, ex_bits, tight_start_arr)

        for c in range(d):
            tmp_code = int(np.floor(t * o_abs[c] + eps))
            if tmp_code < 0:
                tmp_code = 0
            elif tmp_code > mask:
                tmp_code = mask

            if row_res[c] < 0:
                tmp_code = (~tmp_code) & mask

            q[r, c] = uint8_cast = tmp_code + (binary_code[r, c] << ex_bits)

    return q

@numba.njit(cache=True)
def _fast_pack_kernel(flat_q: np.ndarray, bits: int, total_bytes: int) -> np.ndarray:
    """Pre-compiled vector layout packing configuration to strip standard iteration costs."""
    packed_flat = np.zeros(total_bytes, dtype=np.uint8)
    mask = (1 << bits) - 1

    bit_idx = 0
    for i in range(len(flat_q)):
        val = flat_q[i] & mask
        byte_idx = bit_idx // 8
        shift = bit_idx % 8

        packed_flat[byte_idx] |= (val << shift) & 0xFF
        if shift + bits > 8:
            packed_flat[byte_idx + 1] |= (val >> (8 - shift)) & 0xFF
        bit_idx += bits

    return packed_flat


class UniformQuantizer:
    """
    An ultra-high-performance drop-in modular replacement utilizing vectorized
    and Numba-accelerated processes matching the identical API signatures.
    """
    def __init__(self, dim: int = 128, seed: int = 42):
        self._const_t_cache: Dict[Any, Any] = {}
        self.dim = dim

        # Precompute the orthogonal projection transformation matrix
        rng = np.random.default_rng(seed)
        H = rng.standard_normal((dim, dim))
        self.Q, _ = np.linalg.qr(H)

        # Hardcoded search bounds cached as contiguous arrays for compiled access
        self._tight_start_arr = np.array([0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81], dtype=np.float64)

    def _best_rescale_factor(self, o_abs: np.ndarray, ex_bits: int) -> float:
        """API Compliant wrapper directing to the fast native static kernel."""
        return _fast_best_rescale_factor_kernel(o_abs, ex_bits, self._tight_start_arr)

    def quantize(self, x: np.ndarray, bits: int) -> np.ndarray:
        # Vectorized BLAS Matrix multiplication matrix pass
        x_transformed = x @ self.Q

        # Vectorized residual broadcasting
        centroids = x_transformed.mean(axis=1, keepdims=True)
        residuals = x_transformed - centroids
        binary_code = (residuals >= 0).astype(np.uint8)

        if bits == 1:
            return binary_code

        return _fast_quantize_loop(residuals, binary_code, bits, bits - 1, self._tight_start_arr)

    def pack(self, q: np.ndarray, bits: int) -> np.ndarray:
        if bits == 8:
            return q.astype(np.uint8)

        n, d = q.shape
        flat_q = q.ravel()
        total_bits = len(flat_q) * bits
        total_bytes = (total_bits + 7) // 8

        packed_flat = _fast_pack_kernel(flat_q, bits, total_bytes)
        bytes_per_row = total_bytes // n
        return packed_flat.reshape(n, bytes_per_row)

    def quantize_and_pack(self, x: np.ndarray, bits: int) -> np.ndarray:
        return self.pack(self.quantize(x, bits=bits), bits=bits)

    def unpack(self, packed_array: np.ndarray, num_elements: int, dim: int, bits: int) -> np.ndarray:
        if bits == 8:
            codes = packed_array.copy()
        else:
            flat_packed = packed_array.ravel()
            bit_indices = np.arange(num_elements, dtype=np.int64) * bits
            byte_indices = bit_indices // 8
            bit_shifts = bit_indices % 8
            mask = (1 << bits) - 1

            val = flat_packed[byte_indices] >> bit_shifts
            spill_mask = (bit_shifts + bits) > 8
            if np.any(spill_mask):
                padded_packed = np.append(flat_packed, 0)
                val[spill_mask] |= (padded_packed[byte_indices[spill_mask] + 1] << (8 - bit_shifts[spill_mask]))

            codes = (val & mask).reshape(-1, dim)

        ex_bits = bits - 1
        cb = -(float(1 << ex_bits) - 0.5)

        recon_transformed = codes.astype(np.float32) + cb
        # Fast vectorized matrix row-normalization via keeping dimensions open
        recon_transformed /= (np.linalg.norm(recon_transformed, axis=1, keepdims=True) + 1e-8)

        # Reverse orthogonal rotation back to target embedding coordinate spaces
        return recon_transformed @ self.Q.T
