#pragma once

#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"

/*
 * Copy n-by-m slice from slice_ptr to dst. Used for the naive single copy strategy.
 *
 * Copied straight from scipy.
 */
template<typename T>
void copy_slice_F(T* dst, const T* slice_ptr, const npy_intp n, const npy_intp m, const npy_intp s2, const npy_intp s1) {
    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = 0; j < m; j++) {
            dst[i + j*n] = *(slice_ptr + (i*s2/sizeof(T)) + (j*s1/sizeof(T)));  // == src[i*m + j]
        }
    }
}


/*
 * Copy n-by-m slice from slice_ptr to dst. In conjunction with `swap_cf` used for the
 * double copy strategy.
 *
 * Copied straight from scipy.
 */
template <typename T>
void copy_slice(T *dst, const T *slice_ptr, const npy_intp n, const npy_intp m, const npy_intp s2,
                const npy_intp s1) {
    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = 0; j < m; j++) {
            dst[i * m + j] = *(slice_ptr + (i * s2 / sizeof(T)) + (j * s1 / sizeof(T)));
        }
    }
}


/*
 * Copied straight from scipy.
 *
 * Used in conjunction with `copy_slice` for the double copy strategy.
 */
template <typename T>
inline void swap_cf(T *src, T *dst, const Py_ssize_t r, const Py_ssize_t c, const Py_ssize_t n) {
    Py_ssize_t i, j, ith_row, r2, c2;
    T *bb = dst;
    T *aa = src;
    if ((r < 16) && (c < 16)) {
        for (j = 0; j < c; j++) {
            ith_row = 0;
            for (i = 0; i < r; i++) {
                bb[ith_row] = aa[i];
                ith_row += n;
            }
            aa += n;
            bb += 1;
        }
    } else {
        // If tall
        if (r > c) {
            r2 = r / 2;
            swap_cf(src, dst, r2, c, n);
            swap_cf(src + r2, dst + (r2)*n, r - r2, c, n);
        } else { // Nope
            c2 = c / 2;
            swap_cf(src, dst, r, c2, n);
            swap_cf(src + (c2)*n, dst + c2, r, c - c2, n);
        }
    }
}


/*
 * New implementation of the `copy-and-transpose` operation proposed in
 * https://github.com/scipy/scipy/issues/24340#issuecomment-3744412228
 *
 * Added taking into account the strides of the input array.
 *
 * New strategy proposal that uses partitioning and a single copy strategy.
 */
template <typename T>
void copy_transposed_v2(T *dst, const T *src, const npy_intp n, npy_intp s2, npy_intp s1) {
    // For larger matrices (n >= 16), the array is partitioned as:
    // [       |   ]
    // [   A   |   ]
    // [       | C ]
    // [-------|   ]
    // [   B   |   ]
    //
    // where A is the largest (k*8)x(k*8) block, for some k <= n/8
    // and if exist, B is the bottom edge, C is the right pane.
    s1 = s1 / sizeof(T);
    s2 = s2 / sizeof(T);

    const int n_max = n & ~7; // Round down to multiple of 8

    // Main 8x8 blocks: reverse pattern (sequential writes, strided reads)
    for (int rb = 0; rb < n_max; rb += 8) {
        for (int cb = 0; cb < n_max; cb += 8) {
            const T *src_origin = src + cb * s2 + rb * s1;
            T *dst_origin = dst + rb * n + cb;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    dst_origin[i * n + j] = src_origin[j * s2 + i * s1];
                }
            }
        }
    }

    // Handle B and C blocks if present
    if (n_max < n) {
        // Bottom edge (remainder rows, cols of A)
        const int n_remain = n - n_max;
        for (int cb = 0; cb < n_max; cb += 8) {
            const T *src_origin = src + cb * s2 + n_max * s1;
            T *dst_origin = dst + n_max * n + cb;
            for (int i = 0; i < n_remain; i++) {
                for (int j = 0; j < 8; j++) {
                    dst_origin[i * n + j] = src_origin[j * s2 + i * s1];
                }
            }
        }

        // Right edge + corner (remainder cols, all rows)
        // Merged to maintain sequential writes pattern
        for (int i = 0; i < n; i++) {
            const T *src_row = src + n_max * s2 + i * s1;
            T *dst_row = dst + i * n + n_max;
            for (int j = 0; j < n_remain; j++) {
                dst_row[j] = src_row[j * s2];
            }
        }
    }
}
