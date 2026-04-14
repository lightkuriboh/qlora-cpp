// matmul.h

#ifndef QLORA_CPP_MATMUL_H
#define QLORA_CPP_MATMUL_H

#include <immintrin.h>
#include <omp.h>

#include <stdexcept>

#include "matrix.h"
#include "simd_kernel.h"

namespace qlora::ops {

/**
 * Performs matrix multiplication: C = alpha * (A * B) + beta * C.
 * Supports transposition of input matrices.
 */
template <typename T>
void MatMul(const data_structure::Matrix<T>& a, bool transpose_a,
            const data_structure::Matrix<T>& b, bool transpose_b,
            data_structure::Matrix<T>& c, T alpha = 1.0, T beta = 0.0) {
  const size_t m = transpose_a ? a.num_cols() : a.num_rows();
  const size_t k_dim = transpose_a ? a.num_rows() : a.num_cols();
  const size_t n = transpose_b ? b.num_rows() : b.num_cols();

  const size_t k_check = transpose_b ? b.num_cols() : b.num_rows();

  if (k_dim != k_check) {
    throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");
  }

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      T sum = 0;

      // Both inner loops then traverse contiguous memory.
      if constexpr (std::is_same_v<T, float>) {
        if (!transpose_a && !transpose_b) {
          __m256 v_acc = _mm256_setzero_ps();
          size_t k = 0;

          // Process 8 elements at a time using AVX2.
          for (; k + 8 <= k_dim; k += 8) {
            __m256 va = _mm256_load_ps(&a[i, k]);
            __m256 vb = _mm256_load_ps(&b[j, k]);
            v_acc = _mm256_fmadd_ps(va, vb, v_acc);
          }

          sum = kernels::HorizontalSumAVX2(v_acc);

          // Scalar tail handling.
          for (; k < k_dim; ++k) {
            sum += a[i, k] * b[j, k];
          }
        } else {
          // Standard scalar fallback.
          for (size_t k = 0; k < k_dim; ++k) {
            const T a_val = transpose_a ? a[k, i] : a[i, k];
            const T b_val = transpose_b ? b[j, k] : b[k, j];
            sum += a_val * b_val;
          }
        }
      } else {
        // Generic scalar fallback for non-float types.
        for (size_t k = 0; k < k_dim; ++k) {
          const T a_val = transpose_a ? a[k, i] : a[i, k];
          const T b_val = transpose_b ? b[j, k] : b[k, j];
          sum += a_val * b_val;
        }
      }

      // Apply Alpha and Beta scaling.
      if (beta == static_cast<T>(0)) {
        c[i, j] = alpha * sum;
      } else {
        c[i, j] = alpha * sum + beta * c[i, j];
      }
    }
  }
}

}  // namespace qlora::ops

#endif //QLORA_CPP_MATMUL_H
