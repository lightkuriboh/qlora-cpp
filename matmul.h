// matmul.h

#ifndef QLORA_CPP_MATMUL_H
#define QLORA_CPP_MATMUL_H

#include <immintrin.h>
#include <stdexcept>

#include "matrix.h"
#include "simd_kernel.h"

namespace qlora::ops {
  template <typename T>
  static void MatMul(const ::qlora::data_structure::Matrix<T>& a_matrix, bool matrix_a_transposed,
                     const ::qlora::data_structure::Matrix<T>& b_matrix, bool matrix_b_transposed,
                     ::qlora::data_structure::Matrix<T>& c_matrix,
                     T alpha = 1.0, T beta = 0.0) {
    const size_t M = matrix_a_transposed ? a_matrix.num_cols() : a_matrix.num_rows();
    const size_t K = matrix_a_transposed ? a_matrix.num_rows() : a_matrix.num_cols();
    const size_t N = matrix_b_transposed ? b_matrix.num_rows() : b_matrix.num_cols();

    if (K != (matrix_b_transposed ? b_matrix.num_cols() : b_matrix.num_rows())) {
      throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");
    }

    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        // We only use SIMD if both A and B are being accessed row-major in the inner loop
        // This happens if (not A_transposed) AND (B_transposed)
        if (!matrix_a_transposed && matrix_b_transposed) {
          __m256 v_acc = _mm256_setzero_ps();
          size_t k = 0;

          for (; k + 8 <= K; k += 8) {
            __m256 va = _mm256_load_ps(&a_matrix[i, k]);
            __m256 vb = _mm256_load_ps(&b_matrix[j, k]);
            v_acc = _mm256_fmadd_ps(va, vb, v_acc);
          }

          T sum = ::qlora::kernels::HorizontalSumAVX2(v_acc);

          for (; k < K; ++k) {
            sum += a_matrix[i, k] * b_matrix[j, k];
          }

          c_matrix[i, j] = alpha * sum + (beta == 0 ? 0 : beta * c_matrix[i, j]);
        }
        else {
          // Fallback for other transpose combinations (Scalar for now)
          T sum = 0;
          for (size_t k = 0; k < K; ++k) {
            T a_val = matrix_a_transposed ? a_matrix[k, i] : a_matrix[i, k];
            T b_val = matrix_b_transposed ? b_matrix[j, k] : b_matrix[k, j];
            sum += a_val * b_val;
          }
          c_matrix[i, j] = alpha * sum + (beta == 0 ? 0 : beta * c_matrix[i, j]);
        }
      }
    }
  }
}

#endif //QLORA_CPP_MATMUL_H
