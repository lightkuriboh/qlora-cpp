// simd_kernel.h

#ifndef QLORA_CPP_SIMD_KERNEL_H
#define QLORA_CPP_SIMD_KERNEL_H

#include <immintrin.h>

#include "nf4_constants.h"

namespace qlora::kernels {

/**
 * @brief Dequantizes 8 weights (4 bytes) into a 256-bit SIMD register.
 * @param packed_bytes Pointer to 4 contiguous bytes containing 8 NF4 indices.
 * @param quantization_scale The quantization scale for this block.
 * @param quantization_mean The quantization mean for this block.
 */
inline __m256 Dequantize8WeightsAVX2(const uint8_t* packed_bytes, float quantization_scale, float quantization_mean) {
  // Load 4 bytes into a 128-bit register (only first 32 bits are used)
  const __m128i raw = _mm_cvtsi32_si128(*(reinterpret_cast<const int*>(packed_bytes)));
  // Extract Low and High nibbles
  const __m128i low_indices = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
  const __m128i high_indices = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
  // Interleave: [L0, H0, L1, H1, L2, H2, L3, H3]
  const __m128i indices_8bit = _mm_unpacklo_epi8(low_indices, high_indices);
  // Widen 8-bit to 32-bit for the Gather instruction
  __m256i indices_32bit = _mm256_cvtepu8_epi32(indices_8bit);
  // Gather floats from the Lookup Table
  const auto weights = _mm256_i32gather_ps(::qlora::nf4_constants::kNf4Centroids.data(), indices_32bit, 4);
  // Fused Multiply-Add: (weights * scale) + mean
  const __m256 v_scale = _mm256_set1_ps(quantization_scale);
  const __m256 v_mean = _mm256_set1_ps(quantization_mean);

  return _mm256_add_ps(_mm256_mul_ps(weights, v_scale), v_mean);
}

  inline float HorizontalSumAVX2(__m256 register_value) {
  // Fold the 256-bit register in half (extract top 128 and add to bottom 128)
  const __m128 low_register_value = _mm256_castps256_ps128(register_value);
  const __m128 high_register_value = _mm256_extractf128_ps(register_value, 1);
  const __m128 v128 = _mm_add_ps(low_register_value, high_register_value);

  // Fold the 128-bit register in half
  __m128 shuf = _mm_movehdup_ps(v128);  // Broadcast elements 1 and 3
  __m128 sums = _mm_add_ps(v128, shuf);

  shuf = _mm_movehl_ps(shuf, sums);  // High half to low half
  sums = _mm_add_ss(sums, shuf);

  return _mm_cvtss_f32(sums);
}

} // namespace qlora::kernels

#endif //QLORA_CPP_SIMD_KERNEL_H
