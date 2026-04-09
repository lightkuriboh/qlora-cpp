// quantization.h

#ifndef QLORA_QUANTIZATION_H_
#define QLORA_QUANTIZATION_H_

#include <algorithm>
#include <iostream>
#include <span>
#include <vector>

#include "nf4_constants.h"
#include "numeric_util.h"
#include "quantized_data.h"


namespace qlora::quantization {

template <typename T>
inline double CalculateCompressionRatio(
      const ::qlora::data_structure::QuantizedData<T>& quantized_data) {
  const size_t original_bytes = quantized_data.original_data_size() * sizeof(T);
  const size_t packed_indices_bytes =
      (quantized_data.original_data_size() + 1) / 2 * sizeof(std::uint8_t);
  const size_t double_quantize_constant_indices_bytes =
      (quantized_data.num_blocks() + 1) / 2 * sizeof(std::uint8_t);
  const size_t double_quantize_constants_bytes =
      quantized_data.num_blocks_quantized_constants() * sizeof(T);
  const size_t total_quantized_bytes =
      packed_indices_bytes +
      double_quantize_constant_indices_bytes +
      double_quantize_constants_bytes +
      sizeof(T);  // + sizeof(T) for quantize constant mean

  return static_cast<double>(original_bytes) /
         static_cast<double>(total_quantized_bytes);
}

template <typename T>
inline ::qlora::data_structure::QuantizedData<T> BlockWiseNf4Quantization(
    const std::vector<T>& input, const size_t block_size,
    const size_t quantize_constants_blocks_size) {
  if (input.empty()) return {};

  ::qlora::data_structure::QuantizedData<T> quantized_data(
      input.size(), block_size, quantize_constants_blocks_size);

  const size_t num_blocks = (input.size() + block_size - 1) / block_size;
  const size_t num_blocks_quantized_constants =
      (num_blocks + quantize_constants_blocks_size - 1) /
      quantize_constants_blocks_size;
  std::vector<T> quantize_constants(num_blocks);

  for (size_t block = 0; block < num_blocks; ++block) {
    const size_t block_start = block * block_size;
    const size_t block_end = std::min(block_start + block_size, input.size());

    const T abs_max =
        ::qlora::numeric_utility::GetAbsMax(
            std::span<const T>{input}
                .subspan(block_start, block_end - block_start + 1));

    quantize_constants[block] = abs_max;
    for (size_t i = block_start; i < block_end; ++i) {
      const std::uint8_t closest_centroid_index = static_cast<std::uint8_t>(
          ::qlora::nf4_constants::GetClosestCentroidIndex<T>(input[i], abs_max));
      quantized_data.AssignQuantizedValue(i, closest_centroid_index);
    }
  }

  const T quantize_constant_mean =
      ::qlora::numeric_utility::MeanCentering(quantize_constants);
  quantized_data.SetQuantizeConstantMean(quantize_constant_mean);

  for (size_t double_quantize_block = 0;
       double_quantize_block < num_blocks_quantized_constants;
       ++double_quantize_block) {
    const size_t block_start = double_quantize_block * quantize_constants_blocks_size;
    const size_t block_end =
        std::min(block_start + quantize_constants_blocks_size, num_blocks);
    const T abs_max =
        ::qlora::numeric_utility::GetAbsMax(
            std::span<const T>{quantize_constants}
                .subspan(block_start, block_end - block_start + 1));
    quantized_data.SetDoubleQuantizeConstant(double_quantize_block, abs_max);

    for (size_t i = block_start; i < block_end; ++i) {
      std::uint8_t closest_centroid_index = static_cast<std::uint8_t>(
          ::qlora::nf4_constants::GetClosestCentroidIndex<T>(
              quantize_constants[i], abs_max));
      quantized_data.SetQuantizeConstantNf4CentroidIndex(i, closest_centroid_index);
    }
  }
  return quantized_data;
}

template<typename T>
inline std::vector<T> Dequantize(const data_structure::QuantizedData<T>& quantized_data) {
  std::vector<T> dequantized_values(quantized_data.original_data_size());
  for (size_t i = 0; i < quantized_data.original_data_size(); i += 2) {
    const auto [high_nibble_centroid_index, low_nibble_centroid_index] =
        quantized_data.GetNf4CentroidIndicesPair(i);

    size_t current_block_i = i / quantized_data.block_size();
    const T quantize_constant_i = quantized_data.GetDoubleQuantizeConstant(i) *
                                  ::qlora::nf4_constants::kNf4Centroids[
                                      quantized_data.GetQuantizeConstantNf4CentroidIndex(
                                          current_block_i)] +
                                  quantized_data.quantize_constant_mean();
    dequantized_values[i] =
        static_cast<T>(quantize_constant_i *
                       ::qlora::nf4_constants::kNf4Centroids[high_nibble_centroid_index]);
        
    if (i + 1 < quantized_data.original_data_size()) {
      const size_t current_block_i_plus_1 = (i + 1) / quantized_data.block_size();
      const T quantize_constant_i_plus_1 =
          current_block_i == current_block_i_plus_1
          ? quantize_constant_i
          : quantized_data.GetDoubleQuantizeConstant(i + 1) *
            ::qlora::nf4_constants::kNf4Centroids[
                quantized_data.GetQuantizeConstantNf4CentroidIndex(current_block_i_plus_1)] +
            quantized_data.quantize_constant_mean();
      dequantized_values[i + 1] =
          static_cast<T>(quantize_constant_i_plus_1 *
                         ::qlora::nf4_constants::kNf4Centroids[low_nibble_centroid_index]);
    }
  }
  return dequantized_values;
}

}  // namespace qlora::core

#endif  // QLORA_QUANTIZATION_H_
