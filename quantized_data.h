// quantized_data.h

#ifndef QLORA_QUANTIZED_DATA_H_
#define QLORA_QUANTIZED_DATA_H_

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "numeric_util.h"


namespace qlora::data_structure {

// A class to represent quantized data.
// It uses 8-bit values to pack two 4-bit values (nibbles).
template <typename T>
class QuantizedData {
 public:
  QuantizedData()
      : original_data_size_(0),
        block_size_(0),
        num_blocks_(0),
        quantize_constants_blocks_size_(0),
        num_blocks_quantized_constants_(0),
        weight_nf4_centroid_indices_(0),
        quantize_constants_nf4_centroid_indices_(0),
        double_quantized_constants_(0) {}

  QuantizedData(std::size_t original_data_size, std::size_t block_size,
                std::size_t quantize_constants_blocks_size)
      : original_data_size_(original_data_size),
        block_size_(block_size),
        num_blocks_((original_data_size + block_size - 1) / block_size),
        quantize_constants_blocks_size_(quantize_constants_blocks_size) {

    num_blocks_quantized_constants_ =
        (num_blocks_ + quantize_constants_blocks_size - 1) /
        quantize_constants_blocks_size;

    weight_nf4_centroid_indices_.resize((original_data_size + 1) / 2);
    quantize_constants_nf4_centroid_indices_.resize((num_blocks_ + 1) / 2);
    double_quantized_constants_.resize(num_blocks_quantized_constants_);
  }


  std::size_t original_data_size() const { return original_data_size_; }
  std::size_t block_size() const { return block_size_; }
  std::size_t num_blocks() const { return num_blocks_; }
  std::size_t quantize_constants_blocks_size() const {
    return quantize_constants_blocks_size_;
  }
  std::size_t num_blocks_quantized_constants() const {
    return num_blocks_quantized_constants_;
  }
  T quantize_constant_mean() const { return quantize_constant_mean_; }

  // Set quantize constant mean for mean centering.
  void SetQuantizeConstantMean(T mean) { quantize_constant_mean_ = mean; }


  // Get both high and low nibbles as a pair for a specific [target_index] in the original data.
  std::pair<std::uint8_t, std::uint8_t> GetNf4CentroidIndicesPair(size_t target_index) const {
    if (target_index >= weight_nf4_centroid_indices_.size() * 2) {
      throw std::out_of_range("Index out of range for quantized values.");
    }
    return ::qlora::numeric_utility::UnpackNibbleByte(weight_nf4_centroid_indices_,
                                                      target_index);
  }

  // Assign 4-bit quantized_value to a specific [target_index].
  void AssignQuantizedValue(size_t target_index, std::uint8_t value) {
    if (target_index >= original_data_size_) {
      throw std::out_of_range("Index out of range for quantized values.");
    }
    ::qlora::numeric_utility::PackNibble(weight_nf4_centroid_indices_, target_index, value);
  }

  // Get both high and low nibbles as a pair for a specific [target_index] in the original data.
  std::uint8_t GetQuantizeConstantNf4CentroidIndex(size_t block_index) const {
    if (block_index >= num_blocks_) {
      throw std::out_of_range("Index out of range for quantize constants.");
    }
    const size_t target_index = block_index / 2;
    return (block_index % 2 == 0)
               ? ::qlora::numeric_utility::GetHighNibble(
                     quantize_constants_nf4_centroid_indices_[target_index])
               : ::qlora::numeric_utility::GetLowNibble(
                     quantize_constants_nf4_centroid_indices_[target_index]);
  }

  // Set the quantize constant NF4 centroid index at a specific [target_index] in the quantize constants centroids.
  void SetQuantizeConstantNf4CentroidIndex(size_t target_index, std::uint8_t value) {
    if (target_index >= quantize_constants_nf4_centroid_indices_.size() * 2) {
      throw std::out_of_range("Index out of range for quantize constants.");
    }
    ::qlora::numeric_utility::PackNibble(quantize_constants_nf4_centroid_indices_, target_index, value);
  }

  // Get the double quantize constant for a specific [target_index] in the original data.
  T GetDoubleQuantizeConstant(size_t original_data_index) const {
    const size_t target_index =
        original_data_index / block_size() / quantize_constants_blocks_size_;
    if (target_index >= double_quantized_constants_.size()) {
      throw std::out_of_range("Index out of range for quantize constants.");
    }
    return double_quantized_constants_[target_index];
  }

  // Set the doubled quantize constant at a specific [target_index] in the double quantized constants.
  void SetDoubleQuantizeConstant(size_t target_index, T value) {
    if (target_index >= double_quantized_constants_.size()) {
      throw std::out_of_range("Index out of range for quantize constants.");
    }
    double_quantized_constants_[target_index] = value;
  }

  struct DequantizationCursor {
    const QuantizedData& data;
    T current_scale;
    std::pair<uint8_t, uint8_t> current_nibbles;
    size_t last_block_index = -1;
    size_t last_processed_index = -1;

    T GetWeight(size_t weight_index) {
      if (const size_t block_index = weight_index / data.block_size(); block_index != last_block_index) {
        const uint8_t const_nf4_idx = data.GetQuantizeConstantNf4CentroidIndex(block_index);
        T doubled_quantize_constant = data.GetDoubleQuantizeConstant(weight_index);
        current_scale = (doubled_quantize_constant * nf4_constants::kNf4Centroids[const_nf4_idx]) +
                        data.quantize_constant_mean();
        last_block_index = block_index;
      }

      if (weight_index % 2 == 0) {
        current_nibbles = data.GetNf4CentroidIndicesPair(weight_index);
        last_processed_index = weight_index;
        return current_scale * nf4_constants::kNf4Centroids[current_nibbles.first];
      }
      if (weight_index != last_processed_index + 1) {
        current_nibbles = data.GetNf4CentroidIndicesPair(weight_index - 1);
      }
      last_processed_index = weight_index;
      return current_scale * nf4_constants::kNf4Centroids[current_nibbles.second];
    }
  };

  DequantizationCursor GetCursor() const { return {*this}; }

 private:
  std::size_t original_data_size_;
  std::size_t block_size_;
  std::size_t num_blocks_;
  std::size_t quantize_constants_blocks_size_;
  std::size_t num_blocks_quantized_constants_;
  T quantize_constant_mean_;

  std::vector<std::uint8_t> weight_nf4_centroid_indices_;
  std::vector<std::uint8_t> quantize_constants_nf4_centroid_indices_;
  std::vector<T> double_quantized_constants_;
};

}  // namespace qlora::data_structure

#endif  // QLORA_QUANTIZED_DATA_H_
