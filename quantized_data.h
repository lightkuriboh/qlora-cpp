// quantized_data.h

#ifndef QLORA_QUANTIZED_DATA_H_
#define QLORA_QUANTIZED_DATA_H_

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "numeric_util.h"


namespace qlora::data_structure
{
    // A class to represent quantized data.
    // It uses 8-bit values to pack two 4-bit values (nibbles).
    template <typename T>
    class QuantizedData {
      public:
        QuantizedData(): block_size_(0), original_data_size_(0), num_blocks_(0) {};

        QuantizedData(std::size_t block_size, std::size_t quantized_data_size, std::size_t quantize_constant_size)
            : block_size_(block_size),
              original_data_size_(quantized_data_size),
              num_blocks_(quantize_constant_size),
              weight_nf4_centroid_indices_((quantized_data_size + 1) / 2),
              quantize_constants_(quantize_constant_size) {}

        std::size_t block_size() const { return block_size_; }
        std::size_t original_data_size() const { return original_data_size_; }
        std::size_t num_blocks() const { return num_blocks_; }

        // Assign 4-bit quantized_value to a specific [target_index].
        void AssignQuantizedValue(size_t target_index, std::uint8_t value) {
            if (target_index >= original_data_size_) {
                throw std::out_of_range("Index out of range for quantized values.");
            }
            ::qlora::numeric_utility::PackNibble(weight_nf4_centroid_indices_, target_index, value);
        }

        // Get the quantized value at a specific [target_index].
        std::uint8_t GetNf4CentroidIndex(size_t target_index) const {
            if (target_index >= weight_nf4_centroid_indices_.size() * 2) {
                throw std::out_of_range("Index out of range for quantized values.");
            }
            size_t actual_index = target_index / 2;
            if (target_index % 2 == 0) {
                return (weight_nf4_centroid_indices_[actual_index] >> 4) & 0x0F;
            } else {
                return weight_nf4_centroid_indices_[actual_index] & 0x0F;
            }
        }

        // Get both high and low nibbles as a pair for a specific [target_index].
        std::pair<std::uint8_t, std::uint8_t> GetNf4CentroidIndicesPair(size_t target_index) const {
            if (target_index >= weight_nf4_centroid_indices_.size() * 2) {
                throw std::out_of_range("Index out of range for quantized values.");
            }
            return ::qlora::numeric_utility::UnpackNibbleByte(weight_nf4_centroid_indices_, target_index);
        }

        // Set the quantize constant at a specific [target_index].
        void SetQuantizeConstant(size_t target_index, T value) {
            if (target_index >= quantize_constants_.size()) {
                throw std::out_of_range("Index out of range for quantize constants.");
            }
            quantize_constants_[target_index] = value;
        }

        // Get the quantize constant at a specific [target_index].
        T GetQuantizeConstant(size_t target_index) const {
            if (target_index >= quantize_constants_.size()) {
                throw std::out_of_range("Index out of range for quantize constants.");
            }
            return quantize_constants_[target_index];
        }

      private:
        std::size_t block_size_;
        std::size_t original_data_size_;
        std::size_t num_blocks_;

        std::vector<std::uint8_t> weight_nf4_centroid_indices_;
        std::vector<T> quantize_constants_;
    };
}  // namespace qlora::data_structure

#endif  // QLORA_QUANTIZED_DATA_H_
