// numeric_util.h

#ifndef QLORA_NUMERIC_UTIL_H_
#define QLORA_NUMERIC_UTIL_H_

#include <cmath>
#include <chrono>
#include <random>
#include <vector>

#include "nf4_constants.h"


namespace qlora::numeric_utility
{
  template <typename T> std::vector<T> GenerateGaussianVector(size_t target_size,
                                                              std::mt19937& generator,
                                                              T range_min = 0.0,
                                                              T range_max = 1.0) {
    std::normal_distribution<T> distribution(range_min, range_max);
    std::vector<T> vec(target_size);
    std::generate(vec.begin(), vec.end(), [&]() { return distribution(generator); });
    return vec;
  }

  template <typename T>
  double CalculateMeanSquaredError(const std::vector<T>& original,
                                   const std::vector<T>& dequantized) {
    if (original.size() != dequantized.size()) {
      throw std::invalid_argument("Vectors must be of the same size to calculate MSE.");
    }

    double mse = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
      mse += std::pow(static_cast<double>(original[i]) - static_cast<double>(dequantized[i]), 2);
    }
    return mse / static_cast<double>(original.size());
  }

  void PackNibble(std::vector<std::uint8_t>& target_vector, size_t target_index, std::uint8_t value) {
    const std::size_t actual_index = target_index / 2;
    const std::uint8_t nibble = value & 0x0F;

    if (target_index % 2 == 0) {
      target_vector[actual_index] = (target_vector[actual_index] & 0x0F) | (nibble << 4);
    } else {
      target_vector[actual_index] = (target_vector[actual_index] & 0xF0) | nibble;
    }
  }

  std::pair<std::uint8_t, std::uint8_t> UnpackNibbleByte(const std::vector<std::uint8_t>& source_vector, size_t target_index) {
    size_t actual_index = target_index / 2;
    std::uint8_t high_nibble = (source_vector[actual_index] >> 4) & 0x0F;
    std::uint8_t low_nibble = source_vector[actual_index] & 0x0F;
    return {high_nibble, low_nibble};
  }
}  // namespace qlora::numeric_utility


#endif  // QLORA_NUMERIC_UTIL_H_

