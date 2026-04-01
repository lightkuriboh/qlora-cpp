// numeric_util.h

#ifndef QLORA_NUMERIC_UTIL_H_
#define QLORA_NUMERIC_UTIL_H_

#include <cmath>
#include <chrono>
#include <numeric>
#include <random>
#include <vector>

#include "nf4_constants.h"


namespace qlora::numeric_utility
{
  // Generate a vector of specified size filled with random values drawn from a Gaussian distribution with given mean and standard deviation.
  template <typename T> std::vector<T> GenerateGaussianVector(size_t target_size,
                                                              std::mt19937& generator,
                                                              T mean = 0.0,
                                                              T stddev = 1.0) {
    std::normal_distribution<T> distribution(mean, stddev);
    std::vector<T> vec(target_size);
    std::generate(vec.begin(), vec.end(), [&]() { return distribution(generator); });
    return vec;
  }

  // Calculate the mean squared error between the original and dequantized vectors.
  template <typename T>
  double CalculateMeanSquaredError(const std::vector<T>& original,
                                   const std::vector<T>& dequantized) {
    if (original.size() != dequantized.size()) {
      throw std::invalid_argument("Vectors must be of the same size to calculate MSE.");
    }

    double mse = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
      double diff = static_cast<double>(original[i]) - static_cast<double>(dequantized[i]);
      mse += diff * diff;
    }
    return mse / static_cast<double>(original.size());
  }

  // Pack a 4-bit value into the appropriate position in the target vector based on the target index.
  void PackNibble(std::vector<std::uint8_t>& target_vector, size_t target_index, std::uint8_t value) {
    const std::size_t actual_index = target_index / 2;
    const std::uint8_t nibble = value & 0x0F;

    if (target_index % 2 == 0) {
      target_vector[actual_index] = (target_vector[actual_index] & 0x0F) | (nibble << 4);
    } else {
      target_vector[actual_index] = (target_vector[actual_index] & 0xF0) | nibble;
    }
  }

  // Extract the high nibble (4 bits) from a byte.
  std::uint8_t GetHighNibble(std::uint8_t byte) {
    return (byte >> 4) & 0x0F;
  }

  // Extract the low nibble (4 bits) from a byte.
  std::uint8_t GetLowNibble(std::uint8_t byte) {
    return byte & 0x0F;
  }

  // Extract the high nibble (4 bits) from a byte in the source vector at the specified index.
  std::uint8_t GetHighNibble(const std::vector<std::uint8_t>& source_vector, size_t target_index) {
    size_t actual_index = target_index / 2;
    return GetHighNibble(source_vector[actual_index]);
  }

  // Extract the low nibble (4 bits) from a byte in the source vector at the specified index.
  std::uint8_t GetLowNibble(const std::vector<std::uint8_t>& source_vector, size_t target_index) {
    size_t actual_index = target_index / 2;
    return GetLowNibble(source_vector[actual_index]);
  }

  // Unpack a byte from the source vector at the appropriate position based on the target index and return the high and low nibbles as a pair.
  std::pair<std::uint8_t, std::uint8_t> UnpackNibbleByte(const std::vector<std::uint8_t>& source_vector, size_t target_index) {
    size_t actual_index = target_index / 2;
    std::uint8_t high_nibble = GetHighNibble(source_vector[actual_index]);
    std::uint8_t low_nibble = GetLowNibble(source_vector[actual_index]);
    return {high_nibble, low_nibble};
  }
  

  // Calculate the mean of the data and center the data by subtracting the mean from each element. Returns the calculated mean.
  template <typename T>
  T MeanCentering(std::vector<T>& data) {
    if (data.empty()) {
      throw std::invalid_argument("Data vector cannot be empty for mean centering.");
    }

    T mean = std::accumulate(data.begin(), data.end(), T(0)) / static_cast<T>(data.size());
    for (auto& value : data) {
      value -= mean;
    }

    return mean;
  }

  // Get the absolute maximum value from a specified range [start_index, end_index) in the data vector.
  template <typename T>
  T GetAbsMax(const std::vector<T>& data, size_t start_index, size_t end_index) {
    T abs_max = 0.0f;
    for (size_t i = start_index; i < end_index; ++i) {
      abs_max = std::max(abs_max, std::abs(data[i]));
    }
    return abs_max;
  }
}  // namespace qlora::numeric_utility


#endif  // QLORA_NUMERIC_UTIL_H_

