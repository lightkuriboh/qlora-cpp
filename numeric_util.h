// numeric_util.h

#ifndef QLORA_NUMERIC_UTIL_H_
#define QLORA_NUMERIC_UTIL_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "nf4_constants.h"


namespace qlora::numeric_utility {

template <typename T>
void FillGaussianVector(std::span<T> vec,
                        std::mt19937& generator,
                        T mean = 0.0,
                        T stddev = 1.0) {
  std::normal_distribution<T> distribution(mean, stddev);
  std::generate(vec.begin(), vec.end(),
                [&]() { return distribution(generator); });
}

// Generate a vector of specified size filled with random values drawn from a Gaussian distribution.
template <typename T>
std::vector<T> GenerateGaussianVector(size_t target_size,
                                      std::mt19937& generator,
                                      T mean = 0.0,
                                      T stddev = 1.0) {
  std::vector<T> vec(target_size);
  FillGaussianVector<T>(vec, generator, mean, stddev);
  return vec;
}

// Calculate the mean squared error between the original and dequantized vectors.
template <typename T>
double CalculateMeanSquaredError(const std::vector<T>& original,
                                const std::vector<T>& dequantized) {
  if (original.size() != dequantized.size()) {
    throw std::invalid_argument("Vectors must be of the same size.");
  }

  double mse = 0.0;
  for (size_t i = 0; i < original.size(); ++i) {
    double diff = static_cast<double>(original[i]) -
                  static_cast<double>(dequantized[i]);
    mse += diff * diff;
  }
  return mse / static_cast<double>(original.size());
}

// Pack a 4-bit value into the appropriate position in the target vector based on the target index.
inline void PackNibble(std::vector<std::uint8_t>& target_vector, size_t target_index,
                      std::uint8_t value) {
  const std::size_t actual_index = target_index / 2;
  const std::uint8_t nibble = value & 0x0F;

  if (target_index % 2 == 0) {
    target_vector[actual_index] = (target_vector[actual_index] & 0x0F) | (nibble << 4);
  } else {
    target_vector[actual_index] = (target_vector[actual_index] & 0xF0) | nibble;
  }
}

// Extract the high nibble (4 bits) from a byte.
inline std::uint8_t GetHighNibble(std::uint8_t byte) {
  return (byte >> 4) & 0x0F;
}

// Extract the low nibble (4 bits) from a byte.
inline std::uint8_t GetLowNibble(std::uint8_t byte) {
  return byte & 0x0F;
}

// Extract the high nibble (4 bits) from a byte in the source vector at the specified index.
inline std::uint8_t GetHighNibble(const std::vector<std::uint8_t>& source_vector, size_t target_index) {
  size_t actual_index = target_index / 2;
  return GetHighNibble(source_vector[actual_index]);
}

// Extract the low nibble (4 bits) from a byte in the source vector at the specified index.
inline std::uint8_t GetLowNibble(const std::vector<std::uint8_t>& source_vector, size_t target_index) {
  size_t actual_index = target_index / 2;
  return GetLowNibble(source_vector[actual_index]);
}

// Returns the high and low nibbles as a pair from the source vector at [target_index].
inline std::pair<std::uint8_t, std::uint8_t> UnpackNibbleByte(
    const std::vector<std::uint8_t>& source_vector, size_t target_index) {
  size_t actual_index = target_index / 2;
  return {GetHighNibble(source_vector[actual_index]), GetLowNibble(source_vector[actual_index])};
}


// Calculate the mean of the data and center the data by subtracting the mean from each element.
template <typename T>
T MeanCentering(std::vector<T>& data) {
  if (data.empty()) {
    throw std::invalid_argument("Data vector cannot be empty for mean centering.");
  }

  const T sum = std::accumulate(data.begin(), data.end(), T(0));
  const T mean = sum / static_cast<T>(data.size());
  for (auto& value : data) {
    value -= mean;
  }

  return mean;
}

// Get the absolute maximum value from a specified range [start_index, end_index) in the data vector.
template <typename T>
T GetAbsMax(std::span<const T> data) {
  T abs_max = 0.0f;
  for (const T& value : data) {
    abs_max = std::max(abs_max, std::abs(value));
  }
  return abs_max;
}

}  // namespace qlora::numeric_utility


#endif  // QLORA_NUMERIC_UTIL_H_

