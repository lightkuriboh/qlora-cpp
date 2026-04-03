// nf4_constants.h

#ifndef QLORA_NF4_CONSTANTS_H_
#define QLORA_NF4_CONSTANTS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>

namespace qlora::nf4_constants {

constexpr std::array<float, 16> kNf4Centroids = {
    -1.0f, -0.69619f, -0.52507f, -0.39492f, -0.28444f, -0.18477f, -0.09105f, 0.0f, 
    0.07958f, 0.16093f, 0.24611f, 0.33792f, 0.44033f, 0.55848f, 0.70151f, 1.0f};

inline std::uint8_t GetClosestCentroidIndex(float value) {
  const auto it = std::lower_bound(kNf4Centroids.begin(), kNf4Centroids.end(), value);

  if (it == kNf4Centroids.begin()) {
    return 0;
  }
  if (it == kNf4Centroids.end()) {
    return static_cast<std::uint8_t>(kNf4Centroids.size() - 1);
  }

  const auto prev_centroid_iterator = std::prev(it);
  const size_t closest_centroid_index =
      (std::abs(*it - value) < std::abs(*prev_centroid_iterator - value))
          ? static_cast<size_t>(std::distance(kNf4Centroids.begin(), it))
          : static_cast<size_t>(std::distance(kNf4Centroids.begin(),
                                              prev_centroid_iterator));
  return static_cast<std::uint8_t>(closest_centroid_index);
}

template<typename T>
inline std::uint8_t GetClosestCentroidIndex(T value, T abs_max) {
  if (abs_max == static_cast<T>(0)) {
    return 7;
  }

  const float normalized_value = static_cast<float>(value) / static_cast<float>(abs_max);
  return GetClosestCentroidIndex(normalized_value);
}

}  // namespace qlora::nf4_constants

#endif  // QLORA_NF4_CONSTANTS_H_
