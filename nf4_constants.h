// nf4_constants.h

#ifndef QLORA_NF4_CONSTANTS_H_
#define QLORA_NF4_CONSTANTS_H_

#include <array>

namespace qlora::nf4_constants
{
    constexpr std::array<float, 16> kNf4Centroids = {
        -1.0f, -0.69619f, -0.52507f, -0.39492f, -0.28444f, -0.18477f, -0.09105f, 0.0f, 
        0.07958f, 0.16093f, 0.24611f, 0.33792f, 0.44033f, 0.55848f, 0.70151f, 1.0f};
}  // namespace qlora::nf4_constants

#endif  // QLORA_NF4_CONSTANTS_H_
