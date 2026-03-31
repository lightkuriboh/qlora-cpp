#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "nf4_constants.h"
#include "quantized_data.h"


namespace qlora::numeric_utility {
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
}  // namespace qlora::numeric_utility


namespace qlora::core {
    template <typename T>
    ::qlora::data_structure::QuantizedData<T> BlockWiseNf4Quantization(const std::vector<T>& input,
                                                                       const size_t block_size) {
        if (input.empty()) return {};

        auto num_blocks = (input.size() + block_size - 1) / block_size;
        ::qlora::data_structure::QuantizedData<T> quantized_data(block_size, input.size(), num_blocks);

        for (size_t block = 0; block < num_blocks; ++block) {
            const size_t block_start = block * block_size;
            const size_t block_end = std::min(block_start + block_size, input.size());

            T abs_max = 0;
            for (size_t i = block_start; i < block_end; ++i) {
                abs_max = std::max(abs_max, std::abs(input[i]));
            }
            if (abs_max == 0) continue;

            quantized_data.SetQuantizeConstant(block, abs_max);

            for (size_t i = block_start; i < block_end; ++i) {
                const auto normalized_scalar = input[i] / abs_max;
                const auto& centroids = ::qlora::nf4_constants::kNf4Centroids;
                size_t closest_centroid_index = 0;

                const auto it = std::lower_bound(centroids.begin(), centroids.end(), normalized_scalar);
                if (it == centroids.end()) {
                    closest_centroid_index = centroids.size() - 1;
                } else if (it == centroids.begin()) {
                    closest_centroid_index = 0;
                } else {
                    const auto prev_centroid_iterator = std::prev(it);
                    closest_centroid_index =
                        std::abs(*it - normalized_scalar) < std::abs(*prev_centroid_iterator - normalized_scalar)
                            ? std::distance(centroids.begin(), it)
                            : std::distance(centroids.begin(), prev_centroid_iterator);
                }
                
                quantized_data.AssignQuantizedValue(i, static_cast<std::uint8_t>(closest_centroid_index));
            }

        }
        return quantized_data;
    }

    template<typename T>
    std::vector<T> Dequantize(const data_structure::QuantizedData<T>& quantized_data) {
        std::vector<T> dequantized_values(quantized_data.original_data_size());
        for (size_t i = 0; i < quantized_data.original_data_size(); ++i) {
            const std::uint8_t quantized_index = quantized_data.GetQuantizedValue(i);
            const float centroid_value = ::qlora::nf4_constants::kNf4Centroids[quantized_index];
            const T quantize_constant = quantized_data.GetQuantizeConstant(i / quantized_data.block_size());

            dequantized_values[i] = static_cast<T>(quantize_constant * centroid_value);
        }
        return dequantized_values;
    }
}  // namespace qlora::core


int main(int argc, char* argv[]) {

    size_t block_size = 64;
    size_t vector_size = 1024;
    bool is_verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg.find("--block_size=") == 0) {
            block_size = std::stoul(std::string(arg.substr(13)));
        } else if (arg.find("--vector_size=") == 0) {
            vector_size = std::stoul(std::string(arg.substr(14)));
        } else if (arg == "--verbose") {
            is_verbose = true;
        }
    }
    if (is_verbose) {
        std::cout << "Block Size: " << block_size << "\n";
        std::cout << "Vector Size: " << vector_size << "\n";
    }

    std::cout << (is_verbose ? "Verbose mode ON\n" : "Verbose mode OFF\n");

    std::random_device rd;
    std::mt19937 gen(rd());

    // Use PascalCase for the functions we refactored earlier.
    const auto weights_vector = ::qlora::numeric_utility::GenerateGaussianVector<float>(
        vector_size, gen, 0.0f, 1.0f);

    if (is_verbose) {
    std::cout << "Original Weights Sample (first 5):\n";
    for (size_t i = 0; i < std::min<size_t>(5, weights_vector.size()); ++i) {
        std::cout << weights_vector[i] << "\n";
    }
    }

    const auto quantized_data =
        ::qlora::core::BlockWiseNf4Quantization<float>(weights_vector, block_size);

    const auto dequantized_values =
        ::qlora::core::Dequantize<float>(quantized_data);

    const double mse = ::qlora::numeric_utility::CalculateMeanSquaredError(
        weights_vector, dequantized_values);

    std::cout << "Mean Squared Error (MSE): " << mse << "\n";

    return 0;
}
