#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

constexpr std::array<float, 16> kNf4Centroids = {
    -1.0, -0.69619, -0.52507, -0.39492, -0.28444, -0.18477, -0.09105, 0.0,
    0.07958, 0.16093, 0.24611, 0.33792, 0.44033, 0.55848, 0.70151, 1.0};

template <typename T>
std::vector<T> GaussianVector(size_t target_size, std::mt19937& generator, T range_min = 0.0, T range_max = 1.0) {
    std::normal_distribution<T> distribution(range_min, range_max);
    std::vector<T> vec(target_size);
    std::generate(vec.begin(), vec.end(), [&]() {} { return distribution(generator); });
    return vec;
}

int main() {
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    const auto weightsVector = GaussianVector<float>(
        16,
        generator,
        0.0,
        1.0
    );
    for (const auto tensor: weightsVector) {
        std::cout << tensor << "\n";
    }
    return 0;
}
