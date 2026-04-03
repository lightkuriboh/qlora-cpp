
#include "qlora.cc"


int main() {
  const size_t vector_size = 1 << 16;
  const size_t block_size = 64;
  const size_t quantize_constants_blocks_size = 256;
  const bool is_verbose = false;

  if (is_verbose) {
    std::cout << "Block Size: " << block_size << "\n";
    std::cout << "Vector Size: " << vector_size << "\n";
  }

  std::cout << (is_verbose ? "Verbose mode ON\n" : "Verbose mode OFF\n");

  std::random_device rd;
  std::mt19937 gen(rd());

  const auto weights_vector =
      ::qlora::numeric_utility::GenerateGaussianVector<float>(
          vector_size, gen, 0.0f, 1.0f);

  if (is_verbose) {
    std::cout << "Original Weights Sample (first 5):\n";
    for (size_t i = 0; i < std::min<size_t>(5, weights_vector.size()); ++i) {
      std::cout << weights_vector[i] << "\n";
    }
  }

  const auto quantized_data =
    ::qlora::core::BlockWiseNf4Quantization<float>(
        weights_vector, block_size, quantize_constants_blocks_size);

  if (is_verbose) {
    std::cout << "Number of Blocks (aka number of quantize constants indices): "
              << quantized_data.num_blocks() << "\n";
    std::cout << "Number of Quantize Constants: "
              << quantized_data.num_blocks_quantized_constants() << "\n";
    std::cout << "Quantize Constant Mean: "
              << quantized_data.quantize_constant_mean() << "\n";

    std::cout << "Sample Quantize Constant (first 5):\n";
    for (size_t i = 0;
         i < std::min<size_t>(
            5, quantized_data.num_blocks_quantized_constants());
         ++i) {
      std::cout << quantized_data.GetDoubleQuantizeConstant(i) << "\n";
    }
  }

  const auto dequantized_values =
      ::qlora::core::Dequantize<float>(quantized_data);

  const double mse =
      ::qlora::numeric_utility::CalculateMeanSquaredError(
          weights_vector, dequantized_values);

  std::cout << "Mean Squared Error (MSE): " << mse << "\n";

  const double compression_ratio =
      ::qlora::core::CalculateCompressionRatio(quantized_data);
  std::cout << "Compression Ratio: " << compression_ratio << "\n";

  return 0;
}
