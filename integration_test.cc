#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "qlora.h"

namespace qlora::core {

class QLoraIntegrationTest : public ::testing::Test {
protected:
  const size_t kVectorSize = 1 << 16;
  const size_t kBlockSize = 64;
  const size_t kMetaBlockSize = 256;
  const int kIterations = 10;
};

TEST_F(QLoraIntegrationTest, AccuracyAndCompressionCheck) {
  std::random_device rd;
  std::mt19937 gen(rd());

  double total_mse = 0.0;
  double total_compression_ratio = 0.0;

  for (int i = 0; i < kIterations; ++i) {
    const auto weights = numeric_utility::GenerateGaussianVector<float>(
        kVectorSize, gen, 0.0f, 1.0f);

    const auto quantized = BlockWiseNf4Quantization<float>(
        weights, kBlockSize, kMetaBlockSize);

    const auto dequantized = Dequantize<float>(quantized);

    total_mse += numeric_utility::CalculateMeanSquaredError(weights, dequantized);
    total_compression_ratio += CalculateCompressionRatio(quantized);
  }

  const double avg_mse = total_mse / kIterations;
  const double avg_compression = total_compression_ratio / kIterations;

  EXPECT_LT(avg_mse, 0.0095)
      << "Average MSE " << avg_mse << " exceeded the threshold of 0.0095";

  EXPECT_GE(avg_compression, 7.8)
      << "Compression ratio " << avg_compression << " fell below 7.8x";
}
}  // namespace qlora::core
