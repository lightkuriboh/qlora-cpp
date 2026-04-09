#include <gtest/gtest.h>
#include <random>
#include "lora_linear_layer.h"
#include "matrix.h"
#include "quantization.h"

using namespace qlora::quantization;
using namespace qlora::lora;
using namespace qlora::data_structure;

class LoRALinearLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    in_dim = 64;
    out_dim = 32;
    rank = 8;
    alpha = 16.0f;

    std::random_device rd;
    std::mt19937 generator(rd());

    std::vector<float> raw_weights =
        qlora::numeric_utility::GenerateGaussianVector<float>(in_dim * out_dim, generator);
    auto quantized_base = BlockWiseNf4Quantization(raw_weights, 64, 8);
    layer = std::make_unique<LoRALinearLayer<float>>(in_dim, out_dim, rank, alpha, quantized_base, generator);
  }

  size_t in_dim, out_dim, rank;
  float alpha;
  std::unique_ptr<LoRALinearLayer<float>> layer;
};

TEST_F(LoRALinearLayerTest, OutputDimensionsAreCorrect) {
  constexpr size_t batch_size = 4;
  std::mt19937 generator(42);
  Matrix<float> input(batch_size, in_dim);
  input.FillGaussianMatrix(generator);

  const Matrix<float> output = layer->Forward(input);

  EXPECT_EQ(output.num_rows(), batch_size);
  EXPECT_EQ(output.num_cols(), out_dim);
}
