#include "lora_linear_layer.h"

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "qlora.h"

using namespace qlora::lora;
using namespace qlora::data_structure;

class LoRALinearLayerTest : public ::testing::Test {
 protected:
  static constexpr size_t kInDim = 8;
  static constexpr size_t kOutDim = 4;
  static constexpr size_t kRank = 2;
  static constexpr float kTargetScale = 1.0f;
  static constexpr float alpha = kTargetScale * static_cast<float>(kRank);

  static float CalculateMSE(const Matrix<float>& pred, const Matrix<float>& target) {
    float mse = 0;
    for (size_t i = 0; i < pred.num_rows(); ++i) {
      for (size_t j = 0; j < pred.num_cols(); ++j) {
        const float diff = pred[i, j] - target[i, j];
        mse += diff * diff;
      }
    }
    return mse / static_cast<float>(pred.num_rows() * pred.num_cols());
  }
};

TEST_F(LoRALinearLayerTest, StepReducesLoss) {
  std::mt19937 gen(21);
  const auto weights = qlora::numeric_utility::GenerateGaussianVector<float>(kInDim * kOutDim, gen);
  const auto quantized_base = ::qlora::core::BlockWiseNf4Quantization(weights, 64, 256);

  LoRALinearLayer layer(kInDim, kOutDim, kRank, alpha, quantized_base, gen);

  Matrix<float> input(1, kInDim);
  input.FillGaussianMatrix(gen);
  Matrix<float> target(1, kOutDim);
  target.FillGaussianMatrix(gen);

  auto pred = layer.Forward(input);
  const float loss_before = CalculateMSE(pred, target);

  // Calculate Gradient of MSE: 2 * (pred - target) / N
  Matrix<float> grad_output(1, kOutDim);
  for(size_t i = 0; i < kOutDim; ++i) {
    grad_output[0, i] = 2.0f * (pred[0, i] - target[0, i]) / static_cast<float>(kOutDim);
  }

  layer.Backward(grad_output);
  layer.Step(0.01f);

  const float loss_after = CalculateMSE(layer.Forward(input), target);

  EXPECT_LT(loss_after, loss_before) << "Optimization step failed to reduce loss.";
}
