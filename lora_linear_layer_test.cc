#include "lora_linear_layer.h"

#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

#include "quantization.h"

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
  std::mt19937 generator(21);
  const auto weights = qlora::numeric_utility::GenerateGaussianVector<float>(kInDim * kOutDim, generator);
  const auto quantized_base = ::qlora::quantization::BlockWiseNf4Quantization(weights, 64, 256);

  LoRALinearLayer layer(kInDim, kOutDim, kRank, alpha, quantized_base, LayerMode::kTraining, &generator);

  Matrix<float> input(1, kInDim);
  input.FillGaussianMatrix(generator);
  Matrix<float> target(1, kOutDim);
  target.FillGaussianMatrix(generator);

  constexpr size_t num_iterations = 10;
  for (size_t i = 0; i < num_iterations; ++i) {
    auto pred = layer.Forward(input);
    const float loss_before_grad = CalculateMSE(pred, target);
    if (i == 0) {
      EXPECT_FLOAT_EQ(loss_before_grad, 9.54082f);
    }

    // Calculate Gradient of MSE: 2 * (pred - target) / N
    Matrix<float> grad_output(1, kOutDim);
    for(size_t i = 0; i < kOutDim; ++i) {
      grad_output[0, i] = 2.0f * (pred[0, i] - target[0, i]) / static_cast<float>(kOutDim);
    }

    layer.Backward(grad_output);
    layer.Step(0.01f);

    const float loss_after_grad = CalculateMSE(layer.Forward(input), target);
    std::cout << loss_before_grad << " " << loss_after_grad << std::endl;
    EXPECT_LT(loss_after_grad, loss_before_grad) << "Optimization step failed to reduce loss.";
  }
}
