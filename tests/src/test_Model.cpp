//
// Created by constexpr_dog on 2/1/21.
// Tests of the Model class
//

#include <gtest/gtest.h>
#include "mock_Glog.h"
#include "Model.h"

namespace {
    using ::testing::TestWithParam;
    using ::testing::Values;

    // Create fixture
    struct ModelTest : TestWithParam<int64_t> {
        MockGlog glog;
    protected:
        void SetUp() override {
            model = new Model("G_4x.pth", 4, 128, GetParam(), reinterpret_cast<Glog *>(&glog));
        }

        void TearDown() override {
            delete model;
            model = nullptr;
        }

        ~ModelTest() override { delete model; }

        Model *model{nullptr};
        std::vector<int64_t> sides{16, 32, 177, 256};
    };


    TEST_P(ModelTest, Preprocess) { // NOLINT - Suppresses clang-tidy: initialization with static duration may throw an exception
        for (int64_t height : sides) {
            for (int64_t width : sides) {
                at::Tensor rand_tensor = torch::rand({height, width, 3});
                size_t predicted_blocks = std::ceil(static_cast<float>(height) / 32.f) * std::ceil(static_cast<float>(width) / 32.f);
                auto output = model->preprocess(rand_tensor);
                ASSERT_EQ(predicted_blocks, output.size());
                ASSERT_EQ(3, output[0].ndimension());
                ASSERT_EQ(output[0].dtype(), torch::kFloat32);
                ASSERT_EQ(output[0].sizes().front(), 3);
                ASSERT_LE(output[0].max().item<float>(), 1.f);
                ASSERT_GE(output[0].min().item<float>(), 0.f);
            }
        }
    }

    TEST_P(ModelTest, Postprocesses) { // NOLINT
        for (int64_t height : sides) {
            for (int64_t width : sides) {
                // Generate random blocks
                std::vector<at::Tensor> rand_tensors;
                int64_t height_blocks = std::ceil(static_cast<float>(height) / 128.f);
                int64_t width_blocks = std::ceil(static_cast<float>(width) / 128.f);
                for (int64_t i = 0; i < height_blocks; i++) {
                    for (int64_t j = 0; j < width_blocks; j++) {
                        rand_tensors.push_back(torch::rand({
                            1,
                            3,
                            (i != height_blocks - 1) ? 128ll : height - (i) * 128,
                            (j != width_blocks - 1) ? 128ll : width - (j) * 128
                        }));
                    }
                }
                auto output = model->postprocess(rand_tensors, cv::Size(width, height));
                ASSERT_EQ(3, output.size(2));
                ASSERT_EQ(height, output.size(0));
                ASSERT_EQ(width, output.size(1));
            }
        }
    }

    TEST_P(ModelTest, RunVector) { // NOLINT
        for (int64_t height : sides) {
            for (int64_t width : sides) {
                std::vector<at::Tensor> rand_tensors;
                int64_t height_blocks = std::ceil(static_cast<float>(height) / 32.f);
                int64_t width_blocks = std::ceil(static_cast<float>(width) / 32.f);
                for (int64_t i = 0; i < height_blocks; i++) {
                    for (int64_t j = 0; j < width_blocks; j++) {
                        rand_tensors.push_back(torch::rand({
                                                                   3,
                                                                   (i != height_blocks - 1) ? 32ll : height - (i) * 32,
                                                                   (j != width_blocks - 1) ? 32ll : width - (j) * 32
                                                           }));
                    }
                }
                auto output = model->run(rand_tensors);
                ASSERT_EQ(3, output[0].ndimension());
                ASSERT_EQ(output[0].dtype(), torch::kFloat32);
                ASSERT_EQ(output[0].sizes().front(), 3);
                ASSERT_EQ(std::min(128ll, height * 4), output[0].size(1));
                ASSERT_EQ(std::min(128ll, width * 4), output[0].size(2));
            }
        }
    }

    TEST_P(ModelTest, RunMat) { // NOLINT
        // TODO
    }

    INSTANTIATE_TEST_SUITE_P(MultipleBatchSize, ModelTest, Values(1, 2, 4)); // NOLINT
}