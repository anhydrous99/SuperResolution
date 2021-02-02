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
    };


    TEST_P(ModelTest, Preprocess) { // NOLINT - Suppresses clang-tidy: initialization with static duration may throw an exception
        std::vector<int> sides{16, 32, 177, 256};
        for (int height : sides) {
            for (int width : sides) {
                at::Tensor rand_tensor = torch::rand({height, width, 3});
                size_t predicted_blocks = std::ceil(static_cast<float>(height) / 32.f) * std::ceil(static_cast<float>(width) / 32.f);
                auto output = model->preprocess(rand_tensor);
                ASSERT_EQ(predicted_blocks, output.size());
                ASSERT_EQ(3, output[0].ndimension());
                ASSERT_EQ(output[0].dtype(), torch::kFloat32);
                ASSERT_LE(output[0].max().item<float>(), 1.f);
                ASSERT_GE(output[0].min().item<float>(), 0.f);
            }
        }
    }

    TEST_P(ModelTest, Postprocesses) { // NOLINT
        // TODO
    }

    TEST_P(ModelTest, RunVector) { // NOLINT
        // TODO
    }

    TEST_P(ModelTest, RunMat) { // NOLINT
        // TODO
    }

    INSTANTIATE_TEST_SUITE_P(MultipleBatchSize, ModelTest, Values(1, 2, 4, 8)); // NOLINT
}