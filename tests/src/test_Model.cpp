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
        // TODO
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