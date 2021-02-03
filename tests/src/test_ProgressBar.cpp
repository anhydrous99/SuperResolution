//
// Created by constexpr_dog on 2/1/21.
//

#include <gtest/gtest.h>
#include "ProgressBar.h"

namespace {
    using ::testing::TestWithParam;
    using ::testing::Values;
    using ::testing::internal::CaptureStdout;
    using ::testing::internal::GetCapturedStdout;

    // Create fixture
    struct ProgressBarTest : TestWithParam<int> {
    protected:
        void SetUp() override {
            CaptureStdout();
            progress_bar = new ProgressBar(GetParam());
            std::string output = GetCapturedStdout();

            // Always print these three character regardless of terminal size
            ASSERT_EQ(*output.begin(), '[');
            ASSERT_EQ(*(output.end() - 1), '\r');
            ASSERT_EQ(*(output.end() - 2), '%');
        }

        void TearDown() override {
            delete progress_bar;
            progress_bar = nullptr;
        }

        ~ProgressBarTest() override { delete progress_bar; }
        ProgressBar *progress_bar{nullptr};
    };

    TEST_P(ProgressBarTest, AllSteps) { // NOLINT
        CaptureStdout();
        for (int i = 0; i < GetParam(); i++)
            progress_bar->step();
        std::string output = GetCapturedStdout();
        ASSERT_EQ(output.size(), GetParam()*8);
    }

    INSTANTIATE_TEST_SUITE_P(MultipleCounts, ProgressBarTest, Values(0, 1, 3, 10, 100)); // NOLINT
}