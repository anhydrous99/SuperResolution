//
// Created by constexpr_dog on 2/1/21.
//

#include <gtest/gtest.h>
#include "mock_Glog.h"
#include "utils.h"


namespace {
    using ::testing::Test;
    using ::testing::Values;


    TEST(UtilsTest, check_input_extensions_image) { // NOLINT
        MockGlog glog;
        EXPECT_CALL(glog, Check(true, "Input file extension is not supported\n")).Times(1);
        bool res = check_input_extensions(".jpg", reinterpret_cast<Glog *>(&glog));
        ASSERT_EQ(res, true);
    }

    TEST(UtilsTest, check_input_extensions_video) { // NOLINT
        MockGlog glog;
        EXPECT_CALL(glog, Check(true, "Input file extension is not supported\n")).Times(1);
        bool res = check_input_extensions(".mp4", reinterpret_cast<Glog *>(&glog));
        ASSERT_EQ(res, false);
    }

    TEST(UtilsTest, check_input_extensions_invalid) { // NOLINT
        MockGlog glog;
        EXPECT_CALL(glog, Check(false, "Input file extension is not supported\n")).Times(1);
        check_input_extensions("asdf", reinterpret_cast<Glog *>(&glog));
    }

    TEST(UtilsTest, blend_bicubic) { // NOLINT
        cv::Mat input = cv::Mat::zeros(cv::Size(3, 3), CV_8UC3);
        cv::Mat to_blend = cv::Mat::zeros(cv::Size(12, 12), CV_8UC3);
        auto* to_blend_ptr = to_blend.data;
        std::fill(to_blend_ptr, to_blend_ptr + (12 * 12 * 3), 66);

        cv::Mat output = blend_bicubic(input, to_blend, 4, 50);
        auto* output_ptr = output.data;

        for (size_t i = 0; i < 12 * 12 * 3; i++)
            ASSERT_EQ(output_ptr[i], 33);
    }
}