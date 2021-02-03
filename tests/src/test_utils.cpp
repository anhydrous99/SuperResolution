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
}