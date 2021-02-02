//
// Created by constexpr_dog on 2/1/21.
//

#include <gtest/gtest.h>
#include "mock_Glog.h"
#include "utils.h"


namespace {
    using ::testing::Test;
    using ::testing::Values;

    // Create fixture
    struct UtilsTest : Test {
        MockGlog glog;
    };

    TEST_F(UtilsTest, check_input_extensions_image) { // NOLINT
        EXPECT_CALL(glog, Check(false, "Input file extension is not supported\n")).Times(1);
        bool res;
        res = check_input_extensions(".jpg", reinterpret_cast<Glog *>(&glog));
        ASSERT_EQ(res, true);
    }

    TEST_F(UtilsTest, check_input_extensions_video) { // NOLINT
        EXPECT_CALL(glog, Check(false, "Input file extension is not supported\n")).Times(1);
        bool res;
        res = check_input_extensions(".mp4", reinterpret_cast<Glog *>(&glog));
        ASSERT_EQ(res, false);
    }

    TEST_F(UtilsTest, check_input_extensions_invalid) { // NOLINT
        EXPECT_CALL(glog, Check(true, "Input file extension is not supported\n")).Times(1);
        check_input_extensions("asdf", reinterpret_cast<Glog *>(&glog));
    }
}