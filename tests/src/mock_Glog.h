//
// Created by constexpr_dog on 2/1/21.
//

#include <gmock/gmock.h>
#include "Glog.h"

class MockGlog : public IGlog {
public:
    MOCK_METHOD1(Log_Fatal, void(const std::string&));
    MOCK_METHOD2(Check, void(bool, const std::string&));
};