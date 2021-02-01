//
// Created by constexpr_dog on 2/1/21.
//

#include <glog/logging.h>
#include "Glog.h"

static size_t GlogCount = 0;

Glog::Glog(const char * executable_name) {
    if (GlogCount < 1) {
        google::InitGoogleLogging(executable_name);
        GlogCount++;
    } else
        LOG(FATAL) << "Can only create one Glog object.";
}

void Glog::Log_Fatal(const std::string &message) {
    LOG(FATAL) << message;
}

void Glog::Check(bool condition, const std::string &message) {
    CHECK(condition) << message;
}
