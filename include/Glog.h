//
// Created by constexpr_dog on 2/1/21.
//

#ifndef SUPERRESOLUTION_GLOG_H
#define SUPERRESOLUTION_GLOG_H

#include <string>

class IGlog {
public:

    virtual void Log_Fatal(const std::string &message) = 0;
    virtual void Check(bool condition, const std::string &message) = 0;
};

class Glog : public IGlog {
public:
    explicit Glog(const char * executable_name);
    void Log_Fatal(const std::string &message) override;
    void Check(bool condition, const std::string &message) override;
};

#endif //SUPERRESOLUTION_GLOG_H
