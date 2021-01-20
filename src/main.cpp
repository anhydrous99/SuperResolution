#include <iostream>
#include <filesystem>
#include <cxxopts.hpp>
#include <glog/logging.h>

#include "Model.h"

namespace fs = std::filesystem;


int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    cxxopts::Options options("SuperResolution", "Uses ESRGAN to interpolate an image");
    options.add_options()
            ("m,model_path", "The path to the TorchScript ESRGAN Model", cxxopts::value<fs::path>()->default_value("ESRGAN.pt"))
            ("i,input", "The path to the input video or image", cxxopts::value<fs::path>())
            ("side_dim", "The out dimension size for model", cxxopts::value<size_t>()->default_value("128"))
            ("scale", "The upscale factor for model", cxxopts::value<size_t>()->default_value("4"))
            ("h,help", "Print Usage");
    auto results = options.parse(argc, argv);
    if (results.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }
    if (!results.count("input")) {
        std::cout << "Error: Missing input argument\n" << options.help() << std::endl;
        return EXIT_SUCCESS;
    }
    fs::path model_path = results["model_path"].as<fs::path>();
    fs::path input = results["input"].as<fs::path>();
    size_t out_dim_size = results["side_dim"].as<size_t>();
    size_t scale = results["scale"].as<size_t>();
    //CHECK(fs::is_regular_file(model_path)) << "Model is not a regular file or doesn't exist.";
    //CHECK(fs::is_regular_file(input)) << "Input is not a regular file or doesn't exist.";

    // Import Model
    Model model(model_path, scale, out_dim_size);
    return EXIT_SUCCESS;
}
