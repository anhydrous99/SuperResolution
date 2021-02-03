#include <iostream>
#include <algorithm>
#include <filesystem>
#include <cxxopts.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ProgressBar.h"
#include "Glog.h"
#include "Model.h"
#include "utils.h"

namespace fs = std::filesystem;


int main(int argc, char **argv) {
    Glog glog(argv[0]);
    cxxopts::Options options("SuperResolution", "Uses ESRGAN to interpolate an image");
    options.add_options()
            ("m,model_path", "The path to the TorchScript ESRGAN Model",
             cxxopts::value<fs::path>()->default_value("G_4x.pth"))
            ("i,input", "The path to the input video or image", cxxopts::value<fs::path>())
            ("o,output", "The path to the output video or image", cxxopts::value<fs::path>())
            ("side_dim", "The out dimension size for model", cxxopts::value<size_t>()->default_value("128"))
            ("scale", "The upscale factor for model", cxxopts::value<size_t>()->default_value("4"))
            ("b,batch_size", "Number of blocks to batch_size", cxxopts::value<size_t>()->default_value("1"))
            ("w,weight", "Cubic Interpolation weight (between 0 and 100)", cxxopts::value<size_t>()->default_value("0"))
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
    if (!results.count("output")) {
        std::cout << "Error: Missing output argumen\n" << options.help() << std::endl;
        return EXIT_SUCCESS;
    }
    fs::path model_path = results["model_path"].as<fs::path>();
    fs::path input_path = results["input"].as<fs::path>();
    fs::path output_path = results["output"].as<fs::path>();
    size_t out_dim_size = results["side_dim"].as<size_t>();
    size_t scale = results["scale"].as<size_t>();
    size_t batch_size = results["batch_size"].as<size_t>();
    size_t cubic_weight = results["weight"].as<size_t>();
    glog.Check(fs::is_regular_file(model_path), "Model is not a regular file or doesn't exist.");
    glog.Check(fs::is_regular_file(input_path), "Input is not a regular file or doesn't exist.");

    // Import Model
    Model model(model_path, scale, out_dim_size, batch_size, &glog);

    if (check_input_extensions(input_path.extension().string(), &glog)) {
        cv::Mat input_frame = cv::imread(input_path.string());
        cv::Mat output_frame = model.run(input_frame);

        // Blend cubic interpolation
        if (cubic_weight != 0) {
            glog.Check(cubic_weight >= 0 && cubic_weight <= 100, "Weight can only be between 0 and 100.");
            cv::Mat cubic_frame;
            cv::resize(input_frame, cubic_frame, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);
            double alpha = static_cast<double>(cubic_weight) / 100;
            double beta = (1.0 - alpha);
            cv::addWeighted(cubic_frame, alpha, output_frame, beta, 0.0, output_frame);
        }

        cv::imwrite(output_path.string(), output_frame);
    } else {
        // Initiate the video capture
        cv::VideoCapture capture(input_path.string());
        // Get the video's format
        int fourcc = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));
        // Get the video's frames per seconds
        double fps = capture.get(cv::CAP_PROP_FPS);
        // Get number of frames
        int n_frames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
        // Create progress bar
        ProgressBar bar(n_frames);
        // Get the capture image dimensions
        cv::Size input_size(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
                            static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
        // Calculate the super-sampled image dimensions
        cv::Size output_size(input_size.width * scale, input_size.height * scale);
        // Initiate the video writer for the super-sampled video
        cv::VideoWriter writer(output_path.string(), fourcc, fps, output_size);
        // Check if the video capture was opened successfully
        glog.Check(capture.isOpened(), "error opening video stream or file.\n");
        glog.Check(writer.isOpened(), "error opening output video to write.\n");

        while (capture.isOpened()) {
            cv::Mat input_frame;
            // Get frame from camera or video
            capture >> input_frame;

            if (input_frame.empty())
                break;

            // Split image into blocks to perform super sampling (per block)
            cv::Mat output_frame = model.run(input_frame);

            // Blend cubic interpolation
            if (cubic_weight != 0) {
                glog.Check(cubic_weight >= 0 && cubic_weight <= 100, "Weight can only be between 0 and 100.");
                cv::Mat cubic_frame;
                cv::resize(input_frame, cubic_frame, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);
                double alpha = static_cast<double>(cubic_weight) / 100;
                double beta = (1.0 - alpha);
                cv::addWeighted(cubic_frame, alpha, output_frame, beta, 0.0, output_frame);
            }

            // Write frame to video
            writer << output_frame;
            bar.step();
        }
        capture.release();
        writer.release();
    }
    return EXIT_SUCCESS;
}
