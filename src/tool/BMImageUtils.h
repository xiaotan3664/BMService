#ifndef BMIMAGEUTILS_H
#define BMIMAGEUTILS_H
#include<string>
#include<map>
#include "bmcv_api.h"

//#define FFALIGN(x, n) ((((x)+((n)-1))/(n))*(n))

namespace bm {
std::vector<int> calcImageStride(
        int height, int width,
        bm_image_format_ext format,
        bm_image_data_format_ext dtype,
        int align_bytes = 1);

bm_image readAlignedImage(bm_handle_t handle, const std::string& name);

// for inceptionv3
void centralCropAndResize(bm_handle_t handle,
                          std::vector<bm_image>& srcImages,
                          std::vector<bm_image>& dstImages,
                          float centralFactor = 0.875);
// for yolov
void aspectScaleAndPad(bm_handle_t handle,
                          std::vector<bm_image>& srcImages,
                          std::vector<bm_image>& dstImages,
                          bmcv_color_t padColor);

void saveImage(bm_image& bmImage, const std::string& name = "image.jpg");
void dumpImage(bm_image& bmImage, const std::string& name = "image.txt");

std::map<size_t, std::string> loadLabels(const std::string& filename);
std::map<std::string, size_t> loadClassRefs(const std::string& filename, const std::string& prefix="");
}

#endif // BMIMAGEUTILS_H
