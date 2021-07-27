#include <stdio.h>
#include <math.h>
#include<vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "BMImageUtils.h"

namespace bm {
std::vector<int> calcImageStride(int height, int width,
                                 bm_image_format_ext format,
                                 bm_image_data_format_ext dtype,
                                 int align_bytes){
    int data_size = 1;
    switch (dtype) {
    case DATA_TYPE_EXT_FLOAT32:
        data_size = 4;
        break;
    case DATA_TYPE_EXT_4N_BYTE:
    case DATA_TYPE_EXT_4N_BYTE_SIGNED:
        data_size = 4;
        break;
    default:
        data_size = 1;
        break;
    }

    std::vector<int> stride;
    switch (format) {
    case FORMAT_YUV420P: {
        stride.resize(3);
        stride[0] = width * data_size;
        stride[1] = (FFALIGN(width, 2) >> 1) * data_size;
        stride[2] = stride[1];
        break;
    }
    case FORMAT_YUV422P: {
        stride.resize(3);
        stride[0] = width * data_size;
        stride[1] = (FFALIGN(width, 2) >> 1) * data_size;
        stride[2] = stride[1];
        break;
    }
    case FORMAT_YUV444P: {
        stride.assign(3, FFALIGN(width*data_size, align_bytes));
        break;
    }
    case FORMAT_NV12:
    case FORMAT_NV21: {
        stride.resize(2);
        stride[0] = width * data_size;
        stride[1] = FFALIGN(width, 2) * data_size;
        break;
    }
    case FORMAT_NV16:
    case FORMAT_NV61: {
        stride.resize(2);
        stride[0] = width * data_size;
        stride[1] = FFALIGN(width, 2) * data_size;
        break;
    }
    case FORMAT_GRAY: {
        stride.assign(1,FFALIGN(width * data_size, align_bytes));
        break;
    }
    case FORMAT_COMPRESSED: {
        break;
    }
    case FORMAT_BGR_PACKED:
    case FORMAT_RGB_PACKED: {
        stride.assign(1, FFALIGN(width * 3 * data_size, align_bytes));
        break;
    }
    case FORMAT_BGR_PLANAR:
    case FORMAT_RGB_PLANAR: {
        stride.assign(1, FFALIGN(width * data_size, align_bytes));
        break;
    }
    case FORMAT_BGRP_SEPARATE:
    case FORMAT_RGBP_SEPARATE: {
        stride.resize(3, FFALIGN(width * data_size, align_bytes));
        break;
    }
    default:{

    }
    }
    return stride;
}

bm_image readAlignedImage(bm_handle_t handle, const std::string &name)
{
    auto cvImage = cv::imread(name, cv::ImreadModes::IMREAD_COLOR, cv::bmcv::getId(handle));
    bm_image bmImage,  alignedImage;
    cv::bmcv::toBMI(cvImage, &bmImage);
    int stride1[3], stride2[3];
    bm_image_get_stride(bmImage, stride1);
    stride2[0] = FFALIGN(stride1[0], 64);
    stride2[1] = FFALIGN(stride1[1], 64);
    stride2[2] = FFALIGN(stride1[2], 64);
    bm_image_create(handle, bmImage.height, bmImage.width, bmImage.image_format, bmImage.data_type,
                    &alignedImage, stride2);
    bm_image_alloc_dev_mem(alignedImage, BMCV_IMAGE_FOR_IN);
    bmcv_copy_to_atrr_t copyToAttr;
    memset(&copyToAttr, 0, sizeof(copyToAttr));
    copyToAttr.start_x = 0;
    copyToAttr.start_y = 0;
    copyToAttr.if_padding = 1;

    bmcv_image_copy_to(handle, copyToAttr, bmImage, alignedImage);
    bm_image_destroy(bmImage);
    return alignedImage;
}

void centralCropAndResize(bm_handle_t handle,
                          std::vector<bm_image>& srcImages,
                          std::vector<bm_image>& dstImages,
                          float centralFactor){
   int numImage = srcImages.size();
   std::vector<bmcv_rect_t> rects(numImage);
   for(int i=0; i<numImage; i++){
       auto& rect = rects[i];
       auto& srcImage = srcImages[i];
       int height = srcImage.height;
       int width = srcImage.width;
       rect.start_x = round(width*(1-centralFactor)*0.5);
       rect.crop_w = round(width*centralFactor);
       rect.start_y = round(height*(1-centralFactor)*0.5);
       rect.crop_h = round(height*centralFactor);
   }
   std::vector<int> cropNumVec(numImage, 1);
   bmcv_image_vpp_basic(handle, numImage, srcImages.data(), dstImages.data(),
                        cropNumVec.data(), rects.data());
}

void dumpImage(bm_image& bmImage, const std::string& name){
    auto fp = fopen(name.c_str(), "w");
    int plane_num = bm_image_get_plane_num(bmImage);
    int* sizes = new int[plane_num];
    auto buffers = new void*[plane_num];
    bm_image_get_byte_size(bmImage, sizes);
    for(int i=0; i<plane_num; i++){
        buffers[i] = new unsigned char[sizes[i]];
    }
    bm_image_copy_device_to_host(bmImage, buffers);

    fprintf(fp, "plane_num=%d\n", plane_num);
    for(int i=0; i<plane_num; i++){
        fprintf(fp, "plane_size=%d\n", sizes[i]);
        for(int j=0; j<sizes[i]; j++){
            if(bmImage.data_type == DATA_TYPE_EXT_1N_BYTE){
                auto data=(unsigned char*) buffers[i];
                fprintf(fp, "%d\n", data[j]);
            } else if(bmImage.data_type == DATA_TYPE_EXT_1N_BYTE_SIGNED){
                auto data=(char*) buffers[i];
                fprintf(fp, "%d\n", data[j]);
            }
        }
        delete [] ((unsigned char*)buffers[i]);
    }
    delete [] buffers;
    delete [] sizes;
    fclose(fp);
}

void saveImage(bm_image& bmImage, const std::string& name){
    cv::Mat cvImage;
    cv::bmcv::toMAT(&bmImage, cvImage);
    cv::imwrite(name, cvImage);
}

static bool split_id_and_label(const std::string& line, size_t& id, std::string& label){
    auto iter = std::find(line.begin(), line.end(), ':');
    if(iter == line.end()){
        id++;
        label = line;
    } else {
        std::string classStr(line.begin(), iter);
        label = std::string(iter+1, line.end());
        id = std::stoul(classStr);
    }
}

std::map<size_t, std::string> loadLabels(const std::string &filename)
{
    std::ifstream ifs(filename);
    std::string line, label;
    size_t classId = 0;
    std::map<size_t, std::string> labelMap;

    while(std::getline(ifs, line)){
        split_id_and_label(line, classId, label);
        labelMap[classId] = label;
    }
    return labelMap;
}

}
