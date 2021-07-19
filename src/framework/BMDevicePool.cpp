#include "BMDevicePool.h"
#include "BMImageUtils.h"

namespace bm {

BMDeviceContext::BMDeviceContext(DeviceId deviceId, const std::string &bmodel):deviceId(deviceId), batchSize(batchSize) {
    batchSize = -1;
    auto status = bm_dev_request(&handle, deviceId);
    BM_ASSERT_EQ(status, BM_SUCCESS);
    pBMRuntime = bmrt_create(handle);
    BM_ASSERT(pBMRuntime != nullptr, "cannot create bmruntime handle");
    net = std::make_shared<BMNetwork>(pBMRuntime, bmodel);
    batchSize = net->getBatchSize();
    net->showInfo();
}

bm_device_mem_t BMDeviceContext::allocDeviceMem(size_t bytes) {
    bm_device_mem_t mem;
    if(bm_malloc_device_byte(handle, &mem, bytes) != BM_SUCCESS){
        BMLOG(FATAL, "cannot alloc device mem, size=%d", bytes);
    }
    mem_to_free.push_back(mem);
    return mem;
}

void BMDeviceContext::freeDeviceMem(bm_device_mem_t &mem){
    auto iter = std::find_if(mem_to_free.begin(), mem_to_free.end(), [&mem](bm_device_mem_t& m){
            return bm_mem_get_device_addr(m) ==  bm_mem_get_device_addr(mem);
    });
    BM_ASSERT(iter != mem_to_free.end(), "cannot free mem!");
    bm_free_device(handle, mem);
    mem_to_free.erase(iter);
}

void BMDeviceContext::allocMemForTensor(TensorPtr tensor){
    auto mem_size = tensor->get_mem_size();
    auto mem = allocDeviceMem(mem_size);
    tensor->set_device_mem(&mem);
}

std::vector<bm_image> BMDeviceContext::allocImagesWithoutMem(
        int num, int height, int width, bm_image_format_ext format, bm_image_data_format_ext dtype, int align_bytes) {
    auto stride = calcImageStride(height, width, format, dtype, align_bytes);
    std::vector<bm_image> images;
    for(int i=0; i<num; i++){
        bm_image image;
        bm_image_create(handle, height, width, format, dtype, &image, stride.data());
        images.push_back(image);
        info_to_free.push_back(image);
    }
    return images;
}

std::vector<bm_image> BMDeviceContext::allocImages(int num, int height, int width, bm_image_format_ext format, bm_image_data_format_ext dtype, int align_bytes, int heap_id) {
    auto stride = calcImageStride(height, width, format, dtype, align_bytes);
    std::vector<bm_image> images;
    for(int i=0; i<num; i++){
        bm_image image;
        bm_image_create(handle, height, width, format, dtype, &image, stride.data());
        images.push_back(image);
    }
    bm_image_alloc_contiguous_mem(num, images.data(), heap_id);
    images_to_free.push_back(images);
    return images;
}

std::vector<bm_image> BMDeviceContext::allocAlignedImages(int num, int height, int width, bm_image_format_ext format, bm_image_data_format_ext dtype, int heap_id) {
    return allocImages(num, height, width, format, dtype, 64, heap_id);
}

void BMDeviceContext::freeImages(std::vector<bm_image>& images)
{
    auto iter = std::find_if(images_to_free.begin(), images_to_free.end(), [&images](std::vector<bm_image>& ref_images){
            if(ref_images.empty() || images.empty()) return false;
            bm_device_mem_t ref_mem, mem;
            bm_image_get_device_mem(ref_images[0], &ref_mem);
            bm_image_get_device_mem(images[0], &mem);
            return bm_mem_get_device_addr(ref_mem) ==  bm_mem_get_device_addr(mem);
    });
    if(iter == images_to_free.end()) return;
    bm_image_free_contiguous_mem(images.size(), images.data());
    for(auto& image: images){
        bm_image_destroy(image);
    }
    images_to_free.erase(iter);
}

BMDeviceContext::~BMDeviceContext() {
    auto mems = mem_to_free;
    for(auto m : mems){
        freeDeviceMem(m);
    }
    auto images = images_to_free;
    for(auto& image: images){
        freeImages(image);
    }
    for(auto& info: info_to_free){
        bm_image_destroy(info);
    }
    bmrt_destroy(pBMRuntime);
    bm_dev_free(handle);
}

}
