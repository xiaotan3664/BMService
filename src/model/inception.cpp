#include <dirent.h>
#include "inception.h"

namespace bm {
using InType = std::vector<std::string>;
using ClassId = size_t;

struct PostOutType {
    InType rawIns;
    std::vector<ClassId> classes;
};

bool preProcess(const InType& in, const TensorVec& inTensors, ContextPtr ctx){
    auto tensor = inTensors[0];
    size_t batchSize = ctx->getBatchSize();
    return true;
}

bool postProcess(const InType& rawIn, const TensorVec& outTensors, PostOutType& postOut, ContextPtr ctx){
    return true;
}

bool resultProcess(const PostOutType& out){
    return true;
}


int main(){
    const std::string bmodel = "./compilation/compilation.bmodel";
    const std::string dataPath = "./dataset";
    BMDevicePool<InType, PostOutType> runner(bmodel, preProcess, postProcess);
    runner.start();
    size_t batchSize= runner.getBatchSize();

    std::thread dataThread([dataPath, batchSize, &runner](){
        forEachBatch(dataPath, batchSize, [&runner](const InType& imageFiles){
            runner.push(imageFiles);
            return true;
        });
        runner.push({});
    });
    std::thread resultThread([&runner](){
        PostOutType out;
        while(true){
            while(!runner.pop(out)) {
                std::this_thread::yield();
            }
            if(!resultProcess(out)){
                runner.stop();
                break;
            }
        }
    });

    dataThread.join();
    resultThread.join();
    return 0;
}
/*

bool Inception::preProcess(const Inception::InType &in, Inception::PreprocessOutType &out, ContextPtr ctx)
{
    auto net = ctx->getNetwork();
}

std::vector<Inception::PreprocessOutType> Inception::createPreProcessOutput(ContextPtr ctx)
{
    auto net = ctx->getNetwork();
    int image_n = input.size();

     // Check input parameters
     if( image_n > MAX_BATCH) {
         std::cout << "input image size > MAX_BATCH(" << MAX_BATCH << ")." << std::endl;
         return;
     }

     //1. resize image
     std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
     if (image_n > input_tensor->get_shape()->dims[0]) {
         std::cout << "input image size > input_shape.batch(" << input_tensor->get_shape()->dims[0] << ")." << std::endl;
         return;
     }

     int ret = 0;
     for(int i = 0; i < image_n; ++i) {
         //bm_image image1;
         bm_image image_aligned;
         // src_img
         //cv::bmcv::toBMI((cv::Mat&)images[i], &image1);

         int stride1[3], stride2[3];
         bm_image_get_stride(input[i], stride1);
         stride2[0] = FFALIGN(stride1[0], 64);
         stride2[1] = FFALIGN(stride1[1], 64);
         stride2[2] = FFALIGN(stride1[2], 64);

         bm_image_create(m_bmContext->handle(), input[i].height, input[i].width,
                         input[i].image_format, input[i].data_type, &image_aligned, stride2);

         bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
         bmcv_copy_to_atrr_t copyToAttr;
         memset(&copyToAttr, 0, sizeof(copyToAttr));
         copyToAttr.start_x = 0;
         copyToAttr.start_y = 0;
         copyToAttr.if_padding = 1;

         bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, input[i], image_aligned);

 #if USE_ASPECT_RATIO
         bool isAlignWidth = false;
         float ratio = get_aspect_scaled_ratio(input[i].width, input[i].height, m_net_w, m_net_h, &isAlignWidth);
         bmcv_padding_atrr_t padding_attr;
         memset(&padding_attr, 0, sizeof(padding_attr));
         padding_attr.dst_crop_sty = 0;
         padding_attr.dst_crop_stx = 0;
         padding_attr.padding_b = 114;
         padding_attr.padding_g = 114;
         padding_attr.padding_r = 114;
         padding_attr.if_memset = 1;
         if (isAlignWidth) {
             padding_attr.dst_crop_h = m_net_w*ratio;
             padding_attr.dst_crop_w = m_net_w;
         }else{
             padding_attr.dst_crop_h = m_net_h;
             padding_attr.dst_crop_w = m_net_w*ratio;
         }

         bmcv_rect_t crop_rect{0, 0, input[i].width, input[i].height};
         auto ret = bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &m_resized_imgs[i],
                                                   &padding_attr, &crop_rect);
 #else
         auto ret = bmcv_image_vpp_convert(m_bmContext->handle(), 1, src_imgs[i], &m_resized_imgs[i]);
 #endif
         assert(BM_SUCCESS == ret);

 #if DUMP_FILE
         cv::Mat resized_img;
         cv::bmcv::toMAT(&m_resized_imgs[i], resized_img);
         std::string fname = cv::format("resized_img_%d.jpg", i);
         cv::imwrite(fname, resized_img);
 #endif

         bm_image_destroy(input[i]);
         bm_image_destroy(image_aligned);
     }

     //2. converto
     float input_scale = input_tensor->get_scale();
     input_scale = input_scale* (float)1.0/255;
     bmcv_convert_to_attr converto_attr;
     converto_attr.alpha_0 = input_scale;
     converto_attr.beta_0 = 0;
     converto_attr.alpha_1 = input_scale;
     converto_attr.beta_1 = 0;
     converto_attr.alpha_2 = input_scale;
     converto_attr.beta_2 = 0;

     ret = bmcv_image_convert_to(m_bmContext->handle(), image_n, converto_attr, m_resized_imgs, m_converto_imgs);
     CV_Assert(ret == 0);

     //3. attach to tensor
     bm_device_mem_t input_dev_mem;
     bm_image_get_contiguous_device_mem(image_n, m_converto_imgs, &input_dev_mem);
     input_tensor->set_device_mem(&input_dev_mem);
    return {};
}

bool Inception::forward(const Inception::PreprocessOutType &in, Inception::ForwardOutType &out, Inception::ContextPtr ctx)
{

}

std::vector<Inception::ForwardOutType> Inception::createForwardOutput(ContextPtr ctx)
{
    auto net = ctx->getNetwork();
    return {};
}

bool Inception::postProcess(const Inception::ForwardOutType &in, Inception::PostprocessOutType &out, ContextPtr ctx)
{

}
*/


}
