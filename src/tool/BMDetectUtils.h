#ifndef BMDETECTUTILS_H
#define BMDETECTUTILS_H

#include <map>
#include <string>
#include <vector>
#include "bmcv_api.h"

namespace bm {

struct DetectBox {
    size_t category;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float confidence;
    float iou(const DetectBox& b1);
};

void drawDetectBox(bm_image& bmImage, const std::vector<DetectBox>& boxes, const std::string& saveName="", std::map<size_t, std::string> nameMap={});
std::vector<DetectBox> singleNMS(const std::vector<DetectBox>& info,
                                          float iouThresh, bool useSoftNms=false,
                                          float sigma=0.3);

std::vector<std::vector<DetectBox>> batchNMS(const std::vector<std::vector<DetectBox>>& batchInfo,
                                                      float iouThresh, bool useSoftNms=false, float sigma=0.3);


template <typename T, typename Pred = std::function<T(const T &)>>
size_t argmax(
    const T *data, size_t len,
    Pred pred = [](const T &v)
    { return v; })
{
    size_t maxIndex = 0;
    for(size_t i=1; i<len; i++){
        if(pred(data[maxIndex])<pred(data[i])) {
            maxIndex = i;
        }
    }
    return maxIndex;
}


}

#endif // BMDETECTUTILS_H
