#ifndef BMDETECTUTILS_H
#define BMDETECTUTILS_H

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "bmcv_api.h"

namespace bm {

struct DetectBox {
    size_t category;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float confidence;
    std::string categoryName;
    float iou(const DetectBox& b1);
    bool isValid(float width=1.0, float height=1.0) const;
    bool operator < (const DetectBox& other) const {
        return confidence < other.confidence;
    }
    bool operator > (const DetectBox& other) const {
        return confidence > other.confidence;
    }
};


void drawDetectBox(bm_image& bmImage, const std::vector<DetectBox>& boxes, const std::string& saveName="");

void drawDetectBoxEx(bm_image& bmImage, const std::vector<DetectBox>& boxes, const std::vector<DetectBox>& trueBoxes, const std::string& saveName="");
std::vector<DetectBox> singleNMS(const std::vector<DetectBox>& info,
                                 float iouThresh, size_t topk = 0, bool useSoftNms=false, float sigma=0.3);

std::ostream& operator<<(std::ostream& os, const DetectBox& box);

std::vector<std::vector<DetectBox>> batchNMS(const std::vector<std::vector<DetectBox>>& batchInfo,
                                             float iouThresh, size_t topk=0, bool useSoftNms=false, float sigma=0.3);


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

std::map<std::string, std::vector<DetectBox> > readCocoDatasetBBox(const std::string &cocoAnnotationFile);

}

#endif // BMDETECTUTILS_H
