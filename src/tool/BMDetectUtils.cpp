#include <fstream>
#include "jsonxx.h"
#include "opencv2/opencv.hpp"
#include "BMCommonUtils.h"
#include "BMDetectUtils.h"
#include "BMLog.h"

namespace  bm {

float DetectBox::iou(const DetectBox &b1) {
    auto o_xmin = std::max(xmin, b1.xmin);
    auto o_xmax = std::min(xmax, b1.xmax);
    auto o_ymin = std::max(ymin, b1.ymin);
    auto o_ymax = std::min(ymax, b1.ymax);
    if(o_xmin > o_xmax || o_ymin > o_ymax) {
        return 0;
    }
    auto b0_area = (xmax-xmin)*(ymax-ymin);
    auto b1_area = (b1.xmax-b1.xmin)*(b1.ymax-b1.ymin);
    auto o_area = (o_xmax - o_xmin) * (o_ymax - o_ymin);
    return (float)o_area/(b0_area+b1_area-o_area);
}

std::vector<std::vector<DetectBox> > batchNMS(const std::vector<std::vector<DetectBox> > &batchInfo, float iouThresh, bool useSoftNms, float sigma){
    std::vector<std::vector<DetectBox>> results(batchInfo.size());
    for(size_t i=0; i<batchInfo.size(); i++){
        results[i] = singleNMS(batchInfo[i], iouThresh, useSoftNms, sigma);
    }
    return results;
}

std::vector<DetectBox> singleNMS(const std::vector<DetectBox> &info, float iouThresh, bool useSoftNms, float sigma){
    std::map<size_t, std::vector<DetectBox>> classifiedInfo;
    std::vector<DetectBox> bestBoxes;
    for(auto& i: info){
        classifiedInfo[i.category].push_back(i);
    }
    for (auto& ci: classifiedInfo) {
        auto& boxes = ci.second;
        while(!boxes.empty()){
            auto bestIndex = argmax(boxes.data(), boxes.size(), [](const DetectBox& i){ return i.confidence; });
            auto& bestBox = boxes[bestIndex];
            bestBoxes.push_back(bestBox);

            if(!useSoftNms){
                boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [&bestBox, iouThresh](const DetectBox& box){
                    return bestBox.iou(box)>iouThresh;
                }), boxes.end());
            } else {
                boxes.erase(boxes.begin()+bestIndex);
                std::for_each(boxes.begin(), boxes.end(), [&bestBox, sigma](DetectBox& box){
                    auto iouScore = bestBox.iou(box);
                    auto weight = exp(-(1.0 * iouScore*iouScore / sigma));
                    box.confidence *= weight;
                });
            }
        }
    }
    return bestBoxes;
}

using namespace jsonxx;
std::map<std::string, std::vector<DetectBox> > readCocoDatasetBBox(const std::string& cocoAnnotationFile)
{
    BMLOG(INFO, "Parsing annotation %s", cocoAnnotationFile.c_str());
    std::map<size_t, std::string> idToName;
    std::ifstream ifs(cocoAnnotationFile);
    Object coco;
    coco.parse(ifs);
    auto& images = coco.get<Array>("images");
    auto& annotations = coco.get<Array>("annotations");
    for(size_t i=0; i<images.size(); i++){
        auto& image = images.get<Object>(i);
        auto filename = image.get<std::string>("file_name");
        size_t id = image.get<Number>("id");
        idToName[id] = filename;
    }
    auto& categories = coco.get<Array>("categories");
    std::map<size_t, std::string> categoryMap;
    for(size_t i=0; i<categories.size(); i++){
        auto& category = categories.get<Object>(i);
        size_t id = category.get<Number>("id");
        auto name = category.get<std::string>("name");
        categoryMap[id]=name;
        BMLOG(INFO, "  category #%d: %s", id, name.c_str());
    }
    std::map<std::string, std::vector<DetectBox>> imageToBoxes;
    for(size_t i=0; i<annotations.size(); i++){
        auto& annotation = annotations.get<Object>(i);
        size_t imageId = annotation.get<Number>("image_id");
        if(!idToName.count(imageId)) continue;

        size_t categoryId = annotation.get<Number>("category_id");
        auto& bbox = annotation.get<Array>("bbox");
        DetectBox box;
        box.confidence = -1;
        box.category = categoryId;
        box.categoryName = categoryMap[categoryId];
        box.xmin = bbox.get<Number>(0);
        box.ymin = bbox.get<Number>(1);
        box.xmax = box.xmin + bbox.get<Number>(2);
        box.ymax = box.ymin + bbox.get<Number>(3);
        imageToBoxes[idToName[imageId]].push_back(box);
    }
    BMLOG(INFO, "Parsing annotation %s done", cocoAnnotationFile.c_str());
    return imageToBoxes;
}

void drawDetectBoxEx(bm_image &bmImage, const std::vector<DetectBox> &boxes, const std::vector<DetectBox> &trueBoxes, const std::string &saveName)
{
    //Draw a rectangle displaying the bounding box
    cv::Mat cvImage;
    auto status =cv::bmcv::toMAT(&bmImage, cvImage, true);
    BM_ASSERT_EQ(status, BM_SUCCESS);

    size_t borderWidth = 2;
    if(!trueBoxes.empty()){
        BMLOG(INFO, "draw true box for '%s'", saveName.c_str());
        for(size_t i=0; i<trueBoxes.size(); i++){
            auto& box = trueBoxes[i];
            cv::rectangle(cvImage, cv::Point((int)box.xmin, (int)box.ymin), cv::Point((int)box.xmax, (int)box.ymax), cv::Scalar(0, 255, 0), borderWidth);

            //Get the label for the class name and its confidence
            std::string label;
            label = std::to_string(box.category);
            if(box.categoryName != ""){
                label += "-" + box.categoryName;
            }

            //Display the label at the top of the bounding box
            int baseLine;
            cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1, &baseLine);
            auto top = std::max((int)box.ymax, labelSize.height);
            cv::putText(cvImage, label, cv::Point(box.xmin, top), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 1);
            BMLOG(INFO, "  box #%d: [%d, %d, %d, %d], %s", i,
                  (size_t)box.xmin, (size_t)box.ymin, (size_t)box.xmax, (size_t)box.ymax, label.c_str());
        }
    }
    BMLOG(INFO, "draw predicted box for '%s'", saveName.c_str());
    for(size_t i=0; i<boxes.size(); i++){
        auto& box = boxes[i];
        cv::rectangle(cvImage, cv::Point((int)box.xmin, (int)box.ymin), cv::Point((int)box.xmax, (int)box.ymax), cv::Scalar(0, 0, 255), borderWidth);

        //Get the label for the class name and its confidence
        std::string label = std::string(":") + cv::format("%.2f", box.confidence);
        if(box.categoryName != ""){
            label = std::string("-") + box.categoryName +  label;
        }
        label = std::to_string(box.category) + label;

        //Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        auto top = std::max((int)box.ymin, labelSize.height);
        cv::putText(cvImage, label, cv::Point(box.xmin, top), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1);
        BMLOG(INFO, "  box #%d: [%d, %d, %d, %d], %s", i,
              (size_t)box.xmin, (size_t)box.ymin, (size_t)box.xmax, (size_t)box.ymax,
              label.c_str());
    }
    static size_t saveCount=0;
    std::string fullPath = saveName;
    if(fullPath=="") {
        fullPath = std::string("00000000") + std::to_string(saveCount)+".jpg";
        fullPath = fullPath.substr(fullPath.size()-4-8);
    }
    cv::imwrite(fullPath, cvImage);

}

void drawDetectBox(bm_image &bmImage, const std::vector<DetectBox> &boxes, const std::string &saveName)   // Draw the predicted bounding box
{
    return drawDetectBoxEx(bmImage, boxes, {}, saveName);
}

std::ostream& operator<<(std::ostream &os, const DetectBox &box){
    std::string categoryName = box.categoryName;
    strReplaceAll(categoryName, " ", "_");
    os<<categoryName<<" ";
    if(box.confidence>=0){
        os<<box.confidence<<" ";
    }
    os<<box.xmin<<" ";
    os<<box.ymin<<" ";
    os<<box.xmax<<" ";
    os<<box.ymax;
    return os;
}


}
