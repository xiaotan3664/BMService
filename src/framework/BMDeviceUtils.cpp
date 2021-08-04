#include <vector>
#include <sys/stat.h>
#include <dirent.h>
#include "bmlib_runtime.h"
#include "BMEnv.h"
#include "BMLog.h"
#include "BMDeviceUtils.h"
#include "BMCommonUtils.h"

namespace bm {

std::vector<DeviceId> getAvailableDevices() {
    std::vector<DeviceId> devices;
    DeviceId device_num = 0;
    if(bm_dev_getcount((int*)&device_num) != BM_SUCCESS){
        BMLOG(FATAL, "no device is found!");
    }
    if(device_num == 0) return devices;
    const char* device_str = getenv(BM_USE_DEVICE);
    if(!device_str) {
        for(int i=0; i<device_num; i++) devices.push_back(i);
    } else {
        auto candidates = stringToSet<DeviceId>(device_str);
        for(auto id: candidates){
            if(id >= device_num) {
                BMLOG(WARNING, "device_id=%d, but device_num=%d", id, device_num);
                continue;
            }
            devices.push_back(id);
        }
    }
    return devices;
}
void forEachFile(const std::string& path, std::function<bool (const std::string& )> func) {
    DIR *d = nullptr;
    struct dirent *dp = nullptr;
    struct stat st;
    std::string fullpath;
    if(!func) return;
    if(path.empty()) return;
    if(stat(path.c_str(), &st) < 0) return;
    if(S_ISDIR(st.st_mode)){
        if(!(d = opendir(path.c_str()))) return;
        while((dp = readdir(d)) != nullptr) {
            // if((!strncmp(dp->d_name, ".", 1)) || (!strncmp(dp->d_name, "..", 2)))continue;
            fullpath = path + "/" + dp->d_name;
            stat(fullpath.c_str(), &st);
            if(!S_ISDIR(st.st_mode)) {
                if(!func(fullpath)) break;
            }
        }
        closedir(d);
    } else {
        func(fullpath);
    }
}

void forEachBatch(const std::string &path, size_t batchSize, std::function<void (const std::vector<std::string> &)> func)
{
    DIR *d = nullptr;
    struct dirent *dp = nullptr;
    struct stat st;
    std::string fullpath;
    std::vector<std::string> batchPaths;

    if(!func) return;
    if(path.empty()) return;
    if(stat(path.c_str(), &st) < 0) return;
    if(!S_ISDIR(st.st_mode)) return;
    if(!(d = opendir(path.c_str()))) return;

    while((dp = readdir(d)) != nullptr) {
        fullpath = path + "/" + dp->d_name;
        stat(fullpath.c_str(), &st);
        if(!S_ISDIR(st.st_mode)) {
            batchPaths.push_back(fullpath);
            if(batchSize == batchPaths.size()) {
                func(std::move(batchPaths));
		batchPaths.clear();
            }
        }
    }

    if(!batchPaths.empty()){
        func(std::move(batchPaths));
	batchPaths.clear();
    }

    closedir(d);
}

}

