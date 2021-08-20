#include <vector>
#include "BMEnv.h"
#include "BMLog.h"
#include "BMDeviceUtils.h"
#include "BMCommonUtils.h"
#include "bmlib_runtime.h"

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
}

