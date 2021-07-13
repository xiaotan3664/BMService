#include <algorithm>
#include <set>
#include<sstream>
#include<string>

#include "BMUtils.h"
#include "BMEnv.h"
//#include "bmlib_runtime.h"

namespace bm {
template <typename T>
    std::set<T> string_to_set(const std::string &s)
    {
        std::string new_str = s;
        for(auto& c: new_str){
           if(!isdigit(c) && c!='.') c= ' ';
        }
        std::set<T> values;
        std::istringstream is(new_str);
        T v= 0;
        while(is>>v){
            values.insert(v);
        }
        return values;
    }

    std::set<int> get_available_devices(){
        std::set<int> devices;
        int device_num = 10;// get_device_count();
        if(device_num == 0) return devices;
        const char* device_str = getenv(BM_USE_DEVICE);
//        if(!device_str) {
//            for(int i=0; i<device_num; i++) devices.insert(i)
//        } else {
//            auto candidates = string_to_set<int>(device_str);
//            std::copy_if(candidates.begin(), candidates.end(), std::back_inserter(devices),
//                         [device_num](int d) { return d<device_num; })
//        }
        return devices;

    }
};
