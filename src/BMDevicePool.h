#ifndef BMDEVICEPOOL_H
#define BMDEVICEPOOL_H

#include<vector>
#include "BMPipelinePool.h"

namespace bm {

struct BMDeviceContext {
    BMDeviceContext(size_t deviceId);

};

template<typename InType, typename OutType>
class BMDevicePoolBase
{
public:

    BMDevicePoolBase() {}
    virtual ~BMDevicePoolBase() {}

private:
    BMPipelinePool<InType, OutType, BMDeviceContext>  pool;
    std::vector<int> deviceIds;
};

}

#endif // BMDEVICEPOOL_H
