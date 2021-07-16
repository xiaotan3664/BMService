#ifndef BMDEVICEUTILES_H
#define BMDEVICEUTILES_H

#include <vector>
#include <string>
#include <functional>

namespace bm {

using DeviceId = size_t;
std::vector<DeviceId> getAvailableDevices();

void forEachFile(const std::string& path, std::function<void(const std::string& )> func);
void forEachBatch(const std::string& path, size_t batchSize, std::function<void(const std::vector<std::string>&)> func);
}
#endif
