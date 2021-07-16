#ifndef BMENV_H
#define BMENV_H

#define BM_ENV_PREFIX "BMSERVICE_"
// export BMSERVICE_USE_DEVICE="": use all available devices
// export BMSERVICE_USE_DEVICE="1 2": use device_id=1, device_id=2
#define BM_USE_DEVICE (BM_ENV_PREFIX "USE_DEVICE")

#define BM_LOG_LEVEL (BM_ENV_PREFIX "LOG_LEVEL")

#endif // BMENV_H
