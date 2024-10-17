/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ACCELERATIONPOLICYMANAGER_H_
#define ACCELERATIONPOLICYMANAGER_H_
#include <vector>
#include <string>
#include "rapidjson/document.h"

namespace aif
{
    class AccelerationPolicyManager
    {
    public:
        enum Policy
        {
            kCPUOnly = 0x0,
            kMaximumPrecision = 0x1,
            kMinimumLatency = 0x2,
            kEnableLoadBalancing = 0x3,
            kPytorchModelGPU = 0x4,
            kMinRes = 0x5,

            kMinLatencyMinRes = (kMinimumLatency | 0x10),
        };

        typedef struct Caching
        {
            bool useCache;
            std::string serialization_dir;
            std::string model_token;
        } Caching;

        typedef struct NnapiCaching
        {
            std::string cache_dir;
            std::string model_token;
            bool disallow_nnapi_cpu;
            int max_number_delegated_partitions;
            std::string accelerator_name;
        } NnapiCaching;


        AccelerationPolicyManager();
        AccelerationPolicyManager(const std::string &config);

        virtual ~AccelerationPolicyManager();

        bool setPolicy(Policy policy);
        Policy getPolicy();

        void setCache(Caching cache);
        const Caching& getCache();

        void setNnapiCache(std::string cache_dir, std::string model_token, bool disallow_nnapi_cpu = false, int max_number_delegated_partitions = 0, std::string accelerator_name = "");
        const NnapiCaching& getNnapiCache();

        bool setCPUFallbackPercentage(int percentage);
        int getCPUFallbackPercentage();

    private:
        Policy stringToPolicy(const std::string &policy);
        Policy m_policy = kCPUOnly;
        Caching m_cache = {false, "", ""};
        NnapiCaching m_nnapi_cache = {"", "", false, 0, ""};
        int m_cpuFallbackPercentage = 0;
    };
} // end of namespace aif

#endif
