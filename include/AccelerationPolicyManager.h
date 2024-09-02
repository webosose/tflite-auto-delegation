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
            kCPUOnly = 0,
            kMaximumPrecision,
            kMinimumLatency,
            kEnableLoadBalancing,
            kPytorchModelGPU,
        };

        typedef struct Caching
        {
            bool useCache;
            std::string serialization_dir;
            std::string model_token;
        } Caching;


        AccelerationPolicyManager();
        AccelerationPolicyManager(const std::string &config);

        virtual ~AccelerationPolicyManager();

        bool setPolicy(Policy policy);
        Policy getPolicy();

        void setCache(Caching cache);
        Caching getCache();

        bool setCPUFallbackPercentage(int percentage);
        int getCPUFallbackPercentage();

    private:
        Policy stringToPolicy(const std::string &policy);
        Policy m_policy = kCPUOnly;
        Caching m_cache = {false, "", ""};
        int m_cpuFallbackPercentage = 0;
    };
} // end of namespace aif

#endif
