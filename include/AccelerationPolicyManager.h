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

        AccelerationPolicyManager();
        AccelerationPolicyManager(const std::string &config);

        virtual ~AccelerationPolicyManager();

        bool setPolicy(Policy policy);
        Policy getPolicy();

        bool setCPUFallbackPercentage(int percentage);
        int getCPUFallbackPercentage();

    private:
        Policy stringToPolicy(const std::string &policy);
        Policy m_policy = kCPUOnly;
        int m_cpuFallbackPercentage = 0;
    };
} // end of namespace aif

#endif
