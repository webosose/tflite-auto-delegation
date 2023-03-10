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
        AccelerationPolicyManager(std::string config);

        virtual ~AccelerationPolicyManager();

        bool SetPolicy(Policy policy);
        Policy GetPolicy();

        bool SetCPUFallbackPercentage(int percentage);
        int GetCPUFallbackPercentage();

    private:
        Policy stringToPolicy(std::string policy);
        Policy policy_ = kCPUOnly;
        int cpu_fallback_percentage_ = 0;
    };
} // end of namespace aif

#endif
