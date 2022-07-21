/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#ifndef ACCELERATIONPOLICYMANAGER_H_
#define ACCELERATIONPOLICYMANAGER_H_
#include <vector>
#include <string>
#include "rapidjson/document.h"
#include "util.h"

class AccelerationPolicyManager
{
public:
    enum Policy
    {
        kCPUOnly = 0,
        kMaximumPrecision,
        kMinimumLatency,
        kEnableLoadBalancing,
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

#endif