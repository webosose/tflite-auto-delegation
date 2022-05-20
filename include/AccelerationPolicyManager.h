#ifndef ACCELERATIONPOLICYMANAGER_H_
#define ACCELERATIONPOLICYMANAGER_H_
#include <iostream>
#include <vector>

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

    enum DelegateType
    {
        kTfLiteGPUDelegateV2 = 0,
        kWebOSNPU,
    };
    AccelerationPolicyManager(const std::vector<DelegateType> &supportedDelegates);

    virtual ~AccelerationPolicyManager();

    bool SetPolicy(Policy policy);
    Policy GetPolicy();

private:
    const std::vector<DelegateType> &supportedDelegates_;
    Policy policy_ = kCPUOnly;
};

#endif