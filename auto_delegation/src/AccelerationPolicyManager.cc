#include "AccelerationPolicyManager.h"

AccelerationPolicyManager::AccelerationPolicyManager()
{
}

AccelerationPolicyManager::~AccelerationPolicyManager()
{
}

bool AccelerationPolicyManager::SetPolicy(AccelerationPolicyManager::Policy policy)
{
    policy_ = policy;

    return true;
}

AccelerationPolicyManager::Policy AccelerationPolicyManager::GetPolicy()
{
    return policy_;
}