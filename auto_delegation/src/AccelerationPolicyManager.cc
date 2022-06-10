#include "AccelerationPolicyManager.h"

AccelerationPolicyManager::AccelerationPolicyManager()
{
}

AccelerationPolicyManager::AccelerationPolicyManager(std::string config)
{
    rapidjson::Document d;
    d.Parse(config.c_str());

    if (!d.HasParseError() && d.FindMember("policy") != d.MemberEnd())
    {
        Policy policy = stringToPolicy(d["policy"].GetString());
        SetPolicy(policy);

        if (d.FindMember("cpu_fallback_percentage") != d.MemberEnd())
        {
            int cpuFallbackPercentage = d["cpu_fallback_percentage"].GetInt();
            SetCPUFallbackPercentage(cpuFallbackPercentage);
        }
    }
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

bool AccelerationPolicyManager::SetCPUFallbackPercentage(int percentage)
{
    if (percentage < 0)
        percentage = 0;
    else if (percentage > 100)
        percentage = 100;

    this->SetPolicy(kEnableLoadBalancing);
    cpu_fallback_percentage_ = percentage;

    return true;
}

int AccelerationPolicyManager::GetCPUFallbackPercentage()
{
    return cpu_fallback_percentage_;
}

AccelerationPolicyManager::Policy AccelerationPolicyManager::stringToPolicy(std::string policyStr)
{
    AccelerationPolicyManager::Policy policy = AccelerationPolicyManager::Policy::kCPUOnly;
    if (policyStr.compare("CPU_ONLY") == 0)
        policy = AccelerationPolicyManager::Policy::kCPUOnly;
    else if (policyStr.compare("MAX_PRECISION") == 0)
        policy = AccelerationPolicyManager::Policy::kMaximumPrecision;
    else if (policyStr.compare("MIN_LATENCY") == 0)
        policy = AccelerationPolicyManager::Policy::kMinimumLatency;
    else if (policyStr.compare("LOAD_BALANCING") == 0)
        policy = AccelerationPolicyManager::Policy::kEnableLoadBalancing;

    return policy;
}