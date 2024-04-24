/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <AccelerationPolicyManager.h>

using namespace aif;

typedef AccelerationPolicyManager APM;

class AccelerationPolicyManagerTest : public ::testing::Test
{
protected:
    AccelerationPolicyManagerTest() = default;
    ~AccelerationPolicyManagerTest() = default;

    void SetUp() override
    {
    }

    void TearDown() override
    {
    }
};

TEST_F(AccelerationPolicyManagerTest, 01_default)
{
    APM apm;

    EXPECT_EQ(apm.getPolicy(), APM::kCPUOnly);
    EXPECT_EQ(apm.getCPUFallbackPercentage(), 0);
}

TEST_F(AccelerationPolicyManagerTest, 02_01_set_and_get_CPU_ONLY_policy)
{
    APM apm;

    EXPECT_TRUE(apm.setPolicy(APM::kCPUOnly));
    EXPECT_EQ(apm.getPolicy(), APM::kCPUOnly);
}

TEST_F(AccelerationPolicyManagerTest, 02_02_set_and_get_CPU_ONLY_policy)
{
    std::string config(
        "{\n"
        "    \"policy\" : \"CPU_ONLY\"\n"
        "}");

    APM apm(config);

    EXPECT_EQ(apm.getPolicy(), APM::kCPUOnly);
}

TEST_F(AccelerationPolicyManagerTest, 02_03_set_and_get_CPU_ONLY_policy_empty_json)
{
    std::string config("{}");

    APM apm(config);

    EXPECT_EQ(apm.getPolicy(), APM::kCPUOnly);
}

TEST_F(AccelerationPolicyManagerTest, 02_04_set_and_get_CPU_ONLY_policy_empty_string)
{
    std::string config("");

    APM apm(config);

    EXPECT_EQ(apm.getPolicy(), APM::kCPUOnly);
}

#ifdef USE_GPU
TEST_F(AccelerationPolicyManagerTest, 03_01_set_and_get_MAX_PRECISION_policy)
{
    std::string config(
        "{\n"
        "    \"policy\" : \"MAX_PRECISION\"\n"
        "}");
    APM apm(config);

    EXPECT_EQ(apm.getPolicy(), APM::kMaximumPrecision);
}

TEST_F(AccelerationPolicyManagerTest, 03_02_set_and_get_MAX_PRECISION_policy)
{
    APM apm;

    EXPECT_TRUE(apm.setPolicy(APM::kMaximumPrecision));
    EXPECT_EQ(apm.getPolicy(), APM::kMaximumPrecision);
}

TEST_F(AccelerationPolicyManagerTest, 04_01_set_and_get_MIN_LATENCY_policy)
{
    std::string config(
        "{\n"
        "    \"policy\" : \"MIN_LATENCY\"\n"
        "}");
    APM apm(config);

    EXPECT_EQ(apm.getPolicy(), APM::kMinimumLatency);
}

TEST_F(AccelerationPolicyManagerTest, 04_02_set_and_get_MIN_LATENCY_policy)
{
    APM apm;

    EXPECT_TRUE(apm.setPolicy(APM::kMinimumLatency));
    EXPECT_EQ(apm.getPolicy(), APM::kMinimumLatency);
}

TEST_F(AccelerationPolicyManagerTest, 05_01_set_and_get_LOAD_BALANCING_policy)
{
    std::string config(
        "{\n"
        "    \"policy\" : \"LOAD_BALANCING\",\n"
        "    \"cpu_fallback_percentage\": 10\n"
        "}");
    APM apm(config);
    EXPECT_EQ(apm.getCPUFallbackPercentage(), 10);
    EXPECT_EQ(apm.getPolicy(), APM::kEnableLoadBalancing);
}

TEST_F(AccelerationPolicyManagerTest, 06_set_and_get_cpu_fallback_percentage)
{
    APM apm;

    EXPECT_TRUE(apm.setPolicy(APM::kEnableLoadBalancing));
    EXPECT_EQ(apm.getPolicy(), APM::kEnableLoadBalancing);

    EXPECT_TRUE(apm.setCPUFallbackPercentage(25));
    EXPECT_EQ(apm.getCPUFallbackPercentage(), 25);

    EXPECT_TRUE(apm.setCPUFallbackPercentage(120));
    EXPECT_EQ(apm.getCPUFallbackPercentage(), 100);

    EXPECT_TRUE(apm.setCPUFallbackPercentage(-10));
    EXPECT_EQ(apm.getCPUFallbackPercentage(), 0);
}

TEST_F(AccelerationPolicyManagerTest, 07_set_and_get_GPU_Caching)
{
    std::string config = R"(
        {
            "serialization" : {
                "dir_path" : "/usr/share/aif",
                "model_token" : "pose2d_gpu_mid"
            }
        }
    )";

    APM apm(config);
    EXPECT_EQ(apm.getCache().useCache, true);
    EXPECT_EQ(apm.getCache().serialization_dir, "/usr/share/aif");
    EXPECT_EQ(apm.getCache().model_token, "pose2d_gpu_mid");
    EXPECT_EQ(apm.getPolicy(), APM::kCPUOnly);
}
#endif
