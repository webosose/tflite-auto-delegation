#include <gtest/gtest.h>
#include <AccelerationPolicyManager.h>

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

    EXPECT_EQ(apm.GetPolicy(), APM::kCPUOnly);
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 0);
}

TEST_F(AccelerationPolicyManagerTest, 02_01_set_and_get_CPU_ONLY_policy)
{
    APM apm;

    EXPECT_TRUE(apm.SetPolicy(APM::kCPUOnly));
    EXPECT_EQ(apm.GetPolicy(), APM::kCPUOnly);
}

TEST_F(AccelerationPolicyManagerTest, 02_02_set_and_get_CPU_ONLY_policy)
{
    std::string config(
        "{\n"
        "    \"policy\" : \"CPU_ONLY\"\n"
        "}");

    APM apm(config);

    EXPECT_EQ(apm.GetPolicy(), APM::kCPUOnly);
}

TEST_F(AccelerationPolicyManagerTest, 02_03_set_and_get_CPU_ONLY_policy_empty_json)
{
    std::string config("{}");

    APM apm(config);

    EXPECT_EQ(apm.GetPolicy(), APM::kCPUOnly);
}

TEST_F(AccelerationPolicyManagerTest, 02_04_set_and_get_CPU_ONLY_policy_empty_string)
{
    std::string config("");

    APM apm(config);

    EXPECT_EQ(apm.GetPolicy(), APM::kCPUOnly);
}

TEST_F(AccelerationPolicyManagerTest, 03_01_set_and_get_MAX_PRECISION_policy)
{
    std::string config(
        "{\n"
        "    \"policy\" : \"MAX_PRECISION\"\n"
        "}");
    APM apm(config);

    EXPECT_EQ(apm.GetPolicy(), APM::kMaximumPrecision);
}

TEST_F(AccelerationPolicyManagerTest, 03_02_set_and_get_MAX_PRECISION_policy)
{
    APM apm;

    EXPECT_TRUE(apm.SetPolicy(APM::kMaximumPrecision));
    EXPECT_EQ(apm.GetPolicy(), APM::kMaximumPrecision);
}

TEST_F(AccelerationPolicyManagerTest, 04_01_set_and_get_MIN_LATENCY_policy)
{
    std::string config(
        "{\n"
        "    \"policy\" : \"MIN_LATENCY\"\n"
        "}");
    APM apm(config);

    EXPECT_EQ(apm.GetPolicy(), APM::kMinimumLatency);
}

TEST_F(AccelerationPolicyManagerTest, 04_02_set_and_get_MIN_LATENCY_policy)
{
    APM apm;

    EXPECT_TRUE(apm.SetPolicy(APM::kMinimumLatency));
    EXPECT_EQ(apm.GetPolicy(), APM::kMinimumLatency);
}

TEST_F(AccelerationPolicyManagerTest, 05_01_set_and_get_LOAD_BALANCING_policy)
{
    std::string config(
        "{\n"
        "    \"policy\" : \"LOAD_BALANCING\",\n"
        "    \"cpu_fallback_percentage\": 10\n"
        "}");
    APM apm(config);
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 10);
    EXPECT_EQ(apm.GetPolicy(), APM::kEnableLoadBalancing);
}

TEST_F(AccelerationPolicyManagerTest, 05_02_set_and_get_LOAD_BALANCING_policy)
{
    APM apm;

    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 0);

    EXPECT_TRUE(apm.SetCPUFallbackPercentage(25));
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 25);
    EXPECT_EQ(apm.GetPolicy(), APM::kEnableLoadBalancing);
}

TEST_F(AccelerationPolicyManagerTest, 06_set_and_get_cpu_fallback_percentage)
{
    APM apm;

    EXPECT_TRUE(apm.SetPolicy(APM::kMaximumPrecision));
    EXPECT_EQ(apm.GetPolicy(), APM::kMaximumPrecision);

    EXPECT_TRUE(apm.SetCPUFallbackPercentage(25));
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 25);

    EXPECT_EQ(apm.GetPolicy(), APM::kEnableLoadBalancing);

    EXPECT_TRUE(apm.SetCPUFallbackPercentage(120));
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 100);

    EXPECT_TRUE(apm.SetCPUFallbackPercentage(-10));
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 0);
}