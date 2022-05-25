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

TEST_F(AccelerationPolicyManagerTest, 01_default_policy)
{
    APM apm;

    EXPECT_EQ(apm.GetPolicy(), APM::kCPUOnly);
}

TEST_F(AccelerationPolicyManagerTest, 02_set_and_get_policy)
{
    APM apm;

    EXPECT_TRUE(apm.SetPolicy(APM::kCPUOnly));
    EXPECT_EQ(apm.GetPolicy(), APM::kCPUOnly);

    EXPECT_TRUE(apm.SetPolicy(APM::kMaximumPrecision));
    EXPECT_EQ(apm.GetPolicy(), APM::kMaximumPrecision);

    EXPECT_TRUE(apm.SetPolicy(APM::kMinimumLatency));
    EXPECT_EQ(apm.GetPolicy(), APM::kMinimumLatency);

    EXPECT_TRUE(apm.SetPolicy(APM::kEnableLoadBalancing));
    EXPECT_EQ(apm.GetPolicy(), APM::kEnableLoadBalancing);
}
