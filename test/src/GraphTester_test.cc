/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#include <gtest/gtest.h>
#include <AutoDelegateSelector.h>
#include <tools/GraphTester.h>

using namespace aif;

typedef AutoDelegateSelector ADS;
typedef AccelerationPolicyManager APM;

class GraphTesterTest : public ::testing::Test
{
protected:
    GraphTesterTest() = default;
    ~GraphTesterTest() = default;

    void SetUp() override
    {
    }

    void TearDown() override
    {
    }

    std::vector<std::string> model_paths{
        std::string(AIF_INSTALL_DIR) + std::string("/model/face_detection_short_range.tflite")};
};

TEST_F(GraphTesterTest, 01_graphTester_fdshort_CpuOnly)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_EQ(graphTester.GetTotalNodeNum(), 164);
    EXPECT_FALSE(graphTester.IsDelegated());

    APM apm;
    EXPECT_TRUE(apm.SetPolicy(APM::kCPUOnly));
    ADS ads;
    EXPECT_TRUE(ads.SelectDelegate(*interpreter.get(), &apm));

    EXPECT_EQ(graphTester.GetTotalNodeNum(), 164);
    EXPECT_EQ(graphTester.GetDelegatedPartitionNum(), 0);
    EXPECT_EQ(graphTester.GetTotalPartitionNum(), 1);
    EXPECT_FALSE(graphTester.IsDelegated());

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    EXPECT_EQ(graphTester.GetTotalNodeNum(), 1);
    EXPECT_EQ(graphTester.GetDelegatedPartitionNum(), 1);
    EXPECT_EQ(graphTester.GetTotalPartitionNum(), 1);

    EXPECT_TRUE(graphTester.FillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

#ifndef USE_HOST_TEST
TEST_F(GraphTesterTest, 02_graphTester_fdshort)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_EQ(graphTester.GetTotalNodeNum(), 164);
    EXPECT_FALSE(graphTester.IsDelegated());

    APM apm;
    EXPECT_TRUE(apm.SetPolicy(APM::kEnableLoadBalancing));
    EXPECT_TRUE(apm.SetCPUFallbackPercentage(25));
    ADS ads;
    EXPECT_TRUE(ads.SelectDelegate(*interpreter.get(), &apm));

    EXPECT_EQ(graphTester.GetTotalNodeNum(), 97);
    EXPECT_EQ(graphTester.GetDelegatedPartitionNum(), 1);
    EXPECT_EQ(graphTester.GetTotalPartitionNum(), 2);
    EXPECT_TRUE(graphTester.IsDelegated());

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    EXPECT_EQ(graphTester.GetTotalNodeNum(), 2);
    EXPECT_EQ(graphTester.GetDelegatedPartitionNum(), 2);
    EXPECT_EQ(graphTester.GetTotalPartitionNum(), 2);

    EXPECT_TRUE(graphTester.FillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}
#endif
