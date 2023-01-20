/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: Apache-2.0
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

    GraphTester graphTester(*interpreter.get());
    EXPECT_EQ(graphTester.getTotalNodeNum(), 164);
    EXPECT_FALSE(graphTester.isDelegated());

    APM apm;
    EXPECT_TRUE(apm.setPolicy(APM::kCPUOnly));
    ADS ads;
    EXPECT_TRUE(ads.selectDelegate(*interpreter.get(), apm));

    EXPECT_EQ(graphTester.getTotalNodeNum(), 164);
    EXPECT_EQ(graphTester.getDelegatedPartitionNum(), 0);
    EXPECT_EQ(graphTester.getTotalPartitionNum(), 1);
    EXPECT_FALSE(graphTester.isDelegated());

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    EXPECT_EQ(graphTester.getTotalNodeNum(), 1);
    EXPECT_EQ(graphTester.getDelegatedPartitionNum(), 1);
    EXPECT_EQ(graphTester.getTotalPartitionNum(), 1);

    EXPECT_TRUE(graphTester.fillRandomInputTensor());

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

    GraphTester graphTester(*interpreter.get());
    EXPECT_EQ(graphTester.getTotalNodeNum(), 164);
    EXPECT_FALSE(graphTester.isDelegated());

    APM apm;
    EXPECT_TRUE(apm.setPolicy(APM::kEnableLoadBalancing));
    EXPECT_TRUE(apm.setCPUFallbackPercentage(25));
    ADS ads;
    EXPECT_TRUE(ads.selectDelegate(*interpreter.get(), apm));

    EXPECT_EQ(graphTester.getTotalNodeNum(), 97);
    EXPECT_EQ(graphTester.getDelegatedPartitionNum(), 1);
    EXPECT_EQ(graphTester.getTotalPartitionNum(), 2);
    EXPECT_TRUE(graphTester.isDelegated());

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    EXPECT_EQ(graphTester.getTotalNodeNum(), 2);
    EXPECT_EQ(graphTester.getDelegatedPartitionNum(), 2);
    EXPECT_EQ(graphTester.getTotalPartitionNum(), 2);

    EXPECT_TRUE(graphTester.fillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}
#endif
