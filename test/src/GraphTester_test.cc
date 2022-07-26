/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#include <gtest/gtest.h>
#include <AutoDelegateSelector.h>
#include <tools/GraphTester.h>

// custom ops
#include <customOp/transpose_conv_bias.h>
#include <customOp/posenet_decoder_op.h>
#include <customOp/posenet_decoder.h>

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
        "/usr/share/aif/model/face_yunet.tflite",
        "/usr/share/aif/model/face_detection_short_range.tflite",
        "/usr/share/aif/model/posenet_mobilenet_v1_075_353_481_quant_decoder.tflite",
        "/usr/share/aif/model/selfie_segmentation.tflite"};
};

TEST_F(GraphTesterTest, 01_graphTester_fdshort)
{
    std::string model_path = model_paths[1];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_EQ(graphTester.GetTotalNodeNum(), 164);
    EXPECT_FALSE(graphTester.IsDelegated());

    APM apm;
    EXPECT_TRUE(apm.SetPolicy(APM::kEnableLoadBalancing));
    EXPECT_TRUE(apm.SetCPUFallbackPercentage(25));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

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
