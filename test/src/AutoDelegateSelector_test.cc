/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#include <gtest/gtest.h>
#include <AutoDelegateSelector.h>

// custom ops
#include <customOp/transpose_conv_bias.h>
#include <customOp/posenet_decoder_op.h>
#include <customOp/posenet_decoder.h>

using namespace aif;

typedef AutoDelegateSelector ADS;
typedef AccelerationPolicyManager APM;

class AutoDelegateSelectorTest : public ::testing::Test
{
protected:
    AutoDelegateSelectorTest() = default;
    ~AutoDelegateSelectorTest() = default;

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

TEST_F(AutoDelegateSelectorTest, 01_01_selectDelegate_yunet_CPUOnly)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kCPUOnly));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 01_02_selectDelegate_yunet_MaximumPrecision)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kMaximumPrecision));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 01_03_selectDelegate_yunet_MinimumLatency)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kMinimumLatency));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 01_04_selectDelegate_yunet_EnableLoadBalancing)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kEnableLoadBalancing));
    EXPECT_TRUE(apm.SetCPUFallbackPercentage(25));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 01_05_selectDelegate_yunet_EnableLoadBalancing)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    std::string config(
        "{\n"
        "    \"policy\" : \"LOAD_BALANCING\",\n"
        "    \"cpu_fallback_percentage\": 25\n"
        "}");
    APM apm(config);
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 25);
    EXPECT_EQ(apm.GetPolicy(), APM::kEnableLoadBalancing);

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 02_01_selectDelegate_fdshort_CPUOnly)
{
    std::string model_path = model_paths[1];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kCPUOnly));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 02_02_selectDelegate_fdshort_MaximumPrecision)
{
    std::string model_path = model_paths[1];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kMaximumPrecision));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 02_03_selectDelegate_fdshort_MinimumLatency)
{
    std::string model_path = model_paths[1];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kMinimumLatency));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 02_04_selectDelegate_fdshort_EnableLoadBalancing)
{
    std::string model_path = model_paths[1];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kEnableLoadBalancing));
    EXPECT_TRUE(apm.SetCPUFallbackPercentage(15));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 02_05_selectDelegate_fdshort_EnableLoadBalancing)
{
    std::string model_path = model_paths[1];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    std::string config(
        "{\n"
        "    \"policy\" : \"LOAD_BALANCING\",\n"
        "    \"cpu_fallback_percentage\": 30\n"
        "}");
    APM apm(config);
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 30);
    EXPECT_EQ(apm.GetPolicy(), APM::kEnableLoadBalancing);

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 03_01_selectDelegate_posenet_CPUOnly)
{
    std::string model_path = model_paths[2];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kCPUOnly));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 03_02_selectDelegate_posenet_MaximumPrecision)
{
    std::string model_path = model_paths[2];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kMaximumPrecision));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 03_03_selectDelegate_posenet_MinimumLatency)
{
    std::string model_path = model_paths[2];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kMinimumLatency));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 03_04_selectDelegate_posenet_EnableLoadBalancing)
{
    std::string model_path = model_paths[2];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kEnableLoadBalancing));
    EXPECT_TRUE(apm.SetCPUFallbackPercentage(25));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 03_05_selectDelegate_posenet_EnableLoadBalancing)
{
    std::string model_path = model_paths[2];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(coral::kPosenetDecoderOp, coral::RegisterPosenetDecoderOp());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    std::string config(
        "{\n"
        "    \"policy\" : \"LOAD_BALANCING\",\n"
        "    \"cpu_fallback_percentage\": 25\n"
        "}");
    APM apm(config);
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 25);
    EXPECT_EQ(apm.GetPolicy(), APM::kEnableLoadBalancing);

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 04_01_selectDelegate_selfiesegmentation_CPUOnly)
{
    std::string model_path = model_paths[3];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kCPUOnly));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 04_02_selectDelegate_selfiesegmentation_MaximumPrecision)
{
    std::string model_path = model_paths[3];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kMaximumPrecision));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 04_03_selectDelegate_selfiesegmentation_MinimumLatency)
{
    std::string model_path = model_paths[3];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kMinimumLatency));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 04_04_selectDelegate_selfiesegmentation_EnableLoadBalancing)
{
    std::string model_path = model_paths[3];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    AccelerationPolicyManager apm;
    EXPECT_TRUE(apm.SetPolicy(AccelerationPolicyManager::kEnableLoadBalancing));
    EXPECT_TRUE(apm.SetCPUFallbackPercentage(10));

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 04_05_selectDelegate_selfiesegmentation_EnableLoadBalancing)
{
    std::string model_path = model_paths[3];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom("Convolution2DTransposeBias", mediapipe::tflite_operations::RegisterConvolution2DTransposeBias());

    ADS ads(&resolver);

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    std::string config(
        "{\n"
        "    \"policy\" : \"LOAD_BALANCING\",\n"
        "    \"cpu_fallback_percentage\": 15\n"
        "}");
    APM apm(config);
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 15);
    EXPECT_EQ(apm.GetPolicy(), APM::kEnableLoadBalancing);

    EXPECT_TRUE(ads.SelectDelegate(&interpreter, &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    EXPECT_TRUE(ads.FillRandomInputTensor(&interpreter));
    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}
