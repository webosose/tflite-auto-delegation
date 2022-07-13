/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#include <gtest/gtest.h>
#include <AutoDelegateSelector.h>

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
    };
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