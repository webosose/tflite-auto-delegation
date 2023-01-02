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
        std::string(AIF_INSTALL_DIR) + std::string("/model/face_detection_short_range.tflite"),
        std::string(AIF_INSTALL_DIR) + std::string("/model/face-detector-quantized_edgetpu.tflite")};
};

TEST_F(AutoDelegateSelectorTest, 01_01_selectDelegate_fdshort_CPUOnly)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    APM apm;
    EXPECT_TRUE(apm.SetPolicy(APM::kCPUOnly));
    ADS ads;
    EXPECT_TRUE(ads.SelectDelegate(*interpreter.get(), &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_TRUE(graphTester.FillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

#ifndef USE_HOST_TEST
TEST_F(AutoDelegateSelectorTest, 01_02_selectDelegate_fdshort_MaximumPrecision)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    APM apm;
    EXPECT_TRUE(apm.SetPolicy(APM::kMaximumPrecision));
    ADS ads;
    EXPECT_TRUE(ads.SelectDelegate(*interpreter.get(), &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_TRUE(graphTester.FillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 01_03_selectDelegate_fdshort_MinimumLatency)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    APM apm;
    EXPECT_TRUE(apm.SetPolicy(APM::kMinimumLatency));
    ADS ads;
    EXPECT_TRUE(ads.SelectDelegate(*interpreter.get(), &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_TRUE(graphTester.FillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 01_04_selectDelegate_fdshort_EnableLoadBalancing)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    APM apm;
    EXPECT_TRUE(apm.SetPolicy(APM::kEnableLoadBalancing));
    EXPECT_TRUE(apm.SetCPUFallbackPercentage(15));
    ADS ads;
    EXPECT_TRUE(ads.SelectDelegate(*interpreter.get(), &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_TRUE(graphTester.FillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}

TEST_F(AutoDelegateSelectorTest, 01_05_selectDelegate_fdshort_EnableLoadBalancing)
{
    std::string model_path = model_paths[0];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    std::string config(
        "{\n"
        "    \"policy\" : \"LOAD_BALANCING\",\n"
        "    \"cpu_fallback_percentage\": 30\n"
        "}");
    APM apm(config);
    EXPECT_EQ(apm.GetCPUFallbackPercentage(), 30);
    EXPECT_EQ(apm.GetPolicy(), APM::kEnableLoadBalancing);
    ADS ads;
    EXPECT_TRUE(ads.SelectDelegate(*interpreter.get(), &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_TRUE(graphTester.FillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
}
#endif

#ifndef USE_HOST_TEST
#ifdef USE_EDGETPU
TEST_F(AutoDelegateSelectorTest, 02_01_edgetpu_test)
{
    std::string model_path = model_paths[1];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    APM apm;
    EXPECT_TRUE(apm.SetPolicy(APM::kCPUOnly));
    ADS ads;
    EXPECT_TRUE(ads.SelectDelegate(*interpreter.get(), &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_TRUE(graphTester.FillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);

    interpreter.reset();
}

TEST_F(AutoDelegateSelectorTest, 02_02_edgetpu_test)
{
    std::string model_path = model_paths[1];
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    EXPECT_EQ(tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter), kTfLiteOk);

    APM apm;
    EXPECT_TRUE(apm.SetPolicy(APM::kMinimumLatency));
    ADS ads;
    EXPECT_TRUE(ads.SelectDelegate(*interpreter.get(), &apm));

    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    GraphTester graphTester(&interpreter);
    EXPECT_TRUE(graphTester.FillRandomInputTensor());

    EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);

    interpreter.reset();
}
#endif
#endif
