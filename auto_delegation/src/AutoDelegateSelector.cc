/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#include "AutoDelegateSelector.h"
#include "tools/Utils.h"

namespace
{

static PmLogContext s_pmlogCtx = aif::getADPmLogContext();

} // end of anonymous namespace

namespace aif {

AutoDelegateSelector::AutoDelegateSelector(tflite::ops::builtin::BuiltinOpResolver *resolver)
    : resolver_(resolver)
{
  // add custom webOSNPU operation to resolver
}

AutoDelegateSelector::~AutoDelegateSelector()
{
}

bool AutoDelegateSelector::SelectDelegate(std::unique_ptr<tflite::Interpreter> *interpreter, AccelerationPolicyManager *apm)
{
  if (apm->GetPolicy() != AccelerationPolicyManager::kEnableLoadBalancing && apm->GetCPUFallbackPercentage() != 0)
  {
    PmLogInfo(s_pmlogCtx, "ADS", 0, "current policy is not EnableLoadBalancing but non-zero value"
                                    "is set to cpu fallback percentage. So, the value set for cpu fallback percentage is ignored.");
  }

  tflite::Subgraph &subgraph = (*interpreter)->primary_subgraph();

  auto plan = subgraph.execution_plan();
  auto nodes = subgraph.nodes_and_registration();
  for (int i = 0; i < plan.size(); i++)
  {
    auto idx = plan[i];
    auto registration = nodes[idx].second;

    auto op = static_cast<tflite::BuiltinOperator>(registration.builtin_code);
    if (registration.custom_name != nullptr)
    {
      // check if the model is NPU compiled
      if (strcmp(registration.custom_name, "webosnpu-custom-op") == 0)
      {
        if (SetWebOSNPUDelegate(interpreter, apm) == true)
        {
          break;
        }
        else
        {
          PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting webOSNPU delegate");
          return false;
        }
      }
    }
  }

  if (apm->GetPolicy() != AccelerationPolicyManager::kCPUOnly)
    return SetTfLiteGPUDelegate(interpreter, apm);
  else
    return true;
}

bool AutoDelegateSelector::SetWebOSNPUDelegate(std::unique_ptr<tflite::Interpreter> *interpreter, AccelerationPolicyManager *apm)
{
  return true;
}

bool AutoDelegateSelector::SetTfLiteGPUDelegate(std::unique_ptr<tflite::Interpreter> *interpreter, AccelerationPolicyManager *apm)
{
  auto policy = apm->GetPolicy();
  TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  if (policy == AccelerationPolicyManager::kMinimumLatency)
  {
    gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
  }
  else if (policy == AccelerationPolicyManager::kEnableLoadBalancing)
  {
    gpu_opts.cpu_fallback_percentage = apm->GetCPUFallbackPercentage();
    gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
    gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
  }

  gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;

  auto *delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
  if ((*interpreter)->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
  {
    PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting TfLiteGPU delegate");
    return false;
  }

  return true;
}

bool AutoDelegateSelector::Preview(std::unique_ptr<tflite::Interpreter> *interpreter)
{
  int num_partition = 0;
  tflite::Subgraph &subgraph = (*interpreter)->primary_subgraph();

  auto plan = subgraph.execution_plan();
  auto nodes = subgraph.nodes_and_registration();
  std::cout << "------------------------------------------------------------\n\n";
  auto plan_size = plan.size();
  for (int i = 0; i < plan_size; i++)
  {
    auto idx = plan[i];
    auto node = nodes[idx].first;
    auto registration = nodes[idx].second;

    auto op = static_cast<tflite::BuiltinOperator>(registration.builtin_code);
    if (strcmp(tflite::EnumNameBuiltinOperator(op), "DELEGATE") == 0)
      num_partition++;

    std::cout << idx << " <" << tflite::EnumNameBuiltinOperator(op) << ">  ";
    if (registration.custom_name != nullptr)
      std::cout << registration.custom_name << "\n\n";
    else if (node.delegate != nullptr)
      std::cout << "NO delegate or custom\n\n";

    auto inputs = node.inputs;
    std::cout << "INPUTS:\n";
    for (int j = 0; j < inputs->size; j++)
    {
      const TfLiteTensor &t = *((*interpreter)->tensor(j));
      std::cout << t.name << " (" << TfLiteTypeGetName(t.type) << ", " << inputs->data[j] << ")\n";
      TfLiteTensor *tensor_input = (*interpreter)->tensor(inputs->data[j]);
      auto dim_size = tensor_input->dims->size;
      for (int k = 0; k < dim_size; k++)
      {
        std::cout << tensor_input->dims->data[k] << " ";
      }
      std::cout << "\n";
      std::cout << t.quantization.type << "\n";
    }

    std::cout << "\nOUTPUTS:\n";
    auto outputs = node.outputs;
    for (int j = 0; j < outputs->size; j++)
    {
      const TfLiteTensor &t = *((*interpreter)->tensor(j));
      std::cout << t.name << " (" << TfLiteTypeGetName(t.type) << ", " << outputs->data[j] << ")\n";
    }
    std::cout << "\n------------------------------------------------------------\n\n";
  }

  const std::vector<int> &t_inputs = (*interpreter)->inputs();
  auto t_inputs_size = t_inputs.size();
  std::cout << "* Interpreter Input Size : " << t_inputs_size << "\n";
  for (int i = 0; i < t_inputs_size; i++)
  {
    TfLiteTensor *tensor_input = (*interpreter)->tensor(t_inputs[i]);
    std::cout << "\tInput Tensor " << i << " (" << (*interpreter)->GetInputName(i) << ") : [ " << tensor_input->dims->data[0];
    for (int j = 1; j < tensor_input->dims->size; j++)
    {
      std::cout << " x " << tensor_input->dims->data[j];
    }
    std::cout << "]\n";
  }

  const std::vector<int> &t_outputs = (*interpreter)->outputs();
  auto t_outputs_size = t_outputs.size();
  std::cout << "* Interpreter Output Size : " << t_outputs_size << "\n";
  for (int i = 0; i < t_outputs_size; i++)
  {
    TfLiteTensor *tensor_output = (*interpreter)->tensor(t_outputs[0]);
    std::cout << "\tOutput Tensor " << i << " (" << (*interpreter)->GetOutputName(i) << ") : [ " << tensor_output->dims->data[0];
    for (int j = 1; j < tensor_output->dims->size; j++)
    {
      std::cout << " x " << tensor_output->dims->data[j];
    }
    std::cout << "]\n";
  }
  return true;
}

std::random_device rd;
std::mt19937 gen(rd());

bool AutoDelegateSelector::FillRandomInputTensor(std::unique_ptr<tflite::Interpreter> *interpreter)
{
  tflite::Subgraph &subgraph = (*interpreter)->primary_subgraph();
  auto plan = subgraph.execution_plan();
  auto nodes = subgraph.nodes_and_registration();

  auto plan0 = plan[0];
  auto node = nodes[plan0].first;
  auto inputs = node.inputs;
  const TfLiteTensor &t = *((*interpreter)->tensor(0)); // how about input size > 1 ??
  TfLiteType model_input_tensor_type = t.type;

  const std::vector<int> &t_inputs = (*interpreter)->inputs();
  TfLiteTensor *tensor_input = (*interpreter)->tensor(t_inputs[0]);
  int input_dims = tensor_input->dims->size;
  int input_size = 1;
  for (int i = 0; i < input_dims; i++)
  {
    input_size *= tensor_input->dims->data[i];
  }

  for (int i = 0; i < input_size; i++)
  {
    std::uniform_int_distribution<> dist(0, 256);
    int rand_pixel = dist(gen);
    switch (model_input_tensor_type)
    {
    case kTfLiteFloat32:
      (*interpreter)->typed_input_tensor<float>(0)[i] = static_cast<float>(rand_pixel) / static_cast<float>(256);
      break;
    case kTfLiteInt32:
      (*interpreter)->typed_input_tensor<int32_t>(0)[i] = rand_pixel;
      break;
    case kTfLiteUInt8:
      (*interpreter)->typed_input_tensor<uint8_t>(0)[i] = rand_pixel;
      break;
    case kTfLiteInt64:
      (*interpreter)->typed_input_tensor<int64_t>(0)[i] = rand_pixel;
      break;
    case kTfLiteInt16:
      (*interpreter)->typed_input_tensor<int16_t>(0)[i] = rand_pixel;
      break;
    case kTfLiteInt8:
      (*interpreter)->typed_input_tensor<int8_t>(0)[i] = rand_pixel;
      break;
    case kTfLiteFloat64:
      (*interpreter)->typed_input_tensor<double>(0)[i] = static_cast<double>(rand_pixel) / static_cast<float>(256);
      break;
    case kTfLiteUInt64:
      (*interpreter)->typed_input_tensor<uint64_t>(0)[i] = rand_pixel;
      break;
    case kTfLiteUInt32:
      (*interpreter)->typed_input_tensor<uint32_t>(0)[i] = rand_pixel;
      break;
    default:
      break;
    }
  }
  return true;
}

} // end of namespace aif
