/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#include "tools/GraphTester.h"

namespace aif
{
    GraphTester::GraphTester(std::unique_ptr<tflite::Interpreter> *interpreter)
        : interpreter_(interpreter)
    {
    }

    GraphTester::~GraphTester()
    {
    }

    int GraphTester::GetTotalNodeNum()
    {
        tflite::Subgraph &subgraph = (*interpreter_)->primary_subgraph();
        return subgraph.execution_plan().size();
    }

    int GraphTester::GetTotalPartitionNum()
    {
        int num_partition = 0;
        tflite::Subgraph &subgraph = (*interpreter_)->primary_subgraph();
        auto plan = subgraph.execution_plan();
        auto nodes = subgraph.nodes_and_registration();
        auto plan_size = plan.size();
        bool is_prev_partition_delegated = false;
        for (int i = 0; i < plan_size; i++)
        {
            auto idx = plan[i];
            auto node = nodes[idx].first;
            auto registration = nodes[idx].second;
            bool is_cur_partition_delegated = false;
            auto op = static_cast<tflite::BuiltinOperator>(registration.builtin_code);
            if (strcmp(tflite::EnumNameBuiltinOperator(op), "DELEGATE") == 0)
            {
                num_partition++;
                is_cur_partition_delegated = true;
            }
            if (is_prev_partition_delegated == true && is_cur_partition_delegated == false)
                num_partition++;
            if (i == 0 && is_cur_partition_delegated == false)
                num_partition++;
            is_prev_partition_delegated = is_cur_partition_delegated;
        }
        return num_partition;
    }

    int GraphTester::GetDelegatedPartitionNum()
    {
        int num_delegated_partition = 0;
        tflite::Subgraph &subgraph = (*interpreter_)->primary_subgraph();
        auto plan = subgraph.execution_plan();
        auto nodes = subgraph.nodes_and_registration();
        auto plan_size = plan.size();
        for (int i = 0; i < plan_size; i++)
        {
            auto idx = plan[i];
            auto node = nodes[idx].first;
            auto registration = nodes[idx].second;
            auto op = static_cast<tflite::BuiltinOperator>(registration.builtin_code);
            if (strcmp(tflite::EnumNameBuiltinOperator(op), "DELEGATE") == 0)
                num_delegated_partition++;
        }
        return num_delegated_partition;
    }

    bool GraphTester::IsDelegated()
    {
        return GraphTester::GetDelegatedPartitionNum() > 0 ? true : false;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    bool GraphTester::FillRandomInputTensor()
    {
        tflite::Subgraph &subgraph = (*interpreter_)->primary_subgraph();
        auto plan = subgraph.execution_plan();
        auto nodes = subgraph.nodes_and_registration();
        auto plan0 = plan[0];
        auto node = nodes[plan0].first;
        auto inputs = node.inputs;
        const std::vector<int> &t_inputs = (*interpreter_)->inputs();
        TfLiteTensor *tensor_input = (*interpreter_)->tensor(t_inputs[0]);
        TfLiteType model_input_tensor_type = tensor_input->type;
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
                (*interpreter_)->typed_input_tensor<float>(0)[i] = static_cast<float>(rand_pixel) / static_cast<float>(256);
                break;
            case kTfLiteInt32:
                (*interpreter_)->typed_input_tensor<int32_t>(0)[i] = rand_pixel;
                break;
            case kTfLiteUInt8:
                (*interpreter_)->typed_input_tensor<uint8_t>(0)[i] = rand_pixel;
                break;
            case kTfLiteInt64:
                (*interpreter_)->typed_input_tensor<int64_t>(0)[i] = rand_pixel;
                break;
            case kTfLiteInt16:
                (*interpreter_)->typed_input_tensor<int16_t>(0)[i] = rand_pixel;
                break;
            case kTfLiteInt8:
                (*interpreter_)->typed_input_tensor<int8_t>(0)[i] = rand_pixel;
                break;
            case kTfLiteFloat64:
                (*interpreter_)->typed_input_tensor<double>(0)[i] = static_cast<double>(rand_pixel) / static_cast<float>(256);
                break;
            case kTfLiteUInt64:
                (*interpreter_)->typed_input_tensor<uint64_t>(0)[i] = rand_pixel;
                break;
            case kTfLiteUInt32:
                (*interpreter_)->typed_input_tensor<uint32_t>(0)[i] = rand_pixel;
                break;
            default:
                break;
            }
        }
        return true;
    }
} // end of namespace aif
