/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#ifndef GRAPHTESTER_H_
#define GRAPHTESTER_H_

#include <iostream>
#include <cstring>
#include <random>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

namespace aif
{
    class GraphTester
    {
    public:
        GraphTester(std::unique_ptr<tflite::Interpreter> *interpreter);
        virtual ~GraphTester();

        int GetTotalNodeNum();
        int GetTotalPartitionNum();
        int GetDelegatedPartitionNum();
        bool IsDelegated();

        bool FillRandomInputTensor();

    private:
        std::unique_ptr<tflite::Interpreter> *interpreter_;
    };
} // end of namespace aif

#endif
