/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: Apache-2.0
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
        GraphTester(tflite::Interpreter &interpreter);
        virtual ~GraphTester();

        int getTotalNodeNum();
        int getTotalPartitionNum();
        int getDelegatedPartitionNum();
        bool isDelegated();

        bool fillRandomInputTensor();

    private:
        tflite::Interpreter &m_interpreter;
    };
} // end of namespace aif

#endif
