/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef AUTODELEGATESELECTOR_H_
#define AUTODELEGATESELECTOR_H_
#include <iostream>
#include <vector>
#include <cstring>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>

#ifdef USE_EDGETPU
#include <tensorflow/lite/delegates/external/external_delegate.h>
#endif

#ifdef USE_NPU
#include <aif/npu/npu_delegate.h>
#endif

#include "AccelerationPolicyManager.h"

namespace aif
{
    class AutoDelegateSelector
    {
    public:
        AutoDelegateSelector();
        virtual ~AutoDelegateSelector() = default;
        bool selectDelegate(tflite::Interpreter &interpreter, AccelerationPolicyManager &apm);

    private:
#ifdef USE_NPU
        bool setWebOSNPUDelegate(tflite::Interpreter &interpreter);
#endif
        bool setTfLiteGPUDelegate(tflite::Interpreter &interpreter, AccelerationPolicyManager &apm);
#ifdef USE_EDGETPU
        bool setEdgeTPUDelegate(tflite::Interpreter &interpreter);
        const std::string EDGETPU_LIB_PATH = "/usr/lib/libedgetpu.so.1";
#endif
    };
} // end of namespace aif
#endif
