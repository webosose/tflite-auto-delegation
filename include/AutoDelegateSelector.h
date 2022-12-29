/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
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

#include "AccelerationPolicyManager.h"

namespace aif
{
    class AutoDelegateSelector
    {
    public:
        AutoDelegateSelector();
        virtual ~AutoDelegateSelector() = default;
        bool SelectDelegate(tflite::Interpreter &interpreter, AccelerationPolicyManager *apm);

    private:
        bool SetWebOSNPUDelegate(tflite::Interpreter &interpreter);
        bool SetTfLiteGPUDelegate(tflite::Interpreter &interpreter, AccelerationPolicyManager *apm);
#ifdef USE_EDGETPU
        bool SetEdgeTPUDelegate(tflite::Interpreter &interpreter);
        const std::string edgetpu_lib_path_ = "/usr/lib/libedgetpu.so.1";
#endif
    };
} // end of namespace aif
#endif
