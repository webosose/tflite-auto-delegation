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
#include <edgetpu.h>
#endif

#include "AccelerationPolicyManager.h"

namespace aif
{
    class AutoDelegateSelector
    {
    public:
        AutoDelegateSelector(tflite::ops::builtin::BuiltinOpResolver *resolver);
        virtual ~AutoDelegateSelector();
        bool SelectDelegate(std::unique_ptr<tflite::Interpreter> *interpreter, AccelerationPolicyManager *apm);

    private:
        bool SetWebOSNPUDelegate(std::unique_ptr<tflite::Interpreter> *interpreter);
#ifdef USE_EDGETPU
        bool SetEdgeTPUDelegate(std::unique_ptr<tflite::Interpreter> *interpreter);
#endif
        bool SetTfLiteGPUDelegate(std::unique_ptr<tflite::Interpreter> *interpreter, AccelerationPolicyManager *apm);

        tflite::ops::builtin::BuiltinOpResolver *resolver_;
#ifdef USE_EDGETPU
        std::shared_ptr<edgetpu::EdgeTpuContext> edgetpuContext_;
#endif
    };
} // end of namespace aif
#endif
