#ifndef AUTODELEGATESELECTOR_H_
#define AUTODELEGATESELECTOR_H_
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>

#include "AccelerationPolicyManager.h"

class AutoDelegateSelector
{
public:
    AutoDelegateSelector(tflite::ops::builtin::BuiltinOpResolver *resolver,
                         std::unique_ptr<tflite::Interpreter> *interpreter,
                         AccelerationPolicyManager *accelerationPolicyManager);

    virtual ~AutoDelegateSelector();

    bool SelectDelegate();
    bool Preview();
    bool FillRandomInputTensor();

private:
    bool SetWebOSNPUDelegate();
    bool SetTfLiteGPUDelegate();

    tflite::ops::builtin::BuiltinOpResolver *resolver_;
    std::unique_ptr<tflite::Interpreter> *interpreter_;
    AccelerationPolicyManager *accelerationPolicyManager_;
};

#endif