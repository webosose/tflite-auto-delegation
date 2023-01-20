/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tools/Logger.h>

namespace aif
{
    PmLogContext getADPmLogContext()
    {
        static PmLogContext logContext = nullptr;
        if (logContext == nullptr)
        {
            PmLogGetContext("auto_delegation", &logContext);
        }
        return logContext;
    }
} // end of aif namespace
