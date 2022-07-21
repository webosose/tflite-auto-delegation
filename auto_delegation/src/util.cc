#include <util.h>

PmLogContext getPmLogContext()
{
    static PmLogContext logContext = nullptr;
    if (logContext == nullptr)
    {
        PmLogGetContext("auto_delegation", &logContext);
    }
    return logContext;
}