#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456)

#include "ONNXStatus.h"

namespace CommonIR
{
    Status::Status(bool p_ok, const std::string& p_errMsg)
    {
        m_ok = p_ok;
        m_errMsg = p_errMsg;
    }

    Status::Status(const Status& p_other)
    {
        m_ok = p_other.m_ok;
        m_errMsg = p_other.m_errMsg;
    }

    bool Status::Ok() const
    {
        return m_ok;
    }

    const std::string& Status::ErrorMsg() const
    {
        return m_errMsg;
    }

    Status Status::OK()
    {
        static Status ok(true, "");
        return ok;
    }
}

#pragma warning(pop)