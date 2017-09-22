#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456)

#include <cctype>
#include <iterator>
#include <iostream>
#include <sstream>

#include "../protobuf/Data.pb.h"
#include "../protobuf/graph.pb.h"
#include "../protobuf/Type.pb.h"
#include "utils.h"


namespace LotusIR
{
    namespace Utils
    {
        std::unordered_map<std::string, TypeProto> OpUtils::s_typeStrToProtoMap;

        PTYPE OpUtils::ToType(const TypeProto& p_type)
        {
            auto typeStr = ToString(p_type);
            if (s_typeStrToProtoMap.find(typeStr) == s_typeStrToProtoMap.end())
            {
                s_typeStrToProtoMap[typeStr] = p_type;
            }
            return &(s_typeStrToProtoMap.find(typeStr)->first);
        }

        PTYPE OpUtils::ToType(const std::string& p_type)
        {
            TypeProto type;
            FromString(p_type, type);
            return ToType(type);
        }

        const TypeProto& OpUtils::ToTypeProto(const PTYPE& p_type)
        {
            auto it = s_typeStrToProtoMap.find(*p_type);
            if (it != s_typeStrToProtoMap.end())
            {
                return it->second;
            }
            else
            {
                throw std::invalid_argument("PTYPE not found: " + *p_type);
            }
        }

        std::string OpUtils::ToString(const TypeProto& p_type)
        {
            switch (p_type.value_case())
            {
            case TypeProto::ValueCase::kTensorType:
                return ToString(p_type.tensor_type().elem_type());
            case TypeProto::ValueCase::kSparseTensorType:
                return "sparse(" + ToString(p_type.sparse_tensor_type().elem_type()) + ")";
            case TypeProto::ValueCase::kSeqType:
                return "list(" + ToString(p_type.seq_type().elem_type()) + ")";
            case TypeProto::ValueCase::kTupleType:
            {
                int size = p_type.tuple_type().elem_type_size();
                std::string tuple_str("tuple(");
                for (int i = 0; i < size - 1; i++)
                {
                    tuple_str = tuple_str + ToString(p_type.tuple_type().elem_type(i)) + ",";
                }
                tuple_str += ToString(p_type.tuple_type().elem_type(size - 1));
                tuple_str += ")";
                return tuple_str;
            }
            case TypeProto::ValueCase::kMapType:
            {
                std::string map_str("map(");
                map_str = map_str + ToString(p_type.map_type().key_type()) + ","
                    + ToString(p_type.map_type().value_type()) + ")";
                return map_str;
            }
            case TypeProto::ValueCase::kHandleType:
                return "handle";
            }
            return "";
        }

        std::string OpUtils::ToString(const TensorProto::DataType& p_type)
        {
            switch (p_type)
            {
            case TensorProto::DataType::TensorProto_DataType_BOOL:
                return "bool";
            case TensorProto::DataType::TensorProto_DataType_FLOAT:
                return "float32";
            case TensorProto::DataType::TensorProto_DataType_FLOAT16:
                return "float16";
            case TensorProto::DataType::TensorProto_DataType_DOUBLE:
                return "float64";
            case TensorProto::DataType::TensorProto_DataType_INT16:
                return "int16";
            case TensorProto::DataType::TensorProto_DataType_INT32:
                return "int32";
            case TensorProto::DataType::TensorProto_DataType_INT64:
                return "int64";
            case TensorProto::DataType::TensorProto_DataType_INT8:
                return "int8";
            case TensorProto::DataType::TensorProto_DataType_STRING:
                return "string";
            case TensorProto::DataType::TensorProto_DataType_UINT16:
                return "uint16";
            }
            return "";
        }


        void OpUtils::FromString(const std::string& p_src, TypeProto& p_type)
        {
            StringRange s(p_src);
            s.LAndRStrip();
            p_type.Clear();

            if (s.LStrip("list"))
            {
                // TODO
            }
            else if (s.LStrip("tuple"))
            {
                // TODO
            }
            else if (s.LStrip("map("))
            {
                // TODO
            }
            else if (s.LStrip("handle"))
            {
                p_type.mutable_handle_type();
            }
            else if (s.LStrip("sparse("))
            {
                // sparse tensor
                s.RStrip(")");
                TensorProto::DataType e;
                FromString(std::string(s.Data(), s.Size()), e);
                p_type.mutable_sparse_tensor_type()->set_elem_type(e);
            }
            else
            {
                // dense tensor
                TensorProto::DataType e;
                FromString(std::string(s.Data(), s.Size()), e);
                p_type.mutable_tensor_type()->set_elem_type(e);
            }
        }

        void OpUtils::FromString(const std::string& p_typeStr, TensorProto::DataType& p_type)
        {
            if (p_typeStr == "bool")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_BOOL;
            }
            else if (p_typeStr == "float32")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_FLOAT;
            }
            else if (p_typeStr == "float16")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_FLOAT16;
            }
            else if (p_typeStr == "float64")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_DOUBLE;
            }
            else if (p_typeStr == "int16")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_INT16;
            }
            else if (p_typeStr == "int32")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_INT32;
            }
            else if (p_typeStr == "int64")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_INT64;
            }
            else if (p_typeStr == "int8")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_INT8;
            }
            else if (p_typeStr == "string")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_STRING;
            }
            else if (p_typeStr == "uint16")
            {
                p_type = TensorProto::DataType::TensorProto_DataType_UINT16;
            }
            else
            {
                p_type = TensorProto::DataType::TensorProto_DataType_UNDEFINED;
            }
        }

        StringRange::StringRange()
            : m_data(""), m_size(0)
        {}

        StringRange::StringRange(const char* p_data, size_t p_size)
            : m_data(p_data), m_size(p_size)
        {}

        StringRange::StringRange(const std::string& p_str)
            : m_data(p_str.data()), m_size(p_str.size())
        {}

        StringRange::StringRange(const char* p_data)
            : m_data(p_data), m_size(strlen(p_data))
        {}

        const char* StringRange::Data() const
        {
            return m_data;
        }

        size_t StringRange::Size() const
        {
            return m_size;
        }

        bool StringRange::Empty() const
        {
            return m_size == 0;
        }

        char StringRange::operator[](size_t p_idx) const
        {
            return m_data[p_idx];
        }

        void StringRange::Reset()
        {
            m_data = "";
            m_size = 0;
        }

        void StringRange::Reset(const char* p_data, size_t p_size)
        {
            m_data = p_data;
            m_size = p_size;
        }

        void StringRange::Reset(const std::string& p_str)
        {
            m_data = p_str.data();
            m_size = p_str.size();
        }

        bool StringRange::StartsWith(const StringRange& p_str) const
        {
            return ((m_size >= p_str.m_size) && (memcmp(m_data, p_str.m_data, p_str.m_size) == 0));
        }

        bool StringRange::EndsWith(const StringRange& p_str) const
        {
            return ((m_size >= p_str.m_size) &&
                (memcmp(m_data + (m_size - p_str.m_size), p_str.m_data, p_str.m_size) == 0));
        }

        bool StringRange::LStrip() {
            size_t count = 0;
            const char* ptr = m_data;
            while (count < m_size && isspace(*ptr)) {
                count++;
                ptr++;
            }

            if (count > 0)
            {
                return LStrip(count);
            }
            return false;
        }
        bool StringRange::LStrip(size_t p_size)
        {
            if (p_size <= m_size)
            {
                m_data += p_size;
                m_size -= p_size;
                return true;
            }
            return false;
        }

        bool StringRange::LStrip(StringRange p_str)
        {
            if (StartsWith(p_str)) {
                return LStrip(p_str.m_size);
            }
            return false;
        }

        bool StringRange::RStrip() {
            size_t count = 0;
            const char* ptr = m_data + m_size - 1;
            while (count < m_size && isspace(*ptr)) {
                ++count;
                --ptr;
            }

            if (count > 0)
            {
                return RStrip(count);
            }
            return false;
        }

        bool StringRange::RStrip(size_t p_size)
        {
            if (m_size >= p_size)
            {
                m_size -= p_size;
                return true;
            }
            return false;
        }

        bool StringRange::RStrip(StringRange p_str)
        {
            if (EndsWith(p_str)) {
                return RStrip(p_str.m_size);
            }
            return false;
        }

        bool StringRange::LAndRStrip()
        {
            return LStrip() || RStrip();
        }

        size_t StringRange::Find(const char p_ch) const
        {
            size_t idx = 0;
            while (idx < m_size)
            {
                if (m_data[idx] == p_ch)
                {
                    return idx;
                }
                idx++;
            }
            return std::string::npos;
        }
    }
}

#pragma warning(pop)