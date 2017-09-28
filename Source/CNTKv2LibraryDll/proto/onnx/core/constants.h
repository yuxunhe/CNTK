#include <string>
#include <unordered_set>

namespace LotusIR
{
    static const std::string c_constantOp = "Constant";
    static const std::string c_constantValue = "CONSTANT_VALUE";

    // DataType strings. These should match the DataTypes defined in Data.proto
    static const std::string c_float16 = "float16";
    static const std::string c_float = "float";
    static const std::string c_double = "double";
    static const std::string c_int8 = "int8";
    static const std::string c_int16 = "int16";
    static const std::string c_int32 = "int32";
    static const std::string c_int64 = "int64";
    static const std::string c_uint8 = "uint8";
    static const std::string c_uint16 = "uint16";
    static const std::string c_uint32 = "uint32";
    static const std::string c_uint64 = "uint64";
    static const std::string c_complex64 = "complex64";
    static const std::string c_complex128 = "complex128";
    static const std::string c_string = "string";
    static const std::string c_bool = "bool";

    static std::unordered_set<std::string> s_allowedDataTypes = {
        c_float16, c_float, c_double,
        c_int8, c_int16, c_int32, c_int64,
        c_uint8, c_uint16, c_uint32, c_uint64,
        c_complex64, c_complex128,
        c_string, c_bool
    };
}