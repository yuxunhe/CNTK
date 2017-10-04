#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456)

#ifndef CORE_GRAPH_OP_H
#define CORE_GRAPH_OP_H

#include <functional>
#include <unordered_map>

#include "graph.h"
#include "utils.h"

namespace LotusIR
{
    class OperatorSchema;
    class OperatorSchemaSetter;
    typedef OperatorSchemaSetter OpSchema;

    enum class AttrType {
        NONE,
        FLOAT,
        INT,
        STRING,
        GRAPH,
        TENSOR,
        TYPE,
        SHAPE,
        FLOATS,
        INTS,
        STRINGS,
        GRAPHS,
        TENSORS,
        TYPES,
        SHAPES
    };

    // This string array should exactly match the AttrType defined above.
    static const std::string c_attrTypeStr[14] =
    {
        "FLOAT",
        "INT",
        "STRING",
        "GRAPH",
        "TENSOR",
        "TYPE",
        "SHAPE",
        "FLOATS",
        "INTS",
        "STRINGS",
        "GRAPHS",
        "TENSORS",
        "TYPES",
        "SHAPES"
    };

    class TypeUtils
    {
    public:

        // Get attribute type given attribute proto data.
        static Status GetType(const AttributeProto& p_attr, AttrType& p_type);

    };

    // A context to contain information for shape inference function.
    // It includes the operator registry, input arguments definition,
    // and mutable output arguments, whose shapes needs to be filled.
    class InferenceContext
    {
    public:

        // TODO: Add input tensors into constructor.
        // In some cases, node evaluation will be needed to get output shapes.
        InferenceContext(const Node* p_node,
            const OperatorSchema* p_opSchema,
            const std::vector<NodeArg>* p_inputs,
            std::vector<NodeArg>* p_outputs);

        const Node* GetNode() const;

        const OperatorSchema* GetOp() const;

        const std::vector<NodeArg>* GetInputs() const;

        std::vector<NodeArg>* Mutable_Outputs();

    private:

        const Node* m_node;

        const OperatorSchema* m_opSchema;

        const std::vector<NodeArg>* m_inputs;

        std::vector<NodeArg>* m_outputs;
    };


    // Shape inference function define.
    typedef std::function<Status(InferenceContext&)> ShapeInferenceFunc;

    // An attribute parser - it's specified when registering an operator.
    // The parser is designed and used in two ways.
    // 1) It will be used to verify whether a Node's attributes match the
    //    operator's definition.
    // 2) It will be used to parse a Node's attributes into a <T> object,
    //    which makes it be easier to access node attributes.
    // TODO: to implement the 2nd point above, NodeAttributes should be changed
    // to contain a <T> field, which is structured attributes.
    typedef std::function<Status(const NodeAttributes&)> AttributeParser;

    typedef std::tuple<std::string, std::string, std::string> InputOutputParam;
    typedef std::tuple<std::string, std::string, AttrType, AttributeProto> AttrParam;
    typedef std::tuple<std::string, std::vector<std::string>, std::string> TypeConstraintParam;

#define ATTR_SETTER_INTERFACE(TypeName) \
    OperatorSchemaSetter& Attr(const std::string& p_attrName, \
                               const std::string& p_description, \
                               AttrType p_attrType, \
                               const TypeName& p_defaultValue); \
    OperatorSchemaSetter& Attr(const std::string& p_attrName, \
                               const std::string& p_description, \
                               AttrType p_attrType, \
                               const std::vector<TypeName>& p_defaultValues); \
    // Operator registry setter helper.
    // This is used in "REGISTER_OP" macro, to separate setters from getters
    // in OperatorSchema.
    class OperatorSchemaSetter
    {
    public:

        OperatorSchemaSetter() = default;

        OperatorSchemaSetter& Name(const std::string& p_opName);

        OperatorSchemaSetter& Description(const std::string& p_description);

        OperatorSchemaSetter& Input(const std::string& p_inputName,
            const std::string& p_description,
            const std::string& p_type = "");

        OperatorSchemaSetter& Output(const std::string& p_outputName,
            const std::string& p_description,
            const std::string& p_type = "");

        OperatorSchemaSetter& Attr(const std::string& p_attrName,
            const std::string& p_description,
            AttrType p_attrType, bool required = false);

        ATTR_SETTER_INTERFACE(int64_t)
        ATTR_SETTER_INTERFACE(float)
        ATTR_SETTER_INTERFACE(std::string)
        ATTR_SETTER_INTERFACE(TensorProto)
        ATTR_SETTER_INTERFACE(GraphProto)
        ATTR_SETTER_INTERFACE(TypeProto)
        ATTR_SETTER_INTERFACE(TensorShapeProto)

        OperatorSchemaSetter& TypeConstraint(const std::string& p_typeName,
            const std::vector<std::string>& p_constraints,
            const std::string& p_description);

        // Shape inference function will be used to infer outputs' shape with
        // inputs' shape.
        OperatorSchemaSetter& SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc);

        // Attribute parser will be used to parse Node's attributes to see
        // whether Node attributes are matching operator attributes definition.
        OperatorSchemaSetter& SetAttributeParser(
            AttributeParser p_attrParser);

        enum class SupportType {
            COMMON,
            EXPERIMENTAL,
        };
        // Methods added for compatibility with ONNX OpSchema registration API
        OpSchema& NumInputs(int n)
        {
            return NumInputs(n, n);
        }
        OpSchema& NumInputs(int min, int max)
        {
            m_onnxMinInput = min;
            m_onnxMaxInput = max;
            return *this;
        }
        OpSchema& NumInputs(std::set<int> allowed_input_nums)
        {
            return NumInputs([allowed_input_nums](int n)-> bool {
                return allowed_input_nums.count(n) > 0;
            });
        }
        OpSchema& NumInputs(std::function<bool(int)> func)
        {
            m_onnxNumInputsAllowed = func;
            return *this;
        }
        OpSchema& NumOutputs(int n) {
            return NumOutputs(n, n);
        }
        OpSchema& NumOutputs(int min, int max)
        {
            m_onnxMinOutput = min;
            m_onnxMaxOutput = max;
            return *this;
        }
        OpSchema& NumOutputs(std::set<int> allowed_output_nums)
        {
            return NumOutputs([allowed_output_nums](int n)-> bool {
                return allowed_output_nums.count(n) > 0;
            });
        }
        OpSchema& NumOutputs(std::function<bool(int)> func)
        {
            m_onnxNumOutputsAllowed = func;
            return *this;
        }
        OpSchema& NumInputsOutputs(std::function<bool(int, int)> func)
        {
            m_onnxNumInputsOutputsAllowed = func;
            return *this;
        }
        OpSchema& OutputCalculator(std::function<int(int)> calc) { return *this; }
        OpSchema& SameNumberOfOutput() { return *this; }
        OpSchema& AllowConsumed(std::function<std::pair<bool, int>(int)> inplace) { return *this; }
        OpSchema& AllowConsumed(std::unordered_map<int, int> inplace) { return *this; }
        OpSchema& AllowOneToOneConsumed() { return *this; }
        OpSchema& EnforceConsumed(std::function<std::pair<bool, int>(int)> inplace) { return *this; }
        OpSchema& EnforceConsumed(std::unordered_map<int, int> inplace) { return *this; }
        OpSchema& EnforceOneToOneConsumed() { return *this; }
        OpSchema& SetSupportLevel(SupportType supportType) { return *this; }
        OpSchema& AllowUncheckedAttributes() { return *this; }
        OpSchema& FillUsing(std::function<void(OpSchema&)> populator)
        {
            if (populator)
            {
                populator(*this);
            }
            return *this;
        }
        OpSchema& Input(const int n, const char* name, const char* description)
        {
            return Input(name, description);
        }
        OpSchema& Output(const int n, const char* name, const char* description)
        {
            return Output(name, description);
        }
        OpSchema& SetDoc(const std::string& doc)
        {
            return Description(doc);
        }

    private:

        friend class OperatorSchema;

        // Operator name.
        std::string m_name;

        // Operator description.
        std::string m_description;

        // Operator input formal parameters.
        std::vector<InputOutputParam> m_inputs;

        // Operator output formal parameters.
        std::vector<InputOutputParam> m_outputs;

        // Operator attribute definitions.
        std::vector<AttrParam> m_attributes;

        // Operator type constraints.
        std::vector<TypeConstraintParam> m_constraints;

        // Shape inference function.
        // Its functionality is inferring outputs' shape given inputs' shape.
        ShapeInferenceFunc m_shapeInferFunc;

        // Attribute parser.
        AttributeParser m_parser;

        // To support ONNX variable input/output compatibility.
        // Min and Max num arguments of last input/output.
        int m_onnxMinInput = 0;
        int m_onnxMaxInput = std::numeric_limits<int>::max();
        int m_onnxMinOutput = 0;
        int m_onnxMaxOutput = std::numeric_limits<int>::max();
        std::function<bool(int)> m_onnxNumInputsAllowed =
            [](int) { return true; };
        std::function<bool(int)> m_onnxNumOutputsAllowed =
            [](int) { return true; };
        std::function<bool(int, int)> m_onnxNumInputsOutputsAllowed =
            [](int, int) { return true; };
    };


    typedef std::unordered_set<PTYPE> DataTypeSet;
    typedef std::unordered_map<std::string, std::pair<DataTypeSet, std::string>> TypeConstraintMap;

    // Operator registry specification.
    // It defines input formal parameter, output formal parameters and
    // attributes.
    // Once an operator registry created, it's "Read-Only".
    class OperatorSchema
    {
    public:

        // Formal parameter represenation, including parameter name, type.
        class FormalParameter
        {
        public:

            // Constructor.
            explicit FormalParameter(const std::string& p_name,
                const std::string& p_type,
                const std::string& p_description,
                const TypeConstraintMap& p_constraintMap = TypeConstraintMap());

            // Get formal parameter name.
            const std::string& GetName() const;

            // Get supported data types.
            const DataTypeSet& GetTypes() const;

            // Get formal parameter type string.
            const std::string& GetTypeStr() const;

            // Get formal parameter description.
            const std::string& GetDescription() const;

        private:

            FormalParameter() {}

            // Formal parameter name.
            std::string m_name;

            // A set of data types supported for <*this> formal parameter.
            // It should contain at least one element if this formal parameter
            // is good.
            DataTypeSet m_types;

            // The <parameter type> string specified when registring an op.
            // It could be a supported data type or a type constraint key, which
            // maps to a set of supported data types.
            std::string m_typeStr;

            // Formal parameter description
            std::string m_description;

        };

        // Attribute representation, including name, type, and allowed values.
        // The first element of allowed values (if specified) is the default
        // value.
        class Attribute
        {
        public:

            // Constructor.
            explicit Attribute(const std::string& p_attrName,
                AttrType p_type,
                const std::string& p_description);

            // Constructor with default value.
            explicit Attribute(const std::string& p_attrName,
                AttrType p_type,
                const std::string& p_description,
                const AttributeProto& p_defaultVal);

            // Get attribute name.
            const std::string& GetName() const;

            // Get attribute type.
            AttrType GetType() const;

            // Get to know whether this attribute has default value,
            // if yes, <p_value> will be assigned to be the default value.
            bool HasDefaultValue(const AttributeProto** p_value) const;

        private:

            Attribute() {}

            // Attribute name.
            std::string m_name;

            // Attribute type.
            AttrType m_type;

            // Attribute description.
            std::string m_description;

            // Flag indicates whether a default value specified.
            // It it's true, the first element of <m_allowedValues> is the
            // default value.
            bool m_hasDefaultValue;

            // Allowed attribute values.
            std::vector<AttributeProto> m_allowedValues;
        };

        static bool IsValidAttribute(const AttributeProto& p_attribute);

        // Constructor.
        OperatorSchema() = default;

        // Conversion constructor.
        OperatorSchema(const OperatorSchemaSetter& p_setter);

        // Get operator name.
        const std::string& GetName() const;

        // Get operator description.
        const std::string& GetDescription() const;

        // Get input formal parameters.
        const std::vector<FormalParameter>& GetInputs() const;

        // Get output formal parameters.
        const std::vector<FormalParameter>& GetOutputs() const;

        // Get attributes.
        const std::vector<Attribute>& GetAttributes() const;

        // Get shape inference function.
        ShapeInferenceFunc GetShapeInferenceFunc() const;

        // Get attribute parser.
        AttributeParser GetAttributeParser() const;

        // Get type constraint map.
        const TypeConstraintMap& GetTypeConstraintMap() const;

        // To support ONNX variable input/output compatibility.
        // Min and Max num arguments of last input/output.
        int GetOnnxMinInput() const { return m_onnxMinInput; }
        int GetOnnxMaxInput() const { return m_onnxMaxInput; }
        int GetOnnxMinOutput() const { return m_onnxMinOutput; }
        int GetOnnxMaxOutput() const { return m_onnxMaxOutput; }
        std::function<bool(int)> GetOnnxNumInputsAllowedFunc() const
        {
            return m_onnxNumInputsAllowed;
        }
        std::function<bool(int)> GetOnnxNumOutputsAllowedFunc() const
        {
            return m_onnxNumOutputsAllowed;
        }
        std::function<bool(int, int)> GetOnnxNumInputsOutputsAllowedFunc() const
        {
            return m_onnxNumInputsOutputsAllowed;
        }

    private:

        // Operator name.
        std::string m_name;

        // Operator description.
        std::string m_description;

        // Operator input formal parameters.
        std::vector<FormalParameter> m_inputs;

        // Operator output formal parameters.
        std::vector<FormalParameter> m_outputs;

        // Operator attributes' definitions.
        std::vector<Attribute> m_attributes;

        // Map from constraint name to DataTypeSet
        TypeConstraintMap m_typeConstraintMap;

        // Shape inference function.
        // Its functionality is inferring outputs' shape given inputs' shape.
        ShapeInferenceFunc m_shapeInferFunc;

        // Attribute parser.
        AttributeParser m_parser;

        int m_onnxMinInput;
        int m_onnxMaxInput;
        int m_onnxMinOutput;
        int m_onnxMaxOutput;
        std::function<bool(int)> m_onnxNumInputsAllowed;
        std::function<bool(int)> m_onnxNumOutputsAllowed;
        std::function<bool(int, int)> m_onnxNumInputsOutputsAllowed;
    };

    // Operator schema registry. A singleton registry to manage all operator
    // schemas.
    class OperatorSchemaRegistry
    {
    public:

        // Helper function providing a way to call
        // OperatorSchemaFactory::Register().
        class RegisterOnce
        {
        public:

            RegisterOnce(const OperatorSchemaSetter& p_opRegistry);
        };

        // Try to get operator with specified operator name.
        bool TryGetOp(const std::string& p_name,
            const OperatorSchema** p_opRegistry) const;

        // Register an operator.
        Status Register(const OperatorSchema& p_opRegistry);

        // Get the global operator registry factory instance.
        static OperatorSchemaRegistry* Get();

    private:

        OperatorSchemaRegistry() = default;

        // An operator name to operator schema map.
        std::unordered_map<std::string, OperatorSchema> m_operatorRegistryMap;
    };

    // utility function used by ONNX v1 op registration defs.
    size_t ReplaceAll(std::string& s, const char* from, const char* to);
#define OPERATOR_SCHEMA(OpName) REGISTER_OP(OpName)

#define REGISTER_OP(OpName) REGISTER_OP_UNIQ_HELPER(__COUNTER__, OpName)
#define REGISTER_OP_UNIQ_HELPER(Counter, OpName) REGISTER_OP_UNIQ(Counter, OpName)
#define REGISTER_OP_UNIQ(Counter, OpName)                     \
    static OperatorSchemaRegistry::RegisterOnce op_##Counter  \
    = OperatorSchemaSetter().Name(#OpName)

    // Operator registration example.
    // REGISTER_OP(Add).Description("An operator to sum two float numbers.")
    //   .Input("input_1", "docstr for input_1.", "T")
    //   .Input("input_2", "docstr for input_2.", "T")
    //   .Output("output_1", "docstr for output_1.", "T")
    //   .TypeConstraint("T", { "float16", "float32", "float64" }, "Constrain input and output types to floats.");


}

#endif

#pragma warning(pop)