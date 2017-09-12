#ifndef CORE_GRAPH_OP_H
#define CORE_GRAPH_OP_H

#include <functional>
#include <unordered_map>

#include "ONNXGraph.h"

namespace CommonIR
{
    class OperatorRegistry;

    // A context to contain information for shape inference function.
    // It includes the operator registry, input arguments definition,
    // and mutable output arguments, whose shapes needs to be filled.
    class InferenceContext
    {
    public:

        // TODO: Add input tensors into constructor.
        // In some cases, node evaluation will be needed to get output shapes.
        InferenceContext(const Node* p_node,
            const OperatorRegistry* p_opRegistry,
            const std::vector<NodeArg>* p_inputs,
            std::vector<NodeArg>* p_outputs);

        const Node* GetNode() const;

        const OperatorRegistry* GetOp() const;

        const std::vector<NodeArg>* GetInputs() const;

        std::vector<NodeArg>* Mutable_Outputs();

    private:

        const Node* m_node;

        const OperatorRegistry* m_opRegistry;

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

    // Operator registry setter helper.
    // This is used in "REGISTER_OP" macro, to separate setters from getters
    // in OperatorRegistry.
    class OperatorRegistrySetter
    {
    public:

        OperatorRegistrySetter() = default;

        OperatorRegistrySetter& Name(const std::string& p_opName);

        OperatorRegistrySetter& Description(const std::string& p_description);

        OperatorRegistrySetter& Input(const std::string& p_input);

        OperatorRegistrySetter& Output(const std::string& p_output);

        OperatorRegistrySetter& Attr(const std::string& p_attr);

        // Shape inference function will be used to infer outputs' shape with
        // inputs' shape.
        OperatorRegistrySetter& SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc);

        // Attribute parser will be used to parse Node's attributes to see
        // whether Node attributes are matching operator attributes definition.
        OperatorRegistrySetter& SetAttributeParser(
            AttributeParser p_attrParser);

    private:

        friend class OperatorRegistry;

        // Operator name.
        std::string m_name;

        // Operator description.
        std::string m_description;

        // Operator input formal parameters.
        std::vector<std::string> m_inputs;

        // Operator output formal parameters.
        std::vector<std::string> m_outputs;

        // Operator attributes' definitions.
        std::vector<std::string> m_attributes;

        // Shape inference function.
        // Its functionality is inferring outputs' shape given inputs' shape.
        ShapeInferenceFunc m_shapeInferFunc;

        // Attribute parser.
        AttributeParser m_parser;
    };

    typedef std::unordered_set<std::string> DataTypeSet;

    // Operator registry specification.
    // It defines input formal parameter, output formal parameters and
    // attributes.
    // Once an operator registry created, it's "Read-Only".
    class OperatorRegistry
    {
    public:

        // Formal parameter represenation, including parameter name, type.
        class FormalParameter
        {
        public:

            // Constructor.
            // The syntax used to specify a formal parameter in one string
            // is: <parameter name> : <parameter type>.
            // <parameter type> could be a supported data type or an attribute
            // key, which maps to a set of supported data types. Examples:
            // "p_oneNumber:int", "p_oneNumber:T".
            explicit FormalParameter(const std::string& p_paramStr);

            // Get formal parameter name.
            const std::string& GetName() const;

            // Get supportted data types.
            const DataTypeSet& GetTypes() const;

            // Get formal parameter type string.
            const std::string& GetTypeStr() const;

        private:

            FormalParameter() {}

            // Formal parameter name.
            std::string m_name;

            // A set of data types supported for <*this> formal parameter.
            // It should contain at least one element if this formal parameter
            // is good.
            DataTypeSet m_types;

            // The <parameter type> string specified when registring an op.
            // It could be a supported data type or an attribute key, which
            // maps to a set of supported data types.
            std::string m_typeStr;
        };

        // Attribute representation, including name, type, and allowed values.
        // The first element of allowed values (if specified) is the default
        // value.
        class Attribute
        {
        public:

            // Constructor.
            // The syntax used to specify an attribute in one string is:
            // [*]<attribute name> : <attribute type> [= {<attribute value>[, <attribute value>]}]
            // a. [*] means the attribute has to be specified in node
            //    attributes. No default value provided in this case. The value
            //    set (if specified) here are allowed values.
            // b. If no attribute value specified ([*] should be added in this
            //    case), it means the attribute value should be fetched from
            //    node attributes.
            // c. If a set of attribute values (allowed values) are specified,
            //    it means the attribute value specified in node attributes
            //    should be one of this set, or the attribute value is NOT
            //    specified in node attributes and the first (default) value in
            //    this set will be used (no [*] specified in this case).
            explicit Attribute(const std::string& p_attributeStr);

            // Get attribute name.
            const std::string& GetName() const;

            // Get attribute type.
            const TypeProto& GetType() const;

            // Get to know whether this attribute has default value,
            // if yes, <p_value> will be assigned to be the default value.
            bool HasDefaultValue(const AttributeProto** p_value) const;

            // Get attribute values.
            // Return number of allowed values specifed.
            size_t GetAllowedValues(const AttributeProto** p_values) const;

            // Get to know whether this attribute is mandatory.
            bool IsMandatory() const;

        private:

            Attribute() {}

            // Attribute name.
            std::string m_name;

            // Attribute type.
            TypeProto m_type;

            // Flag indicates whether a default value specified.
            // It it's true, the first element of <m_allowedValues> is the
            // default value.
            bool m_hasDefaultValue;

            // Flag indicates whether <*this> attribute is mandatory.
            // If it's true, then Node that refers to operator with <*this>
            // attribute has to specify value for <*this> attribute.
            bool m_isMandatory;

            // Allowed attribute values.
            std::vector<AttributeProto> m_allowedValues;
        };

        // Constructor.
        OperatorRegistry() = default;

        // Conversion constructor.
        OperatorRegistry(const OperatorRegistrySetter& p_setter);

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

        // Shape inference function.
        // Its functionality is inferring outputs' shape given inputs' shape.
        ShapeInferenceFunc m_shapeInferFunc;

        // Attribute parser.
        AttributeParser m_parser;
    };

    // Operator registry factory. A singleton factory to manage all operator
    // registries.
    class OperatorRegistryFactory
    {
    public:

        // Helper function providing a way to call
        // OperatorRegistryFactory::Register().
        class RegisterOnce
        {
        public:

            RegisterOnce(const OperatorRegistrySetter& p_opRegistry);
        };

        // Try to get operator with specified operator name.
        bool TryGetOp(const std::string& p_name,
            const OperatorRegistry** p_opRegistry) const;

        // Register an operator.
        Status Register(const OperatorRegistry& p_opRegistry);

        // Get the global operator registry factory instance.
        static OperatorRegistryFactory* Get();

    private:

        OperatorRegistryFactory() = default;

        // An operator name to operator registry map.
        std::unordered_map<std::string, OperatorRegistry> m_operatorRegistryMap;
    };

#define REGISTER_OP(OpName) REGISTER_OP_UNIQ(__COUNTER__, OpName)
#define REGISTER_OP_UNIQ(Counter, OpName)                      \
    static OperatorRegistryFactory::RegisterOnce op_##Counter  \
    = OperatorRegistrySetter().Name(#OpName)

    // Operator registering example.
    // REGISTER_OP("Add").Description("An operator to sum two numbers");
    //    .Input("input_1:T")
    //    .Input("input_2:T")
    //    .Output("output_1:T")
    //    .Attr("*T:List<TypeProto>={int, float, double}");


}

#endif