#ifndef CORE_GRAPH_OP_H
#define CORE_GRAPH_OP_H

#include <functional>
#include <unordered_map>

#include "graph.h"

namespace LotusIR
{
    class OperatorSchema;

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

    // Operator registry setter helper.
    // This is used in "REGISTER_OP" macro, to separate setters from getters
    // in OperatorSchema.
    class OperatorSchemaSetter
    {
    public:

        OperatorSchemaSetter() = default;

        OperatorSchemaSetter& Name(const std::string& p_opName);

        OperatorSchemaSetter& Description(const std::string& p_description);

        OperatorSchemaSetter& Input(const std::string& p_input);

        OperatorSchemaSetter& Output(const std::string& p_output);

        OperatorSchemaSetter& Attr(const std::string& p_attr);

        // Shape inference function will be used to infer outputs' shape with
        // inputs' shape.
        OperatorSchemaSetter& SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc);

        // Attribute parser will be used to parse Node's attributes to see
        // whether Node attributes are matching operator attributes definition.
        OperatorSchemaSetter& SetAttributeParser(
            AttributeParser p_attrParser);

    private:

        friend class OperatorSchema;

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
    class OperatorSchema
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

#define REGISTER_OP(OpName) REGISTER_OP_UNIQ_HELPER(__COUNTER__, OpName)
#define REGISTER_OP_UNIQ_HELPER(Counter, OpName) REGISTER_OP_UNIQ(Counter, OpName)
#define REGISTER_OP_UNIQ(Counter, OpName)                     \
    static OperatorSchemaRegistry::RegisterOnce op_##Counter  \
    = OperatorSchemaSetter().Name(#OpName)

    // Operator registering example.
    // REGISTER_OP("Add").Description("An operator to sum two numbers");
    //    .Input("input_1:T")
    //    .Input("input_2:T")
    //    .Output("output_1:T")
    //    .Attr("*T:List<TypeProto>={int, float, double}");


}

#endif
