#include "op.h"

namespace LotusIR
{
    InferenceContext::InferenceContext(const Node* p_node,
        const OperatorSchema* p_opSchema,
        const std::vector<NodeArg>* p_inputs,
        std::vector<NodeArg>* p_outputs)
        : m_node(p_node),
        m_opSchema(p_opSchema),
        m_inputs(p_inputs),
        m_outputs(p_outputs)
    {
    }

    const Node* InferenceContext::GetNode() const
    {
        return m_node;
    }

    const OperatorSchema* InferenceContext::GetOp() const
    {
        return m_opSchema;
    }

    const std::vector<NodeArg>* InferenceContext::GetInputs() const
    {
        return m_inputs;
    }

    std::vector<NodeArg>* InferenceContext::Mutable_Outputs()
    {
        return m_outputs;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Name(const std::string& p_opName)
    {
        m_name = p_opName;
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Description(const std::string& p_description)
    {
        m_description = p_description;
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Input(const std::string& p_input)
    {
        m_inputs.push_back(p_input);
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Output(const std::string& p_output)
    {
        m_outputs.push_back(p_output);
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attr)
    {
        m_attributes.push_back(p_attr);
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc)
    {
        m_shapeInferFunc = p_shapeInferFunc;
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::SetAttributeParser(
            AttributeParser p_attrParser)
    {
        m_parser = p_attrParser;
        return *this;
    }

    OperatorSchema::FormalParameter::FormalParameter(
        const std::string& p_paramStr)
    {
        // TODO: add implementation.
    }

    const std::string& OperatorSchema::FormalParameter::GetName() const
    {
        return m_name;
    }

    const DataTypeSet&
        OperatorSchema::FormalParameter::GetTypes() const
    {
        return m_types;
    }

    const std::string& OperatorSchema::FormalParameter::GetTypeStr() const
    {
        return m_typeStr;
    }

    OperatorSchema::Attribute::Attribute(const std::string& p_attributeStr)
    {
        // TODO: add implementation.
    }

    const std::string& OperatorSchema::Attribute::GetName() const
    {
        return m_name;
    }

    const TypeProto& OperatorSchema::Attribute::GetType() const
    {
        return m_type;
    }

    bool OperatorSchema::Attribute::HasDefaultValue(
        const AttributeProto** p_value) const
    {
        if (m_hasDefaultValue
            && nullptr != p_value)
        {
            *p_value = &(m_allowedValues[0]);
        }

        return m_hasDefaultValue;
    }

    size_t OperatorSchema::Attribute::GetAllowedValues(
        const AttributeProto** p_values) const
    {
        if (nullptr == p_values)
        {
            return 0;
        }

        *p_values = m_allowedValues.data();
        return m_allowedValues.size();
    }

    bool OperatorSchema::Attribute::IsMandatory() const
    {
        return m_isMandatory;
    }

    OperatorSchema::OperatorSchema(const OperatorSchemaSetter& p_setter)
        : m_name(p_setter.m_name),
        m_description(p_setter.m_description),
        m_shapeInferFunc(p_setter.m_shapeInferFunc),
        m_parser(p_setter.m_parser)
    {
        for (auto input : p_setter.m_inputs)
        {
            m_inputs.push_back(FormalParameter(input));
        }

        for (auto output : p_setter.m_outputs)
        {
            m_outputs.push_back(FormalParameter(output));
        }

        for (auto attr : p_setter.m_attributes)
        {
            m_attributes.push_back(Attribute(attr));
        }
    }

    const std::string& OperatorSchema::GetName() const
    {
        return m_name;
    }

    const std::string& OperatorSchema::GetDescription() const
    {
        return m_description;
    }

    const std::vector<OperatorSchema::FormalParameter>&
        OperatorSchema::GetInputs() const
    {
        return m_inputs;
    }

    const std::vector<OperatorSchema::FormalParameter>&
        OperatorSchema::GetOutputs() const
    {
        return m_outputs;
    }

    const std::vector<OperatorSchema::Attribute>&
        OperatorSchema::GetAttributes() const
    {
        return m_attributes;
    }

    ShapeInferenceFunc OperatorSchema::GetShapeInferenceFunc() const
    {
        return m_shapeInferFunc;
    }

    AttributeParser OperatorSchema::GetAttributeParser() const
    {
        return m_parser;
    }

    OperatorSchemaRegistry::RegisterOnce::RegisterOnce(
        const OperatorSchemaSetter& p_opSchema)
    {
        OperatorSchemaRegistry::Get()->Register(p_opSchema);
    }

    bool OperatorSchemaRegistry::TryGetOp(const std::string& p_name,
        const OperatorSchema** p_opRegistry) const
    {
        if (nullptr == p_opRegistry)
        {
            return false;
        }

        auto iter = m_operatorRegistryMap.find(p_name);
        if (m_operatorRegistryMap.end() == iter)
        {
            return false;
        }
        *p_opRegistry = &(iter->second);
        return true;
    }

    Status OperatorSchemaRegistry::Register(
        const OperatorSchema& p_opSchema)
    {
        auto iter = m_operatorRegistryMap.find(p_opSchema.GetName());
        if (m_operatorRegistryMap.end() != iter)
        {
            Status status(false,
                "Error: operator schema with same name ("
                + p_opSchema.GetName() + ") exists.");
            return status;
        }
        else
        {
            m_operatorRegistryMap[p_opSchema.GetName()] = p_opSchema;
            return Status::OK();
        }
    }

    OperatorSchemaRegistry* OperatorSchemaRegistry::Get()
    {
        static OperatorSchemaRegistry* s_registry
            = new OperatorSchemaRegistry();
        return s_registry;
    }
}
