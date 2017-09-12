#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456)

#include "ONNXOP.h"

namespace CommonIR
{
    InferenceContext::InferenceContext(const Node* p_node,
        const OperatorRegistry* p_opRegistry,
        const std::vector<NodeArg>* p_inputs,
        std::vector<NodeArg>* p_outputs)
        : m_node(p_node),
        m_opRegistry(p_opRegistry),
        m_inputs(p_inputs),
        m_outputs(p_outputs)
    {
    }

    const Node* InferenceContext::GetNode() const
    {
        return m_node;
    }

    const OperatorRegistry* InferenceContext::GetOp() const
    {
        return m_opRegistry;
    }

    const std::vector<NodeArg>* InferenceContext::GetInputs() const
    {
        return m_inputs;
    }

    std::vector<NodeArg>* InferenceContext::Mutable_Outputs()
    {
        return m_outputs;
    }

    OperatorRegistrySetter&
        OperatorRegistrySetter::Name(const std::string& p_opName)
    {
        m_name = p_opName;
        return *this;
    }

    OperatorRegistrySetter&
        OperatorRegistrySetter::Description(const std::string& p_description)
    {
        m_description = p_description;
        return *this;
    }

    OperatorRegistrySetter&
        OperatorRegistrySetter::Input(const std::string& p_input)
    {
        m_inputs.push_back(p_input);
        return *this;
    }

    OperatorRegistrySetter&
        OperatorRegistrySetter::Output(const std::string& p_output)
    {
        m_outputs.push_back(p_output);
        return *this;
    }

    OperatorRegistrySetter&
        OperatorRegistrySetter::Attr(const std::string& p_attr)
    {
        m_attributes.push_back(p_attr);
        return *this;
    }

    OperatorRegistrySetter&
        OperatorRegistrySetter::SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc)
    {
        m_shapeInferFunc = p_shapeInferFunc;
        return *this;
    }

    OperatorRegistrySetter&
        OperatorRegistrySetter::SetAttributeParser(
            AttributeParser p_attrParser)
    {
        m_parser = p_attrParser;
        return *this;
    }

    OperatorRegistry::FormalParameter::FormalParameter(
        const std::string& p_paramStr)
    {
        // TODO: add implementation.
    }

    const std::string& OperatorRegistry::FormalParameter::GetName() const
    {
        return m_name;
    }

    const DataTypeSet&
        OperatorRegistry::FormalParameter::GetTypes() const
    {
        return m_types;
    }

    const std::string& OperatorRegistry::FormalParameter::GetTypeStr() const
    {
        return m_typeStr;
    }

    OperatorRegistry::Attribute::Attribute(const std::string& p_attributeStr)
    {
        // TODO: add implementation.
    }

    const std::string& OperatorRegistry::Attribute::GetName() const
    {
        return m_name;
    }

    const TypeProto& OperatorRegistry::Attribute::GetType() const
    {
        return m_type;
    }

    bool OperatorRegistry::Attribute::HasDefaultValue(
        const AttributeProto** p_value) const
    {
        if (m_hasDefaultValue
            && nullptr != p_value)
        {
            *p_value = &(m_allowedValues[0]);
        }

        return m_hasDefaultValue;
    }

    size_t OperatorRegistry::Attribute::GetAllowedValues(
        const AttributeProto** p_values) const
    {
        if (nullptr == p_values)
        {
            return 0;
        }

        *p_values = m_allowedValues.data();
        return m_allowedValues.size();
    }

    bool OperatorRegistry::Attribute::IsMandatory() const
    {
        return m_isMandatory;
    }

    OperatorRegistry::OperatorRegistry(const OperatorRegistrySetter& p_setter)
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

    const std::string& OperatorRegistry::GetName() const
    {
        return m_name;
    }

    const std::string& OperatorRegistry::GetDescription() const
    {
        return m_description;
    }

    const std::vector<OperatorRegistry::FormalParameter>&
        OperatorRegistry::GetInputs() const
    {
        return m_inputs;
    }

    const std::vector<OperatorRegistry::FormalParameter>&
        OperatorRegistry::GetOutputs() const
    {
        return m_outputs;
    }

    const std::vector<OperatorRegistry::Attribute>&
        OperatorRegistry::GetAttributes() const
    {
        return m_attributes;
    }

    ShapeInferenceFunc OperatorRegistry::GetShapeInferenceFunc() const
    {
        return m_shapeInferFunc;
    }

    AttributeParser OperatorRegistry::GetAttributeParser() const
    {
        return m_parser;
    }

    OperatorRegistryFactory::RegisterOnce::RegisterOnce(
        const OperatorRegistrySetter& p_opRegistry)
    {
        OperatorRegistryFactory::Get()->Register(p_opRegistry);
    }

    bool OperatorRegistryFactory::TryGetOp(const std::string& p_name,
        const OperatorRegistry** p_opRegistry) const
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

    Status OperatorRegistryFactory::Register(
        const OperatorRegistry& p_opRegistry)
    {
        auto iter = m_operatorRegistryMap.find(p_opRegistry.GetName());
        if (m_operatorRegistryMap.end() != iter)
        {
            Status status(false,
                "Error: operator registry with same name ("
                + p_opRegistry.GetName() + ") exists.");
            return status;
        }
        else
        {
            m_operatorRegistryMap[p_opRegistry.GetName()] = p_opRegistry;
            return Status::OK();
        }
    }

    OperatorRegistryFactory* OperatorRegistryFactory::Get()
    {
        static OperatorRegistryFactory* s_factory
            = new OperatorRegistryFactory();
        return s_factory;
    }
}

#pragma warning(pop)