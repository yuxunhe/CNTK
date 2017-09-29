#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456)

#include "op.h"
#include "utils.h"

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
        OperatorSchemaSetter::Input(const std::string& p_inputName,
            const std::string& p_description,
            const std::string& p_type)
    {
        m_inputs.push_back(std::make_tuple(p_inputName, p_description, p_type));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Output(const std::string& p_outputName,
            const std::string& p_description,
            const std::string& p_type)
    {
        m_outputs.push_back(std::make_tuple(p_outputName, p_description, p_type));
        return *this;
    }

    OperatorSchemaSetter&
        OperatorSchemaSetter::Attr(const std::string& p_attrName,
            const std::string& p_description,
            AttrType p_attrType, bool required)
    {
        m_attributes.push_back(make_tuple(p_attrName, p_description, p_attrType, AttributeProto()));
        return *this;
    }

#define ATTR_SETTER_BASIC_IMPL(type, field)                                               \
    OperatorSchemaSetter&                                                                 \
        OperatorSchemaSetter::Attr(const std::string& p_attrName,                         \
            const std::string& p_description,                                             \
            AttrType p_attrType,                                                          \
            const type& p_defaultValue)                                                   \
    {                                                                                     \
        AttributeProto a;                                                                 \
        a.set_name(p_attrName);                                                           \
        a.set_##field(p_defaultValue);                                                    \
        m_attributes.push_back(make_tuple(p_attrName, p_description, p_attrType, a));     \
        return *this;                                                                     \
    }                                                                                     \

#define ATTR_SETTER_LIST_IMPL(type, field)                                                \
    OperatorSchemaSetter&                                                                 \
        OperatorSchemaSetter::Attr(const std::string& p_attrName,                         \
            const std::string& p_description,                                             \
            AttrType p_attrType,                                                          \
            const std::vector<type>& p_defaultValue)                                      \
    {                                                                                     \
        AttributeProto a;                                                                 \
        a.set_name(p_attrName);                                                           \
        for (const auto& v : p_defaultValue)                                              \
        {                                                                                 \
            a.add_##field(v);                                                             \
        }                                                                                 \
        m_attributes.push_back(make_tuple(p_attrName, p_description, p_attrType, a));     \
        return *this;                                                                     \
    }                                                                                     \

    ATTR_SETTER_BASIC_IMPL(int64_t, i)
    ATTR_SETTER_BASIC_IMPL(float, f)
    ATTR_SETTER_BASIC_IMPL(std::string, s)
    ATTR_SETTER_LIST_IMPL(int64_t, ints)
    ATTR_SETTER_LIST_IMPL(float, floats)
    ATTR_SETTER_LIST_IMPL(std::string, strings)

    OperatorSchemaSetter&
    OperatorSchemaSetter::TypeConstraint(const std::string& p_typeName,
        const std::vector<std::string>& p_constraints,
        const std::string& p_description)
    {
        m_constraints.push_back(std::make_tuple(p_typeName, p_constraints, p_description));
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
        const std::string& p_name, const std::string& p_type,
        const std::string& p_description,
        const TypeConstraintMap& p_constraintMap)
        : m_name(p_name), m_typeStr(p_type), m_description(p_description)
    {
        auto it = p_constraintMap.find(p_type);
        if (it != p_constraintMap.end())
        {
            m_types = it->second.first;
        }
        else
        {
            if (!p_type.empty())
            {
                m_types.emplace(Utils::OpUtils::ToType(m_typeStr));
            }
        }
    }

    const std::string& OperatorSchema::FormalParameter::GetName() const
    {
        return m_name;
    }

    const DataTypeSet& OperatorSchema::FormalParameter::GetTypes() const
    {
        return m_types;
    }

    const std::string& OperatorSchema::FormalParameter::GetTypeStr() const
    {
        return m_typeStr;
    }

    const std::string& OperatorSchema::FormalParameter::GetDescription() const
    {
        return m_description;
    }

    OperatorSchema::Attribute::Attribute(
        const std::string& p_attrName,
        AttrType p_type,
        const std::string& p_description,
        const AttributeProto& p_defaultVal)
        : m_name(p_attrName), m_type(p_type), m_description(p_description),
        m_hasDefaultValue(true)
    {
        m_allowedValues.push_back(p_defaultVal);
    }

    OperatorSchema::Attribute::Attribute(
        const std::string& p_attrName,
        AttrType p_type,
        const std::string& p_description)
        : m_name(p_attrName), m_type(p_type), m_description(p_description),
        m_hasDefaultValue(false)
    {
    }

    const std::string& OperatorSchema::Attribute::GetName() const
    {
        return m_name;
    }

    AttrType OperatorSchema::Attribute::GetType() const
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

    OperatorSchema::OperatorSchema(const OperatorSchemaSetter& p_setter)
        : m_name(p_setter.m_name),
        m_description(p_setter.m_description),
        m_shapeInferFunc(p_setter.m_shapeInferFunc),
        m_parser(p_setter.m_parser)
    {
#ifdef ONNX_V1_OPSCHEMA_COMPAT
        m_onnxMinInput = p_setter.m_onnxMinInput;
        m_onnxMaxInput = p_setter.m_onnxMaxInput;
        m_onnxMinOutput = p_setter.m_onnxMinOutput;
        m_onnxMaxOutput = p_setter.m_onnxMaxOutput;
        m_onnxNumInputsAllowed = p_setter.m_onnxNumInputsAllowed;
        m_onnxNumOutputsAllowed = p_setter.m_onnxNumOutputsAllowed;
        m_onnxNumInputsOutputsAllowed = p_setter.m_onnxNumInputsOutputsAllowed;
#endif // #ifdef ONNX_V1_OPSCHEMA_COMPAT

        // Process type constraints.
        for (const auto& constraint : p_setter.m_constraints)
        {
            std::string name;
            std::vector<std::string> types;
            std::string desc;
            std::tie(name, types, desc) = constraint;

            auto it = m_typeConstraintMap.find(name);
            if (it == m_typeConstraintMap.end())
            {
                DataTypeSet d;
                for (const auto& t : types)
                {
                    d.insert(Utils::OpUtils::ToType(t));
                }
                m_typeConstraintMap.insert(std::make_pair(name, std::make_pair(d, desc)));
            }
            else
            {
                // already a constraint with the same name. error.
            }
        }

        m_inputs.reserve(p_setter.m_inputs.size());
        for (const auto& input : p_setter.m_inputs)
        {
            std::string name;
            std::string type;
            std::string desc;
            std::tie(name, desc, type) = input;
            m_inputs.push_back(FormalParameter(name, type, desc, m_typeConstraintMap));
        }
#ifdef ONNX_V1_OPSCHEMA_COMPAT
        if (0 == m_inputs.size()
            && m_onnxMinInput == m_onnxMaxInput
            && m_onnxMinInput > 0)
        {
            for (int i = 0; i < m_onnxMinInput; ++i)
            {
                std::string name = "p" + std::to_string(i);
                std::string desc = "Input Parameter " + std::to_string(i);
                m_inputs.push_back(FormalParameter(name, "", desc, m_typeConstraintMap));
            }
        }
#endif

        m_outputs.reserve(p_setter.m_outputs.size());
        for (const auto& output : p_setter.m_outputs)
        {
            std::string name;
            std::string type;
            std::string desc;
            std::tie(name, desc, type) = output;
            m_outputs.push_back(FormalParameter(name, type, desc, m_typeConstraintMap));
        }
#ifdef ONNX_V1_OPSCHEMA_COMPAT
        if (0 == m_outputs.size()
            && m_onnxMinOutput== m_onnxMaxOutput
            && m_onnxMinOutput > 0)
        {
            for (int i = 0; i < m_onnxMinOutput; ++i)
            {
                std::string name = "p" + std::to_string(i);
                std::string desc = "Output Result " + std::to_string(i);
                m_outputs.push_back(FormalParameter(name, "", desc, m_typeConstraintMap));
            }
        }
#endif

        m_attributes.reserve(p_setter.m_attributes.size());
        for (const auto& attr : p_setter.m_attributes)
        {
            std::string name;
            AttrType type;
            std::string desc;
            AttributeProto a;
            std::tie(name, desc, type, a) = attr;
            if (a.name() == name)
            {
                m_attributes.push_back(Attribute(name, type, desc, a));
            }
            else
            {
                m_attributes.push_back(Attribute(name, type, desc));
            }
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

    const TypeConstraintMap& OperatorSchema::GetTypeConstraintMap() const
    {
        return m_typeConstraintMap;
    }

    bool OperatorSchema::IsValidAttribute(const AttributeProto& p_attr)
    {
        if (p_attr.name().empty())
        {
            return false;
        }

        int num_fields =
            p_attr.has_f() +
            p_attr.has_i() +
            p_attr.has_s() +
            p_attr.has_t() +
            p_attr.has_g() +
            (p_attr.floats_size() > 0) +
            (p_attr.ints_size() > 0) +
            (p_attr.strings_size() > 0) +
            (p_attr.tensors_size() > 0) +
            (p_attr.graphs_size() > 0) +
            p_attr.has_type() +
            (p_attr.types_size() > 0) +
            p_attr.has_shape() +
            (p_attr.shapes_size() > 0);

        if (num_fields == 1)
        {
            return true;
        }
        return false;
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

    Status TypeUtils::GetType(const AttributeProto& p_attr, AttrType& p_type)
    {
        if (!OperatorSchema::IsValidAttribute(p_attr))
        {
            return Status(false, "Invalid AttributeProto.");
        }

        if (p_attr.has_f())
        {
            p_type = AttrType::FLOAT;
        }
        else if (p_attr.has_i())
        {
            p_type = AttrType::INT;
        }
        else if (p_attr.has_s())
        {
            p_type = AttrType::STRING;
        }
        else if (p_attr.has_t())
        {
            p_type = AttrType::TENSOR;
        }
        else if (p_attr.has_g())
        {
            p_type = AttrType::GRAPH;
        }
        else if (p_attr.floats_size())
        {
            p_type = AttrType::FLOATS;
        }
        else if (p_attr.ints_size())
        {
            p_type = AttrType::INTS;
        }
        else if (p_attr.strings_size())
        {
            p_type = AttrType::STRINGS;
        }
        else if (p_attr.tensors_size())
        {
            p_type = AttrType::TENSORS;
        }
        else if (p_attr.graphs_size())
        {
            p_type = AttrType::GRAPHS;
        }
        else if (p_attr.has_type())
        {
            p_type = AttrType::TYPE;
        }
        else if (p_attr.types_size())
        {
            p_type = AttrType::TYPES;
        }
        else if (p_attr.has_shape())
        {
            p_type = AttrType::SHAPE;
        }
        else if (p_attr.has_shape())
        {
            p_type = AttrType::SHAPES;
        }
        else
        {
            p_type = AttrType::NONE;
            return Status(false, "Invalid AttributeProto.");
        }

        return Status::OK();
    }

#ifdef ONNX_V1_OPSCHEMA_COMPAT
    size_t ReplaceAll(std::string& s, const char* from, const char* to)
    {
        size_t numReplaced = 0;
        std::string::size_type lenFrom = std::strlen(from);
        std::string::size_type lenTo = std::strlen(to);
        for (std::string::size_type pos = s.find(from); pos != std::string::npos;
            pos = s.find(from, pos + lenTo)) {
            s.replace(pos, lenFrom, to);
            numReplaced++;
        }
        return numReplaced;
    }
#endif
}

#pragma warning(pop)