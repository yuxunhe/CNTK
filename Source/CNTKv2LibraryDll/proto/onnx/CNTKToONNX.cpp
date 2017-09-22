//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKToONNX.h"
#include "./core/graph.h"
#include "Utils.h"
#include "Operators.h"

using namespace CNTK::ONNX;

namespace CNTK
{

class CNTKToONNXHelper
{
public:
    static LotusIR::TensorShapeProto CNTKToONNXHelper::ToTensorShape(const NDShape& shape);
    static LotusIR::TypeProto ToONNXType(DataType dataType);
    static std::string ToOPName(const FunctionPtr& src);
    static LotusIR::Node* CreateNode(const FunctionPtr& src,
                                      std::unique_ptr<LotusIR::Graph>& graph,
                                      std::unordered_map<FunctionPtr, LotusIR::Node*>& functionNodes,
                                      std::unordered_map<Variable, LotusIR::Node*>& variableNodes);
};

std::unique_ptr<LotusIR::Graph> CNTKToONNX::CreateGraph(const FunctionPtr& src)
{
    std::unique_ptr<LotusIR::Graph> graph(new LotusIR::Graph("CNTKGraph", 1, 1, "CNTK"));
    std::unordered_map<FunctionPtr, LotusIR::Node*> functionNodes;
    std::unordered_map<Variable, LotusIR::Node*> variableNodes;

    CNTKToONNXHelper::CreateNode(src, graph, functionNodes, variableNodes);

    return graph;
}

LotusIR::TensorShapeProto CNTKToONNXHelper::ToTensorShape(const NDShape& shape)
{
    LotusIR::TensorShapeProto newShape;
    for (auto dimension : shape.Dimensions())
        newShape.add_dim()->set_dim_value(dimension);

    return newShape;
}

LotusIR::TypeProto CNTKToONNXHelper::ToONNXType(DataType dataType)
{
    LotusIR::TypeProto type;
    switch (dataType)
    {
    case DataType::Float:
        type.mutable_tensor_type()->set_elem_type(LotusIR::TensorProto_DataType_FLOAT);
        break;
    case DataType::Double:
        type.mutable_tensor_type()->set_elem_type(LotusIR::TensorProto_DataType_DOUBLE);
        break;
    default:
        NOT_IMPLEMENTED;
    }

    return type;
}

std::string CNTKToONNXHelper::ToOPName(const FunctionPtr& src)
{
    auto lookup = Operators::CntkToONNXLookup();
    assert(lookup.count(src->OpName()) != 0);

    std::string opName = ToString(src->OpName());

    if (lookup.count(src->OpName()) == 1)
        opName = lookup.find(src->OpName())->second;
    else
    {
        if (src->OpName() == L"Pooling")
        {
            PoolingType poolingType = (PoolingType)src->Attributes()[L"poolingType"].Value<size_t>();
            if (poolingType == PoolingType::Max)
                opName = "MaxPool";
            else
                opName = "AveragePool";
        }
    }

    return opName;
}

LotusIR::Node* CNTKToONNXHelper::CreateNode(const FunctionPtr& src,
                                            std::unique_ptr<LotusIR::Graph>& graph,
                                            std::unordered_map<FunctionPtr, LotusIR::Node*>& functionNodes,
                                            std::unordered_map<Variable, LotusIR::Node*>& variableNodes)
{
    auto iter = functionNodes.find(src);
    if (iter != functionNodes.end())
        return iter->second;

    LotusIR::Node* functionNode = nullptr;

    //
    // If this block node equivalent to a primitive ONNX OP, then treated as such.
    // And just maps its argument to ONNX node.
    //
    if (src->IsBlock() && !Operators::IsSupportedCNTKOP(src->OpName()))
    {
        functionNode = CreateNode(src->BlockRoot(), graph, functionNodes, variableNodes);
    }
    else if (Operators::IsSupportedCNTKOP(src->OpName()))
    {
        std::vector<LotusIR::NodeArg> inputs;
        std::vector<LotusIR::NodeArg> outputs;

        for (const auto& output : src->Outputs())
        {
            LotusIR::NodeArg outputArg(ToString(output.Uid()),
                                       CNTKToONNXHelper::ToONNXType(output.GetDataType()),
                                       CNTKToONNXHelper::ToTensorShape(output.Shape()));
            outputs.push_back(outputArg);
        }

        for (const auto& input : src->Inputs())
        {
            if (input.IsPlaceholder())
                continue;

            LotusIR::NodeArg inputArg(ToString(input.Uid()),
                                      CNTKToONNXHelper::ToONNXType(input.GetDataType()),
                                      CNTKToONNXHelper::ToTensorShape(input.Shape()));

            inputs.push_back(inputArg);

            if (input.IsInput() || input.IsParameter() || input.IsConstant())
            {
                if (variableNodes.find(input) == variableNodes.end())
                {
                    std::vector<LotusIR::NodeArg> varInputs;
                    std::vector<LotusIR::NodeArg> varOutputs;

                    varOutputs.push_back({ inputArg });
                    LotusIR::Node* variableNode = nullptr;
                    if (input.IsParameter() || input.IsConstant())
                        variableNode = graph->AddNode(ToString(input.Uid()), "Constant", varInputs, varOutputs);
                    else
                        variableNode = graph->AddNode(ToString(input.Uid()), "Variable", varInputs, varOutputs);

                    variableNodes.emplace(input, variableNode);
                }
            }
            else if (input.IsOutput())
                CreateNode(input.Owner(), graph, functionNodes, variableNodes);
        }

        functionNode = graph->AddNode(ToString(src->Uid()), CNTKToONNXHelper::ToOPName(src), inputs, outputs);
    }
    else
        LogicError("Node '%S': Unsupported node.", src->AsString().c_str());

    functionNodes.emplace(src, functionNode);
    return functionNode;
}

}