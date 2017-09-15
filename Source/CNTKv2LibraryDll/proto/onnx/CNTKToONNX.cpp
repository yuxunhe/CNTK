//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKToONNX.h"
#include "./core/ONNXGraph.h"
#include "Utils.h"

namespace CNTK
{

class CNTKToONNXHelper
{
public:
    static CommonIR::TensorShapeProto CNTKToONNXHelper::ToTensorShape(const NDShape& shape);
    static CommonIR::TypeProto ToONNXType(DataType dataType);
    static CommonIR::Node* CreateNode(const FunctionPtr& src,
                                      std::unique_ptr<CommonIR::Graph>& graph,
                                      std::unordered_map<FunctionPtr, CommonIR::Node*>& functionNodes,
                                      std::unordered_map<Variable, CommonIR::Node*>& variableNodes);
};

std::unique_ptr<CommonIR::Graph> CNTKToONNX::CreateGraph(const FunctionPtr& src)
{
    std::unique_ptr<CommonIR::Graph> graph(new CommonIR::Graph("CNTKGraph", 1));
    std::unordered_map<FunctionPtr, CommonIR::Node*> functionNodes;
    std::unordered_map<Variable, CommonIR::Node*> variableNodes;

    CNTKToONNXHelper::CreateNode(src, graph, functionNodes, variableNodes);

    return graph;
}

CommonIR::TensorShapeProto CNTKToONNXHelper::ToTensorShape(const NDShape& shape)
{
    CommonIR::TensorShapeProto newShape;
    for (auto dimension : shape.Dimensions())
        newShape.add_dim()->set_dim_value(dimension);

    return newShape;
}

CommonIR::TypeProto CNTKToONNXHelper::ToONNXType(DataType dataType)
{
    CommonIR::TypeProto type;
    switch (dataType)
    {
    case DataType::Float:
        type.mutable_tensor_type()->set_elem_type(CommonIR::TypeProto_DataType_FLOAT);
        break;
    case DataType::Double:
        type.mutable_tensor_type()->set_elem_type(CommonIR::TypeProto_DataType_DOUBLE);
        break;
    default:
        NOT_IMPLEMENTED;
    }

    return type;
}

CommonIR::Node* CNTKToONNXHelper::CreateNode(const FunctionPtr& src,
                                             std::unique_ptr<CommonIR::Graph>& graph,
                                             std::unordered_map<FunctionPtr, CommonIR::Node*>& functionNodes,
                                             std::unordered_map<Variable, CommonIR::Node*>& variableNodes)
{
    auto iter = functionNodes.find(src);
    if (iter != functionNodes.end())
        return iter->second;

    CommonIR::Node* functionNode = nullptr;

    if (src->IsBlock())
        functionNode = CreateNode(src->BlockRoot(), graph, functionNodes, variableNodes);
    else
    {
        std::vector<std::vector<CommonIR::NodeArg>> inputs;
        std::vector<CommonIR::NodeArg> outputs;

        for (const auto& output : src->Outputs())
        {
            CommonIR::NodeArg outputArg(ToString(output.Uid()),
                                        CNTKToONNXHelper::ToONNXType(output.GetDataType()),
                                        CNTKToONNXHelper::ToTensorShape(output.Shape()));
            outputs.push_back(outputArg);
        }

        for (const auto& input : src->Inputs())
        {
            if (input.IsPlaceholder())
                continue;

            if (input.IsInput() || input.IsParameter() || input.IsConstant())
            {
                CommonIR::NodeArg inputArg(ToString(input.Uid()),
                                           CNTKToONNXHelper::ToONNXType(input.GetDataType()),
                                           CNTKToONNXHelper::ToTensorShape(input.Shape()));

                inputs.push_back(std::vector<CommonIR::NodeArg>({ inputArg }));

                if (variableNodes.find(input) == variableNodes.end())
                {
                    std::vector<std::vector<CommonIR::NodeArg>> varInputs;
                    std::vector<CommonIR::NodeArg> varOutputs;

                    varOutputs.push_back({ inputArg });
                    CommonIR::Node* variableNode = graph->AddNode(ToString(input.Uid()), "Variable", varInputs, varOutputs);
                    variableNodes.emplace(input, variableNode);
                }
            }
            else if (input.IsOutput())
                CreateNode(input.Owner(), graph, functionNodes, variableNodes);
        }

        functionNode = graph->AddNode(ToString(src->Uid()), ToString(src->OpName()), inputs, outputs);
    }

    functionNodes.emplace(src, functionNode);
    return functionNode;
}

}