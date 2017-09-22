//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNXToCNTK.h"
#include "./core/graph.h"
#include "Utils.h"
#include "Operators.h"

using namespace LotusIR;

namespace CNTK
{
    typedef std::unordered_map<const Node *, FunctionPtr> ONNXToCNTKMap;
class ONNXToCNTKHelper
{
public:
    static FunctionPtr FromONNXNode(const Node *node,ONNXToCNTKMap &constructedNodeMap);
    static FunctionPtr CreateCNTKFunction(const Node *node, std::vector<Variable> &inputs);
};

FunctionPtr ONNXToCNTKHelper::FromONNXNode(const Node *node, ONNXToCNTKMap &constructedNodeMap)
{
    ONNXToCNTKMap::iterator itONNXToCNTKMap = constructedNodeMap.find(node);
    if (itONNXToCNTKMap != constructedNodeMap.end())
    {
        return itONNXToCNTKMap->second;
    }

    std::vector<Variable> inputs;
    for (Node::NodeConstIterator it = node->InputNodes_begin(); it != node->InputNodes_end(); ++it)
    {
        const Node *onnxNode = *it;
        ONNXToCNTKMap::iterator itNodeMap = constructedNodeMap.find(const_cast<Node *>(onnxNode));
        if (itNodeMap != constructedNodeMap.end())
        {
            inputs.push_back(itNodeMap->second);
        }
        else
        {
            FunctionPtr input = FromONNXNode(onnxNode, constructedNodeMap);
            inputs.push_back(input);
        }
    }

    FunctionPtr cntkFunction = CreateCNTKFunction(node, inputs);
    constructedNodeMap.insert(ONNXToCNTKMap::value_type(node, cntkFunction));
    return cntkFunction;
}

FunctionPtr ONNXToCNTKHelper::CreateCNTKFunction(const Node *node, std::vector<Variable> &inputs)
{
    string onnxOpName = node->OpType();

    if (onnxOpName == "NoOp")
    {
        // TODO: this is for sink or source - what type of variable for it?
        NDShape shape;
        Constant constantVariable(shape, 0.5F, DeviceDescriptor::UseDefaultDevice(), ToWString(node->Name()));
        return constantVariable;
    }
    else if (onnxOpName == "Constant")
    {
        if (node->Name() == "Parameter5")
        {
            NDShape shape({ 3, 3, 1, 4 });
            Constant constantVariable(shape, 0.5F, DeviceDescriptor::UseDefaultDevice(), ToWString(node->Name()));
            return constantVariable;
        }
        else
        { 
            NDShape shape;
            Constant constantVariable(shape, 0.5F, DeviceDescriptor::UseDefaultDevice(), ToWString(node->Name()));
            return constantVariable;
        }
    }
    else if (onnxOpName == "Variable")
    {
        // TODO: placeholder or input, or even output?
        NDShape shape({28, 28});
        bool isSparse = false;
        DataType dataType = DataType::Float;
        bool needsGradient = false; 
        std::vector<Axis> dynamicAxes = Axis::DefaultInputVariableDynamicAxes();
        Variable variable = InputVariable(shape, isSparse, dataType, needsGradient, ToWString(node->Name()), dynamicAxes);
        return variable;
    }
    else if (onnxOpName == "AveragePool" || onnxOpName == "MaxPool")
    {
        NDShape poolingWindowShape;
        NDShape strides; 
        std::vector<bool> autoPadding;
        bool ceilOutDim = false;
        bool includePad = false;
        FunctionPtr cntkFunction = Pooling(inputs[0], 
            onnxOpName == "AveragePool" ? PoolingType::Average : PoolingType::Max,
            poolingWindowShape, strides, autoPadding, ceilOutDim, includePad, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Conv")
    {
        NDShape strides({2, 2});
        std::vector<bool> sharing({ true });
        std::vector<bool> autoPadding({ true});
        NDShape dilation({1});
        size_t reductionRank = 1; 
        size_t maxTempMemSizeInSamples = 0;
        FunctionPtr cntkFunction = Convolution(
            inputs[1], 
            inputs[0],
            strides,
            sharing,
            autoPadding,
            dilation,
            reductionRank,
            maxTempMemSizeInSamples,
            ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ConvTranspose")
    {
        return nullptr;
    }
    else if (onnxOpName == "GlobalAveragePool")
    {
        return nullptr;
    }
    else if (onnxOpName == "GlobalMaxPool")
    {
        return nullptr;
    }
    else if (onnxOpName == "BatchNormalization")
    {
        return nullptr;
    }
    else if (onnxOpName == "Dropout")
    {
        return nullptr;
    }
    else if (onnxOpName == "RandomUniform")
    {
        // ??? create a constant??
        return nullptr;
    }
    else if (onnxOpName == "RandomNormal")
    {
        return nullptr;
    }
    else if (onnxOpName == "RandomUniformLike")
    {
        return nullptr;
    }
    else if (onnxOpName == "RandomNormalLike")
    {
        return nullptr;
    }
    else if (onnxOpName == "Add")
    {
        FunctionPtr cntkFunction = Plus(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sub")
    {
        FunctionPtr cntkFunction = Minus(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Mul")
    {
        FunctionPtr cntkFunction = ElementTimes(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Div")
    {
        FunctionPtr cntkFunction = ElementDivide(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Neg")
    {
        FunctionPtr cntkFunction = Negate(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Abs")
    {
        FunctionPtr cntkFunction = Abs(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Reciprocal")
    {
        FunctionPtr cntkFunction = Reciprocal(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Floor")
    {
        FunctionPtr cntkFunction = Floor(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Ceil")
    {
        FunctionPtr cntkFunction = Ceil(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sqrt")
    {
        FunctionPtr cntkFunction = Sqrt(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Relu")
    {
        FunctionPtr cntkFunction = ReLU(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "LeakyRelu")
    {
        FunctionPtr cntkFunction = LeakyReLU(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Selu")
    {
        double scale = 1;
        double alpha = 0.5;
        FunctionPtr cntkFunction = SELU(inputs[0], scale, alpha, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Elu")
    {
        FunctionPtr cntkFunction = ELU(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Exp")
    {
        FunctionPtr cntkFunction = Exp(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Log")
    {
        FunctionPtr cntkFunction = Log(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Tanh")
    {
        FunctionPtr cntkFunction = Tanh(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Pow")
    {
        return nullptr;
    }
    else if (onnxOpName == "Dot")
    {
        FunctionPtr cntkFunction = Times(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "PRelu")
    {
        FunctionPtr cntkFunction = PReLU(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sigmoid")
    {
        FunctionPtr cntkFunction = Sigmoid(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Max")
    {
        return nullptr;
    }
    else if (onnxOpName == "Min")
    {
        return nullptr;
    }
    else if (onnxOpName == "Sum")
    {
        return nullptr;
    }
    else if (onnxOpName == "Softmax")
    {
        FunctionPtr cntkFunction = Softmax(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMax")
    {
        Axis axis;
        FunctionPtr cntkFunction = ReduceMax(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMin")
    {
        Axis axis;
        FunctionPtr cntkFunction = ReduceMin(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceSum")
    {
        Axis axis;
        FunctionPtr cntkFunction = ReduceSum(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMean")
    {
        Axis axis;
        FunctionPtr cntkFunction = ReduceMean(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceProd")
    {
        Axis axis;
        FunctionPtr cntkFunction = ReduceProd(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceLogSumExp")
    {
        Axis axis;
        FunctionPtr cntkFunction = ReduceLogSum(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ArgMax")
    {
        Axis axis;
        FunctionPtr cntkFunction = Argmax(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ArgMin")
    {
        Axis axis;
        FunctionPtr cntkFunction = Argmin(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Reshape")
    {
        NDShape newShape;
        FunctionPtr cntkFunction = Reshape(inputs[0], newShape, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Concat")
    {
        Axis axis;
        std::vector<Variable> operands;
        std::transform(inputs.begin(), inputs.end(), operands.begin(), [](FunctionPtr fn) { return fn->Output(); });
        FunctionPtr cntkFunction = Splice(operands, axis, ToWString(node->Name()));
        return cntkFunction;
    }
        // { L"", "Split)
    else if (onnxOpName == "Slice")
    {
        std::vector<Axis> axis;
        std::vector<int> beginIndex;
        std::vector<int> endIndex;
        FunctionPtr cntkFunction = Slice(inputs[0], axis, beginIndex, endIndex, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Transpose")
    {
        FunctionPtr cntkFunction = Transpose(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Gather")
    {
        FunctionPtr cntkFunction = GatherOp(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else
    {
        // throw 
        return nullptr;
    }
}

FunctionPtr ONNXToCNTK::CreateGraph(const std::unique_ptr<LotusIR::Graph>& src)
{
    FunctionPtr cntkModel;    
    ONNXToCNTKMap constructedFunctions;
    for (Graph::NodeIterator it = src->Nodes_begin(); it != src->Nodes_end(); ++it)
    {
        const Node *node = *it;

        if (constructedFunctions.find(node) == constructedFunctions.end())
        {
            FunctionPtr cntkNode = ONNXToCNTKHelper::FromONNXNode(node, constructedFunctions);
        }
    }

    // TODO: 
    ONNXToCNTKMap::iterator itNodeFn = std::find_if(constructedFunctions.begin(), constructedFunctions.end(),
        [](ONNXToCNTKMap::value_type nodeFn) {return ToString(nodeFn.second->Name()) == "Plus33"; });
    return itNodeFn->second;
}

}