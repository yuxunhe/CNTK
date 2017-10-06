//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNXToCNTK.h"
#include "proto/onnx/core/graph.h"
#include "Utils.h"
#include "Operators.h"
#include <algorithm>
#include <iostream>

using namespace ONNXIR;
using namespace CNTK;
using namespace CNTK::ONNX;

namespace CNTK
{

typedef std::unordered_map<const Node *, FunctionPtr> ONNXToCNTKMap;
class ONNXToCNTKHelper
{
public:
    static FunctionPtr FromONNXNode(const Node *node, ONNXToCNTKMap &constructedNodeMap,
        const DeviceDescriptor& computeDevice);

private:
    static FunctionPtr CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs,
        const DeviceDescriptor& computeDevice);
    static Constant CreateConvKernelConstant(const Node *node);
    static Constant CreateConstant(const Node *node, const DeviceDescriptor& computeDevice);
    static Variable CreateVariable(const Node *node);
    static Variable CreateVariable(const NodeArg *nodeArg);
    static FunctionPtr CreateFunction(const Node *node, const std::vector<Variable> &inputs);

    static std::vector<Axis> FromINTSToAxes(const std::vector<int64_t> &ints);
    static ONNXIR::TypeProto FromINTS(const std::vector<int64_t> &shape);
    static NDShape FromTypeProto(const ONNXIR::TypeProto& tensorShape);
    static NDShape FromTensorShapeProto(const ONNXIR::TypeProto::TensorShapeProto& tensorShape);
    static std::vector<bool> FromTypeProtoAsBool(const ONNXIR::TypeProto& tensorShape);
    static DataType FromONNXType(ONNXIR::TypeProto type);

    static std::vector<Axis> GetNamedAttributeAsAxis(const Node *node, const string &attributeName);
    static NDShape GetNamedAttributeAsShape(const Node *node, const string &attributeName, bool hasBatchAxis = false);
    static std::vector<bool> GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName);
    static size_t GetNamedAttributeAsInt64(const Node *node, const string &attributeName);
    static float GetNamedAttributeAsFloat(const Node *node, const string &attributeName);

    static NDShape ReverseShape(const NDShape &shape);
};

}

std::vector<Axis> ONNXToCNTKHelper::FromINTSToAxes(const std::vector<int64_t> &ints)
{
    std::vector<Axis> axes;
    for (std::vector<int64_t>::const_iterator it = ints.begin(); it != ints.end(); it++)
    {
        axes.push_back(Axis((int)(*it)));
    }
    return axes;
}

ONNXIR::TypeProto ONNXToCNTKHelper::FromINTS(const std::vector<int64_t> &shape)
{
    ONNXIR::TypeProto newShape;

    for (std::vector<int64_t>::const_iterator it = shape.begin(); it != shape.end(); it++)
    {
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(*it);
    }

    return newShape;
}

NDShape ONNXToCNTKHelper::FromTypeProto(const ONNXIR::TypeProto& tensorShape)
{
    return FromTensorShapeProto(tensorShape.tensor_type().shape());
}

NDShape ONNXToCNTKHelper::FromTensorShapeProto(const ONNXIR::TypeProto::TensorShapeProto& tensorShape)
{
    std::vector<size_t> dimensions;
    for (int index = 0; index < tensorShape.dim_size(); index++)
        dimensions.push_back(tensorShape.dim(index).dim_value());

    // CNTKToONNX ToTensorShape does reverse, need to reverse to restore CNTK shape
    return ReverseShape(dimensions);
}

NDShape ONNXToCNTKHelper::ReverseShape(const NDShape &shape)
{
    std::vector<size_t> dimensions;
    for (int index = shape.Rank() - 1; index >= 0; index--)
    {
        dimensions.push_back(shape[index]);
    }
    return dimensions;
}

std::vector<bool> ONNXToCNTKHelper::FromTypeProtoAsBool(const ONNXIR::TypeProto& tensorShape)
{
    std::vector<bool> dimensions;
    for (int index = 0; index < tensorShape.tensor_type().shape().dim_size(); index++)
        dimensions.push_back(tensorShape.tensor_type().shape().dim(index).dim_value() == 0 ? false : true);

    // CNTKToONNX ToTensorShape does reverse, need to reverse to restore CNTK shape
    std::reverse(dimensions.begin(), dimensions.end());
    return dimensions;
}

DataType ONNXToCNTKHelper::FromONNXType(ONNXIR::TypeProto type)
{
    switch (type.tensor_type().elem_type())
    {
    case ONNXIR::TensorProto_DataType_FLOAT:
        return DataType::Float;
    case ONNXIR::TensorProto_DataType_DOUBLE:
        return DataType::Double;
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

Constant ONNXToCNTKHelper::CreateConvKernelConstant(const Node *node)
{
    // TODO: create Constant from Node attribute and initializer
    return Constant::Scalar(0);
}

Constant ONNXToCNTKHelper::CreateConstant(const Node *node, const DeviceDescriptor& computeDevice)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find("value");
    const ONNXIR::TensorProto valueProto = itValue->second.t();
    auto dataType = valueProto.data_type();

    NDShape shape(std::vector<size_t>(valueProto.dims().begin(), valueProto.dims().end()));

    // the following code is to revert CNTKToONNXHelper::ToTensorShape.to restore a CNTK NDArray
    NDShape reversedShape = ReverseShape(shape);

    auto totalSize = shape.TotalSize();

    switch (dataType)
    {
    case TensorProto_DataType_FLOAT:
    {
        float *data = new float[totalSize];

        // TODO: for onnx tensfor the first 1 or 2 are probably batch and channel. the last 2 are kernal size
        if (shape.Rank() <= 2)
        {
            for (size_t index = 0; index < totalSize; index++)
            {
                data[index] = valueProto.float_data()[index];
            }
        }
        else
        {
            int outputChannels = shape[0], inputChanndels = shape[1];
            NDShape channelKernelShape(std::vector<size_t>(shape.Dimensions().begin() + 2, shape.Dimensions().end()));
            NDShape channelReversedShape = ReverseShape(channelKernelShape);
            int channelKernelSize = channelKernelShape.TotalSize();
            // int nonChannelKernelSize = outputChannels * inputChanndels;
            for (int oC = 0; oC < outputChannels; oC++)
            {
                for (int iC = 0; iC < inputChanndels; iC++)
                {
                    int channelIndex = (oC * inputChanndels + iC);
                    for (int pixel = 0; pixel < channelKernelSize; pixel++)
                    {
                        data[channelIndex * channelKernelSize + pixel] = 
                            valueProto.float_data()[channelIndex * channelKernelSize + pixel];
                    }
                }
            }
        }

        NDArrayViewPtr dstFinal(new NDArrayView(DataType::Float, reversedShape, &data[0], 
            totalSize * sizeof(float), computeDevice.CPUDevice()));
                
        if (computeDevice.Type() == DeviceKind::CPU)
        {
            Constant constantVariable(dstFinal, ToWString(node->Name()));
            return constantVariable;
        }
        else
        {
            NDArrayViewPtr dstFinalGPU(new NDArrayView(DataType::Float, StorageFormat::Dense, reversedShape, computeDevice));
            dstFinalGPU->CopyFrom(*dstFinal);
            Constant constantVariable(dstFinalGPU, ToWString(node->Name()));
            return constantVariable;
        }
    }
    break;
    case TensorProto_DataType_DOUBLE:
        // TODO:
        NOT_IMPLEMENTED;
    break;
    default:
        NOT_IMPLEMENTED;
    }

    // do transpose and reshape
}

Variable ONNXToCNTKHelper::CreateVariable(const NodeArg *nodeArg)
{
    auto dataType = FromONNXType(nodeArg->ToProto().type());
    auto shapeProto = nodeArg->Shape();

    NDShape shape = FromTensorShapeProto(*shapeProto);
    switch (dataType)
    {
    case DataType::Float:
    {
        return InputVariable(shape, DataType::Float, ToWString(nodeArg->Name()));
    }
    case DataType::Double:
    {
        return InputVariable(shape, DataType::Double, ToWString(nodeArg->Name()));
    }
    default:
        NOT_IMPLEMENTED;
    }
}

Variable ONNXToCNTKHelper::CreateVariable(const Node *node)
{
    ONNXIR::NodeArg inputArg = node->OutputDefs()[0];

    auto dataType = FromONNXType(inputArg.ToProto().type());
    auto shapeProto = inputArg.Shape();
    NDShape shape = FromTensorShapeProto(*shapeProto);

    switch (dataType)
    {
    case DataType::Float:
    {
        return InputVariable(shape, DataType::Float, ToWString(node->Name()));
    }
    case DataType::Double:
    {
        return InputVariable(shape, DataType::Double, ToWString(node->Name()));
    }
    default:
        NOT_IMPLEMENTED;
    }
}

namespace CNTK
{
void CheckForAxes(const string &nodeName, const std::vector<Axis> &axes, int requiredAxes)
{
    if (axes.size() != requiredAxes)
        LogicError("%s has %d input axis/axes. It should has %d .", nodeName.c_str(), (int)axes.size(), requiredAxes);
}
}

std::vector<Axis> ONNXToCNTKHelper::GetNamedAttributeAsAxis(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find(attributeName);
    const AttributeProto &attributeProto = itValue->second;
    std::vector<int64_t> axes(attributeProto.ints().begin(), attributeProto.ints().end());
    return FromINTSToAxes(axes);
}

NDShape ONNXToCNTKHelper::GetNamedAttributeAsShape(const Node *node, const string &attributeName, bool hasBatchAxis)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find(attributeName);
    const AttributeProto &attributeProto = itValue->second;
    ::google::protobuf::RepeatedField<::google::protobuf::int64>::const_iterator itBegin =
        attributeProto.ints().begin();
    if (hasBatchAxis)
        itBegin++;
    std::vector<int64_t> shape(itBegin, attributeProto.ints().end());
    return FromTypeProto(FromINTS(shape));
}

std::vector<bool> ONNXToCNTKHelper::GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find(attributeName);
    const AttributeProto &attributeProto = itValue->second;
    std::vector<int64_t> shape(attributeProto.ints().begin(), attributeProto.ints().end());
    return FromTypeProtoAsBool(FromINTS(shape));
}

size_t ONNXToCNTKHelper::GetNamedAttributeAsInt64(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find(attributeName);
    const AttributeProto &attributeProto = itValue->second;
    int64_t size64 = attributeProto.i();
    return size64;
}

float ONNXToCNTKHelper::GetNamedAttributeAsFloat(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find(attributeName);
    const AttributeProto &attributeProto = itValue->second;
    float floatValue = attributeProto.f();
    return floatValue;
}

const AttributesMapping &ONNXToCNTKAttributeNameMapping(const string &onnxOpName)
{
    const std::unordered_multimap<std::wstring, AttributesMapping>& lookup = 
        Operators::CntkToONNXLookup();

    std::unordered_multimap<std::wstring, AttributesMapping>::const_iterator it =
        std::find_if(lookup.begin(), lookup.end(), [&onnxOpName]
        (const std::unordered_multimap<std::wstring, AttributesMapping>::value_type &opMap)
    {
        const AttributesMapping &attributesMapping = opMap.second;
        return std::any_of(attributesMapping.map.begin(), attributesMapping.map.end(), 
            [&onnxOpName](const std::unordered_map<std::wstring, std::string>::value_type &attributesMap)
        {
            return attributesMap.second == onnxOpName;
        });
    });

    assert(it != lookup.end());
    return it->second;
}

//const std::wstring &LookUpCNTKAttributeName(
//    const AttributesMapping &cntkToONNXAttributesMapping, const string &onnxAttributeName)
//{
//    std::unordered_map<std::wstring, std::string>::const_iterator it = 
//        std::find_if(cntkToONNXAttributesMapping.map.begin(), cntkToONNXAttributesMapping.map.end(),
//            [&onnxAttributeName](std::unordered_map<std::wstring, std::string>::iterator it) 
//    {
//        return it->second == onnxAttributeName;
//    });

//    assert(it != cntkToONNXAttributesMapping.map.end()); 
//    return it->first;
//}

std::string ShapeToString(const NDShape &shape)
{
    string s = "[";
    for (int i = 0; i < shape.Rank(); i++)
    {
        s += std::to_string(shape[i]) + " ";
    }
    s += "]";
    return s;
}

void Trace0(const string &onnxOpName, const Variable& variable)
{
    std::string shape = ShapeToString(variable.Shape());
    std::cout << endl;
    std::cout << onnxOpName << endl;
    std::cout << ToString(variable.Name()) << shape << endl;
    std::cout << endl;
}

void Trace1(const string &onnxOpName, const FunctionPtr cntkFunction, const Variable &input0)
{
    std::string outShape = ShapeToString(cntkFunction->Output().Shape());
    std::string input0Shape = ShapeToString(input0.Shape());
    std::cout << endl;
    std::cout << onnxOpName << endl;
    std::cout << ToString(cntkFunction->Name()) << outShape << " -> " << ToString(input0.Name()) << input0Shape << endl;
    std::cout << endl;
}

void Trace2(const string &onnxOpName, const FunctionPtr cntkFunction, const Variable &input0, const Variable &input1)
{
    std::string outShape = ShapeToString(cntkFunction->Output().Shape());
    std::string input0Shape = ShapeToString(input0.Shape());
    std::string input1Shape = ShapeToString(input1.Shape());
    std::cout << endl;
    std::cout << onnxOpName << endl;
    std::cout << ToString(cntkFunction->Name()) << outShape << " -> " << ToString(input0.Name()) << input0Shape << endl;
    std::cout << ToString(cntkFunction->Name()) << outShape << " -> " << ToString(input1.Name()) << input1Shape << endl;
    std::cout << endl;
}

FunctionPtr ONNXToCNTKHelper::CreateFunction(const Node *node, const std::vector<Variable> &inputs)
{
    string onnxOpName = node->OpType();
    auto lookup = Operators::CntkToONNXLookup();
    auto attributesCNTKToONNXMap = ONNXToCNTKAttributeNameMapping(onnxOpName);

    if (onnxOpName == "AveragePool" || onnxOpName == "MaxPool")
    {
        NDShape poolingWindowShape = GetNamedAttributeAsShape(node, "kernel_shape");
        NDShape strides = GetNamedAttributeAsShape(node, "strides");
        std::vector<bool> autoPadding = GetNamedAttributeAsShapeBool(node, attributesCNTKToONNXMap.map[L"autoPadding"]);

        // TODO: get from node's attributes
        bool ceilOutDim = false;
        bool includePad = false;
        FunctionPtr cntkFunction = Pooling(inputs[0],
            onnxOpName == "AveragePool" ? PoolingType::Average : PoolingType::Max,
            poolingWindowShape, strides, autoPadding, ceilOutDim, includePad, ToWString(node->Name()));
        Trace1(onnxOpName, cntkFunction, inputs[0]);
        return cntkFunction;
    }
    else if (onnxOpName == "Conv")
    {
        NDShape strides = GetNamedAttributeAsShape(node, "strides");
        NDShape dilation = GetNamedAttributeAsShape(node, attributesCNTKToONNXMap.map[L"dilation"]);
        std::vector<bool> autoPadding = GetNamedAttributeAsShapeBool(node, attributesCNTKToONNXMap.map[L"autoPadding"]);
        size_t groups = GetNamedAttributeAsInt64(node, "group");

        // TODO: get from node's attributes
        std::vector<bool> sharing({ true });
        size_t reductionRank = 1;
        size_t maxTempMemSizeInSamples = 0;

        // TODO: create the kernel node from an ONNX input if it is available.
        // otherwise build from attribute and onnx initialization list.
        Variable convolutionMap = inputs.size() == 1 ? CreateConvKernelConstant(node) : inputs[1];
        Variable operand = inputs[0];
        FunctionPtr cntkFunction = Convolution(
            convolutionMap,
            operand,
            strides,
            sharing,
            autoPadding,
            dilation,
            reductionRank,
            groups,
            maxTempMemSizeInSamples,
            ToWString(node->Name()));
        if (inputs.size() == 1)
            Trace1(onnxOpName, cntkFunction, inputs[0]);
        else if (inputs.size() == 2)
            Trace2(onnxOpName, cntkFunction, inputs[0], inputs[1]);
        return cntkFunction;
    }
    else if (onnxOpName == "ConvTranspose")
    {
        NDShape strides = GetNamedAttributeAsShape(node, "strides");
        NDShape dilation = GetNamedAttributeAsShape(node, attributesCNTKToONNXMap.map[L"dilation"]);
        std::vector<bool> autoPadding = GetNamedAttributeAsShapeBool(node, attributesCNTKToONNXMap.map[L"autoPadding"]);

        // TODO: "outputShape" attribute may be missing
        NDShape outputShape = GetNamedAttributeAsShape(node, "outputShape");

        // TODO: get from node's attributes
        std::vector<bool> sharing({ true });
        size_t reductionRank = 1;
        size_t maxTempMemSizeInSamples = 0;

        // TODO: are we sure that convolutionMap and operand are in inputs[1], inputs[0] order?
        FunctionPtr cntkFunction = ConvolutionTranspose(
            inputs[1],
            inputs[0],
            strides,
            sharing,
            autoPadding,
            outputShape,
            dilation,
            reductionRank,
            maxTempMemSizeInSamples,
            ToWString(node->Name()));
        Trace2(onnxOpName, cntkFunction, inputs[0], inputs[1]);
        return cntkFunction;
    }
    else if (onnxOpName == "GlobalAveragePool")
    {
        // TODO: operators.cpp may have reversed mapping
        // seems that we do not have Global pooling in CNTK?
        return nullptr;
    }
    else if (onnxOpName == "GlobalMaxPool")
    {
        return nullptr;
    }
    else if (onnxOpName == "BatchNormalization")
    {
        // TODO: implement this right once ready.
        const Variable& operand = inputs[0];
        const Variable& scale = inputs[1];
        const Variable& bias = inputs[2];
        const Variable& runningMean = inputs[3];
        const Variable& runningInvStd = inputs[4];
        const Variable& runningCount = Constant::Scalar(0.0F);
        bool spatial = GetNamedAttributeAsInt64(node, "spatial") != 0;
        double normalizationTimeConstant = 0;
        double blendTimeConstant = 0;
        double epsilon = 0.00001;
        bool useCuDNNEngine = true;
        FunctionPtr cntkFunction = BatchNormalization(operand,
            scale,
            bias,
            runningMean,
            runningInvStd,
            runningCount,
            spatial,
            normalizationTimeConstant,
            blendTimeConstant,
            epsilon,
            useCuDNNEngine,
            ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Dropout")
    {
        const Variable& operand = inputs[0];
        double dropoutRate = GetNamedAttributeAsFloat(node, attributesCNTKToONNXMap.map[L"dropoutRate"]);
        unsigned long seed = SentinelValueForAutoSelectRandomSeed;
        FunctionPtr cntkFunction = Dropout(operand, dropoutRate, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomUniform")
    {
        const NDShape &shape = GetNamedAttributeAsShape(node, attributesCNTKToONNXMap.map[L"newShape"]);

        // TODO get from node's attributes
        DataType dataType = DataType::Float;

        double low = GetNamedAttributeAsFloat(node, "low");
        double high = GetNamedAttributeAsFloat(node, "high");
        unsigned long seed = GetNamedAttributeAsInt64(node, attributesCNTKToONNXMap.map[L"rngSeed"]);
        FunctionPtr cntkFunction = UniformRandom(shape, dataType, low, high, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomNormal")
    {
        const NDShape& shape = GetNamedAttributeAsShape(node, attributesCNTKToONNXMap.map[L"newShape"]);
        DataType dataType = DataType::Float;
        double mean = GetNamedAttributeAsFloat(node, "mean");
        double scale = GetNamedAttributeAsFloat(node, "scale");
        unsigned long seed = GetNamedAttributeAsInt64(node, attributesCNTKToONNXMap.map[L"rngSeed"]);
        FunctionPtr cntkFunction = NormalRandom(shape, dataType, mean, scale, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomUniformLike")
    {
        const Variable& operand = inputs[0];
        double low = GetNamedAttributeAsFloat(node, "low");
        double high = GetNamedAttributeAsFloat(node, "high");
        unsigned long seed = GetNamedAttributeAsInt64(node, attributesCNTKToONNXMap.map[L"rngSeed"]);
        FunctionPtr cntkFunction = UniformRandomLike(operand, low, high, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomNormalLike")
    {
        const Variable& operand = inputs[0];
        double mean = GetNamedAttributeAsFloat(node, "mean");
        double scale = GetNamedAttributeAsFloat(node, "scale");
        unsigned long seed = GetNamedAttributeAsInt64(node, attributesCNTKToONNXMap.map[L"rngSeed"]);
        FunctionPtr cntkFunction = NormalRandomLike(operand, mean, scale, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Add")
    {
        FunctionPtr cntkFunction = Plus(inputs[0], inputs[1], ToWString(node->Name()));
        Trace2(onnxOpName, cntkFunction, inputs[0], inputs[1]);
        return cntkFunction;
    }
    else if (onnxOpName == "Sub")
    {
        // TODO: for binary operation, inputs shall be ordered.
        FunctionPtr cntkFunction = Minus(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Mul")
    {
        FunctionPtr cntkFunction = ElementTimes(inputs[0], inputs[1], ToWString(node->Name()));
        Trace2(onnxOpName, cntkFunction, inputs[0], inputs[1]);
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
        Trace1(onnxOpName, cntkFunction, inputs[0]);
        return cntkFunction;
    }
    else if (onnxOpName == "LeakyRelu")
    {
        FunctionPtr cntkFunction = LeakyReLU(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Selu")
    {
        // TODO: get from node's attributes
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
        // TODO: why order have to be reversed
        FunctionPtr cntkFunction = Times(inputs[1], inputs[0], ToWString(node->Name()));
        Trace2(onnxOpName, cntkFunction, inputs[0], inputs[1]);
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
        FunctionPtr cntkFunction = ElementMax(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Min")
    {
        FunctionPtr cntkFunction = ElementMin(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sum")
    {
        // not specified in Operators.cpp
        return nullptr;
    }
    else if (onnxOpName == "Softmax")
    {
        FunctionPtr cntkFunction = Softmax(inputs[0], ToWString(node->Name()));
        Trace1(onnxOpName, cntkFunction, inputs[0]);
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMax")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceMax(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMin")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceMin(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceSum")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceSum(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMean")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceMean(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceProd")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceProd(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceLogSumExp")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceLogSum(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ArgMax")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = Argmax(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ArgMin")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = Argmin(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Reshape")
    {
        NDShape newShape = GetNamedAttributeAsShape(node, attributesCNTKToONNXMap.map[L"newShape"]);
        FunctionPtr cntkFunction = Reshape(inputs[0], newShape, ToWString(node->Name()));
        Trace1(onnxOpName, cntkFunction, inputs[0]);
        return cntkFunction;
    }
    else if (onnxOpName == "Concat")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        std::vector<Variable> operands;
        std::transform(inputs.begin(), inputs.end(), operands.begin(), [](FunctionPtr fn) { return fn->Output(); });
        FunctionPtr cntkFunction = Splice(operands, axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    // { L"", "Split)
    else if (onnxOpName == "Slice")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        std::vector<int> beginIndex;
        std::vector<int> endIndex;
        FunctionPtr cntkFunction = Slice(inputs[0], axes, beginIndex, endIndex, ToWString(node->Name()));
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

FunctionPtr ONNXToCNTKHelper::FromONNXNode(const Node *node, ONNXToCNTKMap &constructedNodeMap,
    const DeviceDescriptor& computeDevice)
{
    ONNXToCNTKMap::iterator itONNXToCNTKMap = constructedNodeMap.find(node);
    if (itONNXToCNTKMap != constructedNodeMap.end())
    {
        return itONNXToCNTKMap->second;
    }

    std::vector<Variable> inputs;
    const std::vector<NodeArg>& inputDefs = node->InputDefs();
    for (std::vector<NodeArg>::const_iterator it = inputDefs.begin(); it != inputDefs.end(); ++it)
    {
        NodeArg *nodeArg = const_cast<NodeArg *>(&(*it));
        const Node::EdgeEnd* inputEdgeSrcEnd = nullptr;
        Node *cNode = const_cast<Node *>(node);
        if (cNode->InputEdgeSrcEnd(nodeArg, &inputEdgeSrcEnd))
        {
            const Node* inputNode = inputEdgeSrcEnd->GetNode();
            ONNXToCNTKMap::iterator itNodeMap = constructedNodeMap.find(const_cast<Node *>(inputNode));
            if (itNodeMap != constructedNodeMap.end())
            {
                inputs.push_back(itNodeMap->second);
            }
            else
            {
                FunctionPtr input = FromONNXNode(inputNode, constructedNodeMap, computeDevice);
                inputs.push_back(input);
            }
        }
        else
        {
            Variable inputVariable = CreateVariable(nodeArg);
            Trace0(node->OpType(), inputVariable);
            inputs.push_back(inputVariable);
        }
    }

    FunctionPtr cntkFunction = CreateCNTKNode(node, inputs, computeDevice);
    constructedNodeMap.insert(ONNXToCNTKMap::value_type(node, cntkFunction));
    return cntkFunction;
}


FunctionPtr ONNXToCNTKHelper::CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs,
    const DeviceDescriptor& computeDevice)
{
    string onnxOpName = node->OpType();

    if (onnxOpName == "NoOp")
    {
        // TODO: this is for sink or source - what type of variable for it?
        NDShape shape;
        Constant constantVariable(shape, 0.5F, computeDevice, ToWString(node->Name()));
        return constantVariable;
    }
    else if (onnxOpName == "Constant")
    {
        Constant constant = CreateConstant(node, computeDevice);
        Trace0(onnxOpName, constant);
        return constant;
    }
    else if (onnxOpName == "Variable")
    {
        Variable variable = CreateVariable(node);
        Trace0(onnxOpName, variable);
        return variable;
    }
    else
    {
        return CreateFunction(node, inputs);
    }
}

FunctionPtr ONNXToCNTK::CreateGraph(ONNXIR::Graph* src, const DeviceDescriptor& computeDevice)
{
    FunctionPtr cntkModel;    
    ONNXToCNTKMap constructedFunctions;
    for (Graph::NodeIterator it = src->Nodes_begin(); it != src->Nodes_end(); ++it)
    {
        const Node *node = *it;

        if (constructedFunctions.find(node) == constructedFunctions.end())
        {
            FunctionPtr cntkNode = ONNXToCNTKHelper::FromONNXNode(node, constructedFunctions, computeDevice);
        }
    }

    ONNXToCNTKMap::iterator itNodeFn = std::find_if(constructedFunctions.begin(), constructedFunctions.end(),
        [](ONNXToCNTKMap::value_type nodeFn) {return nodeFn.first->Name() == "_Graph_Sink"; });
    if (itNodeFn == constructedFunctions.end())
    {
        return nullptr;
    }

    std::vector<FunctionPtr> functions;
    for (Node::NodeConstIterator it = itNodeFn->first->InputNodes_begin(); it != itNodeFn->first->InputNodes_end(); ++it)
    {
        functions.push_back(constructedFunctions[*it]);
    }

    if (functions.size() == 0)
    {
        return nullptr;
    }
    else if (functions.size() == 1)
    {
        return functions[0];
    }
    else
    {
        return Combine(std::vector<Variable>(functions.begin(), functions.end()));
    }
}
