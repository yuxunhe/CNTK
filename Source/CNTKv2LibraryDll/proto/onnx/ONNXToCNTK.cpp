//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNXToCNTK.h"
#include "./core/graph.h"
#include "Utils.h"
#include "Operators.h"
#include <algorithm>
#include <iostream>

using namespace LotusIR;
using namespace CNTK::ONNX;

namespace CNTK
{
    typedef std::unordered_map<const Node *, FunctionPtr> ONNXToCNTKMap;
    class ONNXToCNTKHelper
    {
    public:
        static FunctionPtr FromONNXNode(const Node *node, ONNXToCNTKMap &constructedNodeMap);

    private:
        static FunctionPtr CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs);
        static Constant CreateConstant(const Node *node);
        static Variable CreateVariable(const Node *node);
        static Variable CreateVariable(const NodeArg *nodeArg);
        static FunctionPtr CreateFunction(const Node *node, const std::vector<Variable> &inputs);

        static LotusIR::TensorShapeProto FromINTS(const std::vector<int64_t> &shape);
        static NDShape FromTensorShape(const TensorShapeProto& tensorShape);
        static std::vector<bool> FromTensorShapeAsBool(const TensorShapeProto& tensorShape);
        static DataType FromONNXType(LotusIR::TypeProto type);


        static NDShape GetNamedAttributeAsShape(const Node *node, const string &attributeName);
        static std::vector<bool> GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName);
        static size_t GetNamedAttributeAsInt64(const Node *node, const string &attributeName);
        static float GetNamedAttributeAsFloat(const Node *node, const string &attributeName);

        static NDShape ReverseShape(const NDShape &shape);
    };

    LotusIR::TensorShapeProto ONNXToCNTKHelper::FromINTS(const std::vector<int64_t> &shape)
    {
        LotusIR::TensorShapeProto newShape;

        for (std::vector<int64_t>::const_iterator it = shape.begin(); it != shape.end(); it++)
        {
            newShape.add_dim()->set_dim_value(*it);
        }

        return newShape;
    }

    NDShape ONNXToCNTKHelper::FromTensorShape(const TensorShapeProto& tensorShape)
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

    std::vector<bool> ONNXToCNTKHelper::FromTensorShapeAsBool(const TensorShapeProto& tensorShape)
    {
        std::vector<bool> dimensions;
        for (int index = 0; index < tensorShape.dim_size(); index++)
            dimensions.push_back(tensorShape.dim(index).dim_value() == 0 ? false : true);

        // CNTKToONNX ToTensorShape does reverse, need to reverse to restore CNTK shape
        std::reverse(dimensions.begin(), dimensions.end());
        return dimensions;
    }
    
    DataType ONNXToCNTKHelper::FromONNXType(LotusIR::TypeProto type)
    {
        switch (type.mutable_tensor_type()->elem_type())
        {
        case LotusIR::TensorProto_DataType_FLOAT:
            return DataType::Float;
        case LotusIR::TensorProto_DataType_DOUBLE:
            return DataType::Double;
            break;
        default:
            NOT_IMPLEMENTED;
        }
    }

    Constant ONNXToCNTKHelper::CreateConstant(const Node *node)
    {
        NodeAttributes::const_iterator itValue = node->GetAttributes().find("value");
        const LotusIR::TensorProto valueProto = itValue->second.t();
        auto dataType = valueProto.data_type();
        NDShape shape(std::vector<size_t>(valueProto.dims().begin(), valueProto.dims().end()));

        //////LotusIR::NodeArg inputArg = node->OutputDefs()[0];
        //////const LotusIR::TensorShapeProto shapeProto = inputArg.Shape();
        //////NDShape shape = FromTensorShape(shapeProto);

        // CNTK transpose does switch between row major and column major.
        // However it does not change the shape. This makes second transpose
        // wrong. Here we have to construct with unchanged data layout
        // but transposed reshape. Then transpose and reshape to recover
        // the transpose operation. That is to make 2 transpose operation
        // an identity transform.
        NDShape reversedShape = ReverseShape(shape);
        auto totalSize = shape.TotalSize();

        switch (dataType)
        {
        case TensorProto_DataType_FLOAT:
        {
            std::vector<float> data(totalSize);
            for (size_t index = 0; index < totalSize; index++)
            {
                data[index] = valueProto.float_data()[index];
            }

           // TODO: for onnx tensfor the first 1 or 2 are probably batch and channel. the last 2 are kernal size
            if (shape.Rank() <= 2)
            {
                // TODO: data has to be allocated on the specified 'device'?
                NDArrayViewPtr reshaped(new NDArrayView(DataType::Float, reversedShape, &data[0],
                    totalSize * sizeof(float), DeviceDescriptor::CPUDevice()));
                NDArrayViewPtr transposed = reshaped->Transpose();
                NDArrayViewPtr dstFinal = transposed->AsShape(shape);
                Constant constantVariable(dstFinal, ToWString(node->Name()));

                return constantVariable;
            }
            else
            {
                std::vector<float> fullKernekData(totalSize);
                int outputChannels = shape[0], inputChanndels = shape[1];
                NDShape channelKernelShape(std::vector<size_t>(shape.Dimensions().begin() + 2, shape.Dimensions().end()));
                NDShape channelReversedShape = ReverseShape(channelKernelShape);
                int channelKernelSize = channelKernelShape.TotalSize();
                int nonChannelKernelSize = outputChannels * inputChanndels;
                for (int oC = 0; oC < outputChannels; oC++)
                {
                    for (int iC = 0; iC < inputChanndels; iC++)
                    {
                        int channelIndex = (oC * inputChanndels + iC);
                        NDArrayViewPtr channelReshaped(new NDArrayView(DataType::Float, channelReversedShape, 
                            &data[channelIndex * channelKernelSize],
                            channelKernelSize * sizeof(float), DeviceDescriptor::CPUDevice()));
                        NDArrayViewPtr channelTransposed = channelReshaped->Transpose();
                        NDArrayViewPtr channelTransposedDstFinal = channelTransposed->AsShape(channelKernelShape);
                        const float *channelData = channelTransposedDstFinal->DataBuffer<float>();
                        for (int pixel = 0; pixel < channelKernelSize; pixel++)
                        {
                            fullKernekData[pixel * nonChannelKernelSize + channelIndex] = channelData[pixel];
                        }
                    }
                }
                NDArrayViewPtr dstFinal(new  NDArrayView(DataType::Double, channelReversedShape, &fullKernekData[0],
                    totalSize * sizeof(double), DeviceDescriptor::CPUDevice()));
                Constant constantVariable(dstFinal, ToWString(node->Name()));
                return constantVariable;
            }
        }
        break;
        case TensorProto_DataType_DOUBLE:
        {
            std::vector<double> data(totalSize);
            for (size_t index = 0; index < totalSize; index++)
            {
                data[index] = valueProto.float_data()[index];
            }

            // TODO: data has to be allocated on the specified 'device'?
            NDArrayViewPtr reshaped(new NDArrayView(DataType::Double, reversedShape, &data[0],
                totalSize * sizeof(double), DeviceDescriptor::CPUDevice()));
            NDArrayViewPtr transposed = reshaped->Transpose();
            NDArrayViewPtr dstFinal = transposed->AsShape(shape);
            Constant constantVariable(dstFinal, ToWString(node->Name()));

            return constantVariable;
        }
        break;
        default:
            NOT_IMPLEMENTED;
        }

        // do transpose and reshape
    }

    Variable ONNXToCNTKHelper::CreateVariable(const NodeArg *nodeArg)
    {
        // TODO: how to get the datatype?
        auto dataType = TensorProto_DataType_FLOAT;

        const LotusIR::TensorShapeProto shapeProto = nodeArg->Shape();

        NDShape shape = FromTensorShape(shapeProto);
        
        switch (dataType)
        {
        case TensorProto_DataType_FLOAT:
        {
            return InputVariable(shape, DataType::Float, ToWString(nodeArg->Name()));
        }
        case TensorProto_DataType_DOUBLE:
        {
            return InputVariable(shape, DataType::Double, ToWString(nodeArg->Name()));
        }
        default:
            NOT_IMPLEMENTED;
        }
    }

    Variable ONNXToCNTKHelper::CreateVariable(const Node *node)
    {
        // TODO: how to get the datatype?
        auto dataType = TensorProto_DataType_FLOAT;

        LotusIR::NodeArg inputArg = node->OutputDefs()[0];
        const LotusIR::TensorShapeProto shapeProto = inputArg.Shape();
        NDShape shape = FromTensorShape(shapeProto);
 
        switch (dataType)
        {
        case TensorProto_DataType_FLOAT:
        {
            return InputVariable(shape, DataType::Float, ToWString(node->Name()));
        }
        case TensorProto_DataType_DOUBLE:
        {
            return InputVariable(shape, DataType::Double, ToWString(node->Name()));
        }
        default:
            NOT_IMPLEMENTED;
        }
    }

    NDShape ONNXToCNTKHelper::GetNamedAttributeAsShape(const Node *node, const string &attributeName)
    {
        NodeAttributes::const_iterator itValue = node->GetAttributes().find(attributeName);
        const AttributeProto &attributeProto = itValue->second;
        std::vector<int64_t> shape(attributeProto.ints().begin(), attributeProto.ints().end());
        return FromTensorShape(FromINTS(shape));
    }

    std::vector<bool> ONNXToCNTKHelper::GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName)
    {
        NodeAttributes::const_iterator itValue = node->GetAttributes().find(attributeName);
        const AttributeProto &attributeProto = itValue->second;
        std::vector<int64_t> shape(attributeProto.ints().begin(), attributeProto.ints().end());
        return FromTensorShapeAsBool(FromINTS(shape));
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
        auto attributesMap = ONNXToCNTKAttributeNameMapping(onnxOpName);

        if (onnxOpName == "AveragePool" || onnxOpName == "MaxPool")
        {
            NDShape poolingWindowShape = GetNamedAttributeAsShape(node, "kernel_shape");
            NDShape strides = GetNamedAttributeAsShape(node, "strides");
            std::vector<bool> autoPadding = GetNamedAttributeAsShapeBool(node, attributesMap.map[L"autoPadding"]);

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
            NDShape dilation = GetNamedAttributeAsShape(node, attributesMap.map[L"dilation"]);
            std::vector<bool> autoPadding = GetNamedAttributeAsShapeBool(node, attributesMap.map[L"autoPadding"]);

            // TODO: get from node's attributes
            std::vector<bool> sharing({ true });
            size_t reductionRank = 1;
            size_t groups = 1;
            size_t maxTempMemSizeInSamples = 0;

            // TODO: are we sure that convolutionMap and operand are in inputs[1], inputs[0] order?
            Variable convolutionMap = (inputs[0].Shape().Rank() > inputs[1].Shape().Rank()) ? inputs[0] : inputs[1];
            Variable operand = (inputs[0].Shape().Rank() > inputs[1].Shape().Rank()) ? inputs[1] : inputs[0];
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
            Trace2(onnxOpName, cntkFunction, inputs[0], inputs[1]);
            return cntkFunction;
        }
        else if (onnxOpName == "ConvTranspose")
        {
            NDShape strides = GetNamedAttributeAsShape(node, "strides");
            NDShape dilation = GetNamedAttributeAsShape(node, attributesMap.map[L"dilation"]);
            std::vector<bool> autoPadding = GetNamedAttributeAsShapeBool(node, attributesMap.map[L"autoPadding"]);

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
            const Variable& runningCount = inputs[5]; 
            bool spatial = true;
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
            double dropoutRate = GetNamedAttributeAsFloat(node, attributesMap.map[L"dropoutRate"]);
            unsigned long seed = SentinelValueForAutoSelectRandomSeed;
            FunctionPtr cntkFunction = Dropout(operand, dropoutRate, seed, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "RandomUniform")
        {
            const NDShape &shape = GetNamedAttributeAsShape(node, attributesMap.map[L"newShape"]);

            // TODO get from node's attributes
            DataType dataType = DataType::Float;

            double low = GetNamedAttributeAsFloat(node, "low");
            double high = GetNamedAttributeAsFloat(node, "high");
            unsigned long seed = GetNamedAttributeAsInt64(node, attributesMap.map[L"rngSeed"]);
            FunctionPtr cntkFunction = UniformRandom(shape, dataType, low, high, seed, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "RandomNormal")
        {
            const NDShape& shape = GetNamedAttributeAsShape(node, attributesMap.map[L"newShape"]);
            DataType dataType = DataType::Float;
            double mean = GetNamedAttributeAsFloat(node, "mean");
            double scale = GetNamedAttributeAsFloat(node, "scale");
            unsigned long seed = GetNamedAttributeAsInt64(node, attributesMap.map[L"rngSeed"]);
            FunctionPtr cntkFunction = NormalRandom(shape, dataType, mean, scale, seed, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "RandomUniformLike")
        {
            const Variable& operand = inputs[0];
            double low = GetNamedAttributeAsFloat(node, "low");
            double high = GetNamedAttributeAsFloat(node, "high");
            unsigned long seed = GetNamedAttributeAsInt64(node, attributesMap.map[L"rngSeed"]);
            FunctionPtr cntkFunction = UniformRandomLike(operand, low, high, seed, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "RandomNormalLike")
        {
            const Variable& operand = inputs[0];
            double mean = GetNamedAttributeAsFloat(node, "mean");
            double scale = GetNamedAttributeAsFloat(node, "scale");
            unsigned long seed = GetNamedAttributeAsInt64(node, attributesMap.map[L"rngSeed"]);
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
            FunctionPtr cntkFunction = Times(inputs[0], inputs[1], ToWString(node->Name()));
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
            // TODO: get from node's attributes
            Axis axis;
            FunctionPtr cntkFunction = ReduceMax(inputs[0], axis, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "ReduceMin")
        {
            // TODO: get from node's attributes
            Axis axis;
            FunctionPtr cntkFunction = ReduceMin(inputs[0], axis, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "ReduceSum")
        {
            // TODO: get from node's attributes
            Axis axis;
            FunctionPtr cntkFunction = ReduceSum(inputs[0], axis, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "ReduceMean")
        {
            // TODO: get from node's attributes
            Axis axis;
            FunctionPtr cntkFunction = ReduceMean(inputs[0], axis, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "ReduceProd")
        {
            // TODO: get from node's attributes
            Axis axis;
            FunctionPtr cntkFunction = ReduceProd(inputs[0], axis, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "ReduceLogSumExp")
        {
            // TODO: get from node's attributes
            Axis axis;
            FunctionPtr cntkFunction = ReduceLogSum(inputs[0], axis, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "ArgMax")
        {
            // TODO: get from node's attributes
            Axis axis;
            FunctionPtr cntkFunction = Argmax(inputs[0], axis, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "ArgMin")
        {
            // TODO: get from node's attributes
            Axis axis;
            FunctionPtr cntkFunction = Argmin(inputs[0], axis, ToWString(node->Name()));
            return cntkFunction;
        }
        else if (onnxOpName == "Reshape")
        {
            // TODO: get from node's attributes
            // flatten for now
            
            size_t totalSize = inputs[0].Shape().TotalSize();
            NDShape newShape(std::vector<size_t> { totalSize });
            FunctionPtr cntkFunction = Reshape(inputs[0], newShape, ToWString(node->Name()));
            Trace1(onnxOpName, cntkFunction, inputs[0]);
            return cntkFunction;
        }
        else if (onnxOpName == "Concat")
        {
            // TODO: get from node's attributes
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
    };

FunctionPtr ONNXToCNTKHelper::FromONNXNode(const Node *node, ONNXToCNTKMap &constructedNodeMap)
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
                FunctionPtr input = FromONNXNode(inputNode, constructedNodeMap);
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

    FunctionPtr cntkFunction = CreateCNTKNode(node, inputs);
    constructedNodeMap.insert(ONNXToCNTKMap::value_type(node, cntkFunction));
    return cntkFunction;
}


FunctionPtr ONNXToCNTKHelper::CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs)
{
    string onnxOpName = node->OpType();

    if (onnxOpName == "NoOp")
    {
        // TODO: this is for sink or source - what type of variable for it?
        NDShape shape;
        Constant constantVariable(shape, 0.5F, DeviceDescriptor::CPUDevice(), ToWString(node->Name()));
        return constantVariable;
    }
    else if (onnxOpName == "Constant")
    {
        Constant constant = CreateConstant(node);
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

}