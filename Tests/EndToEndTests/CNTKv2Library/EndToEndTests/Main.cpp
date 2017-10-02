//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

#include <iostream>
#include <cstdio>
#include <locale>
#include <codecvt>

using namespace CNTK;
using namespace std::placeholders;

void TrainCifarResnet();
void TrainLSTMSequenceClassifier();
void MNISTClassifierTests();
void TrainSequenceToSequenceTranslator();
void TrainTruncatedLSTMAcousticModelClassifier();
void TestFrameMode();
void TestDistributedCheckpointing();

// #include "../proto/onnx/onnx.h";
// #include "../proto/onnx/CNTKToONNX.h"
// #include "../proto/onnx/ONNXToCNTK.h"
// #include "../proto/onnx/core/graph.h"

std::string ToString(const std::wstring& wstring)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    return converter.to_bytes(wstring);
}

static void PrintGraph(const FunctionPtr& function, int spaces, bool useName = false)
{
    std::vector<Variable> inputs = function->Inputs();
    size_t count = inputs.size();
    if (count == 0)
    {
        std::cout << std::string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + ToString(function->AsString()) << std::endl;
        return;
    }

    for (Variable input : inputs)
    {
        std::cout << std::string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + "->" +
            "(" + ToString(useName ? input.Name() : input.Uid()) + ")" + ToString(input.AsString()) << std::endl;
    }

    for (Variable input : inputs)
    {
        if (input.IsOutput())
        {
            FunctionPtr f = input.Owner();
            PrintGraph(f, spaces + 4);
        }
    }
}

std::vector<std::vector<float>> EvaluateMNIST(FunctionPtr modelFunc)
{
    // evaluate
    Variable inputVar = modelFunc->Arguments()[0];

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = modelFunc->Output();

    const size_t inputDim = 784;
    const size_t numOutputClasses = 10;
    const size_t hiddenLayerDim = 200;

    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";
    auto minibatchSource = TextFormatMinibatchSource(L"Test-28x28_cntk_text_small.txt", { { featureStreamName, inputDim },{ labelsStreamName, numOutputClasses } });
    const size_t minibatchSize = 50;
    DeviceDescriptor device = DeviceDescriptor::CPUDevice();
    auto miniBatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
    MinibatchData &arg = miniBatchData[minibatchSource->StreamInfo(featureStreamName)];

    
    std::unordered_map<Variable, ValuePtr> outputs = { { outputVar, nullptr } };
    ValuePtr featureValue = arg.data;
    std::unordered_map<Variable, ValuePtr> arguments = { { inputVar , featureValue } };

    modelFunc->Evaluate(arguments, outputs);
    ValuePtr outputVal = outputs[outputVar];

    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);

    return outputData;
}

std::vector<std::vector<float>> ValidateSimple(FunctionPtr modelFunc, size_t inputDim)
{
    Variable inputVar = modelFunc->Arguments()[0];
    Variable outputVar = modelFunc->Output();

    std::unordered_map<Variable, ValuePtr> outputs = { { outputVar, nullptr } };
    std::vector<float> batchData = { 1,2,3 };
    ValuePtr featureValue = Value::CreateBatch(NDShape({ inputDim }), batchData, DeviceDescriptor::UseDefaultDevice());
    std::unordered_map<Variable, ValuePtr> arguments = { { inputVar , featureValue } };

    modelFunc->Evaluate(arguments, outputs);
    ValuePtr outputVal = outputs[outputVar];

    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);
    return outputData;
}

std::wstring GetIntermediateCNTKFilename(const std::wstring &savedONNXModelFile)
{
    size_t pos = savedONNXModelFile.find(L".model");
    std::wstring convertedModelFile = savedONNXModelFile.substr(0, pos) + L".CNTKFromONNX.model";
    return convertedModelFile;
}

bool ComapareExpectedActural(const std::vector<std::vector<float>> &expected, const std::vector<std::vector<float>> &actual)
{
    if (expected.size() != actual.size())
    {
        return false;
    }
    for (int i = 0; i < expected.size(); i++)
    {
        if (expected[i].size() != actual[i].size())
        {
            return false;
        }
        for (int j = 0; j < expected[i].size(); j++)
        if (fabs(expected[i][j] - actual[i][j]) > 0.01)
        {
            return false;
        }
    }
    return true;
}

void TestSimple()
{
    size_t inputDim = 3, outputDim = 2;

    auto input = InputVariable({ inputDim }, DataType::Float, L"features");

    float timeParamBuf[6] = { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
    NDShape timeParamShape({ 2, 3 });
    NDArrayViewPtr timesParamAV(new NDArrayView(DataType::Float, timeParamShape, &timeParamBuf[0], 
        inputDim * outputDim * sizeof(float), DeviceDescriptor::UseDefaultDevice()));

    auto timesParam = Parameter(timesParamAV, L"timesParam");

    auto timesFunction = Times(timesParam, input, L"times");
    
    FunctionPtr modelFunc = timesFunction;

    //float plusParamBuf[2] = { 0.2F, 0.3F };
    //NDArrayViewPtr plusParamAV(new NDArrayView(DataType::Float, NDShape({ 2 }), &plusParamBuf[0],
    //    outputDim * sizeof(float), DeviceDescriptor::UseDefaultDevice()));
    //auto plusParam = Parameter(plusParamAV, L"plusParam");
    //FunctionPtr modelFunc = Plus(plusParam, timesFunction, L"timesPlus");

    std::vector<std::vector<float>> expected = ValidateSimple(modelFunc, inputDim);

    const std::wstring savedONNXModelFile = L"TestSimple.CNTK.model";
    modelFunc->Save(savedONNXModelFile, ModelFormat::ONNX);

    std::wstring convertedModelFile = GetIntermediateCNTKFilename(savedONNXModelFile);
    FunctionPtr modelConvertedFunc = Function::Load(convertedModelFile);

    std::vector<std::vector<float>> actual = ValidateSimple(modelConvertedFunc, inputDim);
    
    bool b = ComapareExpectedActural(expected, actual);
    assert(b);
}

std::vector<std::vector<float>> ValidateSimpleConv(FunctionPtr modelFunc, 
    int imageWidth, int imageHeight, int channels, int batchs)
{
    Variable inputVar = modelFunc->Arguments()[0];
    Variable outputVar = modelFunc->Output();

    std::unordered_map<Variable, ValuePtr> outputs = { { outputVar, nullptr } };
    std::vector<float> batchData(imageWidth * imageHeight * channels * batchs, 1);
    ValuePtr featureValue = Value::CreateBatch(NDShape({ (size_t)imageWidth, (size_t)imageHeight, (size_t)channels }),
        batchData, DeviceDescriptor::UseDefaultDevice());
    std::unordered_map<Variable, ValuePtr> arguments = { { inputVar , featureValue } };

    modelFunc->Evaluate(arguments, outputs);
    ValuePtr outputVal = outputs[outputVar];

    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);
    return outputData;
}


void TestSimpleConv()
{
    int kernelWidth = 3, kernelHeight = 3, numInputChannels = 1, outFeatureMapCount = 1;
    int kernalDataSize = kernelWidth * kernelHeight * numInputChannels * outFeatureMapCount;
    std::vector<float> kernalData;
    for (int i = 0; i < kernalDataSize; i++)
    {
        kernalData.push_back((float)i + 1);
    }

    NDShape kernalShape({ (size_t)kernelWidth, (size_t)kernelHeight, (size_t)numInputChannels, (size_t)outFeatureMapCount });
    NDArrayViewPtr kernalParamAV(new NDArrayView(DataType::Float, kernalShape, &kernalData[0],
        kernalDataSize * sizeof(float), DeviceDescriptor::UseDefaultDevice()));

    Parameter convParams = Parameter(kernalParamAV, L"kernal");

    int imageWidth = 28, imageHeight = 28, channels = 1, batchs = 1;
    NDShape imageShape({ (size_t)imageWidth, (size_t)imageHeight, (size_t)channels });
    Variable features = InputVariable(imageShape, DataType::Float, "features");
    NDShape strides({ 1,1,1});
    FunctionPtr convFunction = Convolution(convParams, features, strides);

    std::vector<std::vector<float>> expected = ValidateSimpleConv(convFunction,
        imageWidth, imageHeight, channels, batchs);

    const std::wstring savedONNXModelFile = L"TestSimpleConv.CNTK.model";
    convFunction->Save(savedONNXModelFile, ModelFormat::ONNX);

    std::wstring convertedModelFile = GetIntermediateCNTKFilename(savedONNXModelFile);
    FunctionPtr modelConvertedFunc = Function::Load(convertedModelFile);

    std::vector<std::vector<float>> actual = ValidateSimpleConv(modelConvertedFunc, 
        imageWidth, imageHeight, channels, batchs);

    bool b = ComapareExpectedActural(expected, actual);
    assert(b);
}

void TestMNISTSimpleFeedForward()
{
    const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/feedForward_MNIST_classifier_only.net0";
    FunctionPtr cntkModel = Function::Load(cntkModelFile, DeviceDescriptor::UseDefaultDevice(), ModelFormat::CNTKv2);

    std::vector<std::vector<float>> expected = EvaluateMNIST(cntkModel);

    const std::wstring onnxModelFile = L"E:/LiqunWA/CNTK/ONNX/feedForward_MNIST_classifier_only_ONNX.net0.model";
    cntkModel->Save(onnxModelFile, ModelFormat::ONNX);

    std::wstring convertedModelFile = GetIntermediateCNTKFilename(onnxModelFile);
    FunctionPtr modelConvertedFunc = Function::Load(convertedModelFile);
    std::vector<std::vector<float>> actual = EvaluateMNIST(modelConvertedFunc);

    bool b = ComapareExpectedActural(expected, actual);
    assert(b);
}

void TestMNISTConvnet()
{
    const std::wstring savedONNXModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTConvolutionONNX.model";
    const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTConvolution.model";
    const std::wstring cntkModelFile2 = L"E:/LiqunWA/CNTK/ONNX/MNISTConvolution.model";

    FunctionPtr cntkModel = Function::Load(cntkModelFile, DeviceDescriptor::UseDefaultDevice(), ModelFormat::CNTKv2);

    FunctionPtr smallFunction = AsComposite(cntkModel->FindByName(L"pooling1-Convolution"));
    std::vector<std::vector<float>> expected = EvaluateMNIST(smallFunction);
    // EvaluateMNIST(cntkModel);

    cntkModel->Save(savedONNXModelFile, ModelFormat::ONNX);

    std::wstring convertedModelFile = GetIntermediateCNTKFilename(savedONNXModelFile);
    FunctionPtr modelConvertedFunc = Function::Load(convertedModelFile);
    FunctionPtr smallFunction2 = AsComposite(modelConvertedFunc->FindByName(L"Convolution6"));

    std::vector<std::vector<float>> actual = EvaluateMNIST(smallFunction2);
    bool b = ComapareExpectedActural(expected, actual);
    assert(b);

    FunctionPtr cntkModelFromONNX = Function::Load(savedONNXModelFile, DeviceDescriptor::UseDefaultDevice(), ModelFormat::ONNX);
    EvaluateMNIST(cntkModelFromONNX);
}

void TestLoadONNX()
{
    {
        // vgg model load failed
        const std::wstring vgg16 = L"E:/LiqunWA/CNTK/ONNX/vgg16/vgg16/graph.pb";
        FunctionPtr vgg16Model = Function::Load(vgg16, DeviceDescriptor::UseDefaultDevice(), ModelFormat::ONNX);

        const std::wstring denseNet = L"E:/LiqunWA/CNTK/ONNX/densenet121/densenet121/graph.pb";
        FunctionPtr denseNetModel = Function::Load(denseNet, DeviceDescriptor::UseDefaultDevice(), ModelFormat::ONNX);
    }
}

void TestPytorchSuperResModel()
{
    // std::wstring onnxModelFile = L"E:/LiqunWA/CNTK/ONNX/super_resolution.onnx";
    // std::wstring onnxModelFile = L"E:/LiqunWA/CNTK/ONNX/fully-connected_3To2.onnx";
    const std::wstring onnxModelFile = L"E:/LiqunWA/CNTK/ONNX/densenet121/densenet121/graph.pb";
    FunctionPtr fromOnnxModel = Function::Load(onnxModelFile, DeviceDescriptor::UseDefaultDevice(), ModelFormat::ONNX);
}

void RunLotus(DeviceDescriptor device)
{
    if (!DeviceDescriptor::TrySetDefaultDevice(device))
    {
        std::cout << "failed to set default device" << std::endl;
    }
    
    // TestPytorchSuperResModel();
    TestSimple();
    TestMNISTSimpleFeedForward();

    TestSimpleConv();
    TestMNISTConvnet();


    // TestLoadONNX();


    // CNTK function saved as ONNX. 
    // Loaded but graph is not the same as in memory graph. See ONNX.Save commented code.
    // FunctionPtr cntkModel = Function::Load(savedONNXModelFile, device, ModelFormat::ONNX);

    // following are experiment code
    {
        const std::wstring savedONNXModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTConvolutionONNX.model";
        const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTConvolution.model";
        const std::wstring cntkModelFile2 = L"E:/LiqunWA/CNTK/ONNX/MNISTConvolution.model";

        //const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTMLP.model";
        //const std::wstring savedONNXModelFile = L"E:/LiqunWA/CNTK/ONNX/MNISTMLPONNX.model";

        //const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/LogisticRegression.model";
        //const std::wstring savedONNXModelFile = L"E:/LiqunWA/CNTK/ONNX/LogisticRegressionONNX.model";

        FunctionPtr cntkModel = Function::Load(cntkModelFile, device, ModelFormat::CNTKv2);

        cntkModel->Save(savedONNXModelFile, ModelFormat::ONNX);
        // TODO PrintGraph does not work for some reason
        // PrintGraph(cntkModel->RootFunction(), 0);

        // failed here because graph save/load operation is not idempotent
        FunctionPtr cntkModelFromONNX = Function::Load(savedONNXModelFile, device, ModelFormat::ONNX);
        // PrintGraph(cntkModelFromONNX, 0);
        cntkModelFromONNX->Save(cntkModelFile2, ModelFormat::CNTKv2);
    }
}

int main(int argc, char *argv[])
{
#if defined(_MSC_VER)
    // in case of asserts in debug mode, print the message into stderr and throw exception
    if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1) {
        fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        return -1;
    }
#endif

    RunLotus(DeviceDescriptor::CPUDevice());

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

#ifndef CPUONLY
    fprintf(stderr, "Run tests using GPU build.\n");
#else
    fprintf(stderr, "Run tests using CPU-only build.\n");
#endif

    if (argc > 2)
    {
        if (argc == 3 && !std::string(argv[1]).compare("Distribution")) {
            {
                auto communicator = MPICommunicator();
                std::string logFilename = argv[2] + std::to_string(communicator->CurrentWorker().m_globalRank);
                auto result = freopen(logFilename.c_str(), "w", stdout);
                if (result == nullptr)
                {
                    fprintf(stderr, "Could not redirect stdout.\n");
                    return -1;
                }
            }

            TestFrameMode();

            TestDistributedCheckpointing();

            std::string testsPassedMsg = "\nCNTKv2Library-Distribution tests: Passed\n";

            printf("%s", testsPassedMsg.c_str());

            fflush(stdout);
            DistributedCommunicator::Finalize();
            fclose(stdout);
            return 0;
        }
        else
        {
            fprintf(stderr, "Wrong number of arguments.\n");
            return -1;
        }
    }

    std::string testName(argv[1]);

    if (!testName.compare("CifarResNet"))
    {
        if (ShouldRunOnGpu())
        {
            fprintf(stderr, "Run test on a GPU device.\n");
            TrainCifarResnet();
        }

        if (ShouldRunOnCpu())
        {
            fprintf(stderr, "Cannot run TrainCifarResnet test on a CPU device.\n");
        }
    }
    else if (!testName.compare("LSTMSequenceClassifier"))
    {
        TrainLSTMSequenceClassifier();
    }
    else if (!testName.compare("MNISTClassifier"))
    {
        MNISTClassifierTests();
    }
    else if (!testName.compare("SequenceToSequence"))
    {
        TrainSequenceToSequenceTranslator();
    }
    else if (!testName.compare("TruncatedLSTMAcousticModel"))
    {
        TrainTruncatedLSTMAcousticModelClassifier();
    }
    else
    {
        fprintf(stderr, "End to end test not found.\n");
        return -1;
    }

    std::string testsPassedMsg = "\nCNTKv2Library-" + testName + " tests: Passed\n";

    fprintf(stderr, "%s", testsPassedMsg.c_str());
    fflush(stderr);

#if defined(_MSC_VER)
    _CrtSetReportHook2(_CRT_RPTHOOK_REMOVE, HandleDebugAssert);
#endif
}
