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
        if (input.Owner() != NULL)
        {
            FunctionPtr f = input.Owner();
            PrintGraph(f, spaces + 4);
        }
    }
}

void EvaluateMNIST(FunctionPtr modelFunc)
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
}

void ValidateSimple(FunctionPtr modelFunc, size_t inputDim)
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
}

std::wstring GetIntermediateCNTKFilename(const std::wstring &savedONNXModelFile)
{
    size_t pos = savedONNXModelFile.find(L".model");
    std::wstring convertedModelFile = savedONNXModelFile.substr(0, pos) + L".CNTKFromONNX.model";
    return convertedModelFile;
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

    ValidateSimple(modelFunc, inputDim);

    const std::wstring savedONNXModelFile = L"TestSimple.CNTK.model";
    modelFunc->Save(savedONNXModelFile, ModelFormat::ONNX);

    std::wstring convertedModelFile = GetIntermediateCNTKFilename(savedONNXModelFile);
    FunctionPtr modelConvertedFunc = Function::Load(convertedModelFile);

    ValidateSimple(modelConvertedFunc, inputDim);

}

void TestMNISTSimpleFeedForward()
{
    const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/feedForward_MNIST_classifier_only.net0";
    FunctionPtr cntkModel = Function::Load(cntkModelFile, DeviceDescriptor::UseDefaultDevice(), ModelFormat::CNTKv2);
    EvaluateMNIST(cntkModel);

    const std::wstring onnxModelFile = L"E:/LiqunWA/CNTK/ONNX/feedForward_MNIST_classifier_only_ONNX.net0.model";
    cntkModel->Save(onnxModelFile, ModelFormat::ONNX);

    std::wstring convertedModelFile = GetIntermediateCNTKFilename(onnxModelFile);
    FunctionPtr modelConvertedFunc = Function::Load(convertedModelFile);
    EvaluateMNIST(modelConvertedFunc);
}

//{
//    // TODO: handle block and flaceholder nodes
//    //const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/feedForward_classifier_only.net0";
//    //const std::wstring savedONNXModelFile = L"E:/LiqunWA/CNTK/ONNX/feedForward_classifier_only_onnx.net0";
//
//    const std::wstring cntkModelFile = L"E:/LiqunWA/CNTK/ONNX/feedForward_MNIST_classifier_only.net0";
//    const std::wstring savedONNXModelFile = L"E:/LiqunWA/CNTK/ONNX/feedForward_MNIST_classifier_only_onnx.net0";
//    //
//    FunctionPtr cntkModel = Function::Load(cntkModelFile, device, ModelFormat::CNTKv2);
//    EvaluateMNIST(cntkModel);
//
//    cntkModel->Save(savedONNXModelFile, ModelFormat::ONNX);
//
//    FunctionPtr cntkModelFromONNX = Function::Load(savedONNXModelFile, device, ModelFormat::ONNX);
//    EvaluateMNIST(cntkModelFromONNX);
//}


void RunLotus(DeviceDescriptor device)
{
    if (!DeviceDescriptor::TrySetDefaultDevice(device))
    {
        std::cout << "failed to set default device" << std::endl;
    }
    
    // TestSimple();
    TestMNISTSimpleFeedForward();

    {
        // vgg model load failed
        //const std::wstring vgg16 = L"E:/LiqunWA/CNTK/ONNX/vgg16/vgg16/graph.pb";
        //FunctionPtr cntkModel = Function::Load(vgg16, device, ModelFormat::ONNX);
    }

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
