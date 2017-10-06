#include <fcntl.h>
#include <fstream>
#include <io.h>
#include "model.h"

namespace LotusIR
{
    Model::Model(const std::string& p_graphName, bool p_isONNX)
    {
        m_graph.reset(new Graph(p_graphName, p_isONNX));
    }

    Model::Model(const std::string& p_graphName,
        const std::string& p_graphDocString)
    {
        m_graph.reset(new Graph(p_graphName, p_graphDocString));
    }

    Model::Model(const std::string& p_graphName,
        const std::string& p_graphDocString,
        VERSION p_irVersion,
        const std::string& p_producerName,
        const std::string& p_producerVersion,
        const std::string& p_domain,
        VERSION p_modelVersion,
        const std::string& p_docString,
        const std::string& p_modelAuthor,
        const std::string& p_modelLicense)
    {
        m_graph.reset(new Graph(p_graphName, p_graphDocString));
        m_modelProto.set_ir_version(p_irVersion);
        m_modelProto.set_producer_name(p_producerName);
        m_modelProto.set_producer_version(p_producerVersion);
        m_modelProto.set_domain(p_domain);
        m_modelProto.set_model_version(p_modelVersion);
        m_modelProto.set_doc_string(p_docString);
        m_modelProto.set_model_author(p_modelAuthor);
        m_modelProto.set_model_license(p_modelLicense);
    }

    Model::Model(const ModelProto& p_modelProto)
    {
        m_modelProto = p_modelProto;
        if (m_modelProto.has_graph())
        {
            m_graph.reset(new Graph(m_modelProto.graph()));
        }
    }

    VERSION Model::IrVersion() const
    {
        if (m_modelProto.has_ir_version())
        {
            return m_modelProto.ir_version();
        }
        return c_noVersion;
    }

    void Model::SetIrVersion(VERSION p_irVersion)
    {
        m_modelProto.set_ir_version(p_irVersion);
    }

    const std::string& Model::ProducerName() const
    {
        return m_modelProto.producer_name();
    }

    void Model::SetProducerName(const std::string& p_producerName)
    {
        m_modelProto.set_producer_name(p_producerName);
    }

    const std::string& Model::ProducerVersion() const
    {
        return m_modelProto.producer_version();
    }

    void Model::SetProducerVersion(const std::string& p_producerVersion)
    {
        m_modelProto.set_producer_version(p_producerVersion);
    }

    const std::string& Model::Domain() const
    {
        return m_modelProto.domain();
    }

    void Model::SetDomain(const std::string& p_domain)
    {
        m_modelProto.set_domain(p_domain);
    }

    VERSION Model::ModelVersion() const
    {
        if (m_modelProto.has_model_version())
        {
            return m_modelProto.model_version();
        }
        return c_noVersion;
    }

    void Model::SetModelversion(VERSION p_modelVersion)
    {
        m_modelProto.set_model_version(p_modelVersion);
    }

    const std::string& Model::DocString() const
    {
        return m_modelProto.doc_string();
    }

    void Model::SetDocString(const std::string& p_docString)
    {
        m_modelProto.set_doc_string(p_docString);
    }

    const std::string& Model::ModelAuthor() const
    {
        return m_modelProto.model_author();
    }

    void Model::SetModelAuthor(const std::string& p_modelAuthor)
    {
        m_modelProto.set_model_author(p_modelAuthor);
    }

    const std::string& Model::ModelLicense() const
    {
        return m_modelProto.model_license();
    }

    void Model::SetModelLicense(const std::string& p_modelLicense)
    {
        m_modelProto.set_model_license(p_modelLicense);
    }

    Graph* Model::MainGraph()
    {
        return m_graph.get();
    }

    const ModelProto& Model::ToProto()
    {
        *(m_modelProto.mutable_graph()) = m_graph->ToGraphProto();
        return m_modelProto;
    }

    bool Model::Save(const ModelProto& p_modelProto, const std::wstring& p_filePath)
    {
#if NOT_SUPPORTS_IOSTREAMS
        int fd;
        errno_t err = _wsopen_s(&fd, p_filePath.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
        if (err) return false;
        bool result = p_modelProto.SerializeToFileDescriptor(fd);
        _close(fd);
#else
        std::fstream outputFileStream(p_filePath, std::ios::out | std::ios::binary);
        bool result = p_modelProto.SerializeToOstream(&outputFileStream);
        outputFileStream.close();
#endif
        return result;
    }

    bool Model::Save(Model& p_model, const std::wstring& p_filePath)
    {
        Status status = p_model.MainGraph()->Resolve();
        if (!status.Ok())
        {
            return false;
        }

        auto& modelProto = p_model.ToProto();
        return Save(modelProto, p_filePath);
    }

    // Load a ModelProto from a file.
    bool Model::Load(const std::wstring& p_filePath, /*out*/ ModelProto* p_modelProto)
    {
        if (nullptr == p_modelProto)
        {
            return false;
        }
#if NOT_SUPPORTS_IOSTREAMS
        int fd;
        errno_t err = _wsopen_s(&fd, p_filePath.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
        if (err) return false;
        bool result = p_modelProto->ParsePartialFromFileDescriptor(fd);
        _close(fd);
#else
        std::fstream inputFileStream(p_filePath, std::ios::in | std::ios::binary);
        if (!inputFileStream)
        {
            return false;
        }

        bool result = p_modelProto->ParsePartialFromIstream(&inputFileStream);
        inputFileStream.close();
#endif

        return result;
    }

    std::shared_ptr<Model> Model::Load(const std::wstring& p_filePath)
    {
        ModelProto modelProto;
        bool result = Load(p_filePath, &modelProto);
        if (!result)
        {
            return nullptr;
        }
        auto model = std::shared_ptr<Model>(new Model(modelProto));
        auto status = model->MainGraph()->Resolve();
        if (status.Ok())
        {
            return model;
        }
        return nullptr;
    }
}