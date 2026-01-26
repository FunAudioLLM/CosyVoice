#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cstring>
#include <string>
#include <memory>
#include <iostream>
#include <vector>
#include <set>

#define LN_TENSOR_NUM_DIM 3
#define LN_TENSOR_IDX 0
#define LN_GAMMA_IDX 1
#define LN_BETA_IDX 2

struct CustomLn3dEps6Para
{
    int32_t batchSize = -1;
    int32_t seq_len = -1;
    int32_t hidden_dim = -1;
    nvinfer1::DataType datatype = nvinfer1::DataType::kFLOAT;
};

namespace nvinfer1
{
namespace plugin
{
    // Write values into buffer
    template <typename Type, typename BufferType>
    void write(BufferType*& buffer, Type const& val)
    {
        static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
        std::memcpy(buffer, &val, sizeof(Type));
        buffer += sizeof(Type);
    }

    // Read values from buffer
    template <typename OutType, typename BufferType>
    OutType read(BufferType const*& buffer)
    {
        static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
        OutType val{};
        std::memcpy(&val, static_cast<void const*>(buffer), sizeof(OutType));
        buffer += sizeof(OutType);
        return val;
    }

    inline void caughtError(std::exception const& e)
    {
        std::cout << e.what() << std::endl;
    }

    class CustomLn3dEps6Plugin : public IPluginV2DynamicExt
    {
    public:
        explicit CustomLn3dEps6Plugin(const CustomLn3dEps6Para& param);
        CustomLn3dEps6Plugin(void const* data, size_t length);
        ~CustomLn3dEps6Plugin() override = default;

        // IPluginV2 methods
        char const* getPluginType() const noexcept override;
        char const* getPluginVersion() const noexcept override;
        int32_t getNbOutputs() const noexcept override;
        int32_t initialize() noexcept override;
        void terminate() noexcept override;
        size_t getSerializationSize() const noexcept override;
        void serialize(void* buffer) const noexcept override;
        void destroy() noexcept override;
        void setPluginNamespace(char const* libNamespace) noexcept override;
        char const* getPluginNamespace() const noexcept override;

        // IPluginV2Ext methods
        nvinfer1::DataType getOutputDataType(
            int32_t index, nvinfer1::DataType const* inputType, int32_t nbInputs) const noexcept override;

        // IPluginV2DynamicExt methods
        IPluginV2DynamicExt* clone() const noexcept override;
        DimsExprs getOutputDimensions(
            int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
        bool supportsFormatCombination(
            int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
        void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
            int32_t nbOutputs) noexcept override;
        size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
            int32_t nbOutputs) const noexcept override;
        int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
            void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    protected:
        CustomLn3dEps6Para mParam{};
        std::string mNamespace;

    private:
        void deserialize(int8_t const* data, size_t length);
        template <typename T>
        void forwardGpu(const void* gamma, const void* beta, const void* Input, void* Output, void* workspace, cudaStream_t stream);
        template <typename T>
        void forwardGpu_fp16(const void* gamma, const void* beta, const void* Input, void* Output, void* workspace, cudaStream_t stream);
    };

    class CustomLn3dEps6PluginCreator : public nvinfer1::IPluginCreator
    {
    public:
        CustomLn3dEps6PluginCreator();
        ~CustomLn3dEps6PluginCreator() override = default;

        char const* getPluginName() const noexcept override;
        char const* getPluginVersion() const noexcept override;
        PluginFieldCollection const* getFieldNames() noexcept override;

        IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
        IPluginV2DynamicExt* deserializePlugin(
            char const* name, void const* serialData, size_t serialLength) noexcept override;

        void setPluginNamespace(char const* libNamespace) noexcept override
        {
            mNamespace = libNamespace;
        }

        char const* getPluginNamespace() const noexcept override
        {
            return mNamespace.c_str();
        }

    protected:
        PluginFieldCollection mFC;
        CustomLn3dEps6Para mParam;
        std::vector<PluginField> mPluginAttributes;
        std::string mPluginName;
        std::string mNamespace;

    private:
        void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, 
            PluginFieldCollection const* fc);
    };

    REGISTER_TENSORRT_PLUGIN(CustomLn3dEps6PluginCreator);
}
}
