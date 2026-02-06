#include "cus_ln_plugin.h"
#include "cus_ln.h"
#include <cassert>//assert

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::CustomLn3dEps6Plugin;
// using nvinfer1::plugin::CustomLn3dEps6PluginCreator;

namespace
{
    char const* const kCUSTOM_OP_PLUGIN_VERSION{"1"};
    char const* const kCUSTOM_OP_PLUGIN_NAME{"CusLnm3d_eps6"};
} // namespace

CustomLn3dEps6Plugin::CustomLn3dEps6Plugin(const CustomLn3dEps6Para& param)
    : mParam(param)
{
}

CustomLn3dEps6Plugin::CustomLn3dEps6Plugin(void const* data, size_t length)
{
    deserialize(static_cast<int8_t const*>(data), length);
}

void CustomLn3dEps6Plugin::deserialize(int8_t const* data, size_t length)
{
    auto const* d{data};
    mParam = read<CustomLn3dEps6Para>(d);
    assert(d == data + length);
}

char const* CustomLn3dEps6Plugin::getPluginType() const noexcept
{
    return kCUSTOM_OP_PLUGIN_NAME;
}

char const* CustomLn3dEps6Plugin::getPluginVersion() const noexcept
{
    return kCUSTOM_OP_PLUGIN_VERSION;
}

int32_t CustomLn3dEps6Plugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t CustomLn3dEps6Plugin::initialize() noexcept
{
    return 0;
}

void CustomLn3dEps6Plugin::terminate() noexcept 
{
}

size_t CustomLn3dEps6Plugin::getSerializationSize() const noexcept
{
    return sizeof(CustomLn3dEps6Para);
}

void CustomLn3dEps6Plugin::serialize(void* buffer) const noexcept
{
    // serialize：void* -> CustomLn3dEps6Para
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mParam);
    assert(d == a + getSerializationSize());
}

void CustomLn3dEps6Plugin::destroy() noexcept
{
    delete this;
}

void CustomLn3dEps6Plugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* CustomLn3dEps6Plugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType CustomLn3dEps6Plugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    /*
    0: output - float32/float16
    */
    // use the same datatype as the input tensor, tensor,gamma,beta
    return inputTypes[LN_TENSOR_IDX];
}

IPluginV2DynamicExt* CustomLn3dEps6Plugin::clone() const noexcept
{
    try
    {
        auto* plugin = new CustomLn3dEps6Plugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs CustomLn3dEps6Plugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        /*
        0: output - shape(-1, x, y)
        */
        DimsExprs out_dim;

        // Standard
        assert(outputIndex == 0);

        if (outputIndex == 0)
        {
            out_dim.nbDims = LN_TENSOR_NUM_DIM;
            out_dim.d[0] = inputs[LN_TENSOR_IDX].d[0];
            out_dim.d[1] = inputs[LN_TENSOR_IDX].d[1];
            out_dim.d[2] = inputs[LN_TENSOR_IDX].d[2];
        }

        return out_dim;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool CustomLn3dEps6Plugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut[pos].format != PluginFormat::kLINEAR)
        return false;

    assert(nbInputs == 3);
    assert(nbOutputs == 1);
    assert(0 <= pos && pos <= 3);

    // all other inputs/outputs: fp32 or fp16
    return (inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kFLOAT) && (inOut[0].type == inOut[pos].type);
}

void CustomLn3dEps6Plugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        assert(nbInputs == 3);
        assert(nbOutputs == 1);
        mParam.datatype = in[LN_TENSOR_IDX].desc.type;

        assert(in[LN_TENSOR_IDX].desc.dims.nbDims == LN_TENSOR_NUM_DIM);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t CustomLn3dEps6Plugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    size_t total = 0;
    return total;
}

int32_t CustomLn3dEps6Plugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        mParam.batchSize = inputDesc[LN_TENSOR_IDX].dims.d[0];
        mParam.seq_len = inputDesc[LN_TENSOR_IDX].dims.d[1];
        mParam.hidden_dim = inputDesc[LN_TENSOR_IDX].dims.d[2];

        void const* const Input = inputs[LN_TENSOR_IDX];
        void const* const gamme = inputs[LN_GAMMA_IDX];
        void const* const beta = inputs[LN_BETA_IDX];
        
        void* Output = outputs[0];

        if (mParam.datatype == DataType::kFLOAT)
        {
            forwardGpu<float>(gamme, beta, Input, Output, workspace, stream);
        }
        else if (mParam.datatype == DataType::kHALF)
        {
            forwardGpu_fp16<half>(gamme, beta, Input, Output, workspace, stream);
        }
        else
        {
            return -1;
        }

        return 0;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

template <typename T>
void CustomLn3dEps6Plugin::forwardGpu(const void* gamma, const void* beta, const void* Input, void* Output, void* workspace, cudaStream_t stream)
{
    layernorm_fp32x4_ld_launch((T*)Input, (T*)Output, (T*)gamma, (T*)beta, mParam.batchSize, mParam.seq_len, mParam.hidden_dim, stream);
}

template <typename T>
void CustomLn3dEps6Plugin::forwardGpu_fp16(const void* gamma, const void* beta, const void* Input, void* Output, void* workspace, cudaStream_t stream)
{
    layernorm_f16x8_pack_f16_acc_ld_launch((T*)Input, (T*)Output, (T*)gamma, (T*)beta, mParam.batchSize, mParam.seq_len, mParam.hidden_dim, stream);
}



// CustomLn3dEps6PluginCreator
CustomLn3dEps6PluginCreator::CustomLn3dEps6PluginCreator()
    : mParam{}
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* CustomLn3dEps6PluginCreator::getPluginName() const noexcept
{
    return kCUSTOM_OP_PLUGIN_NAME;
}

char const* CustomLn3dEps6PluginCreator::getPluginVersion() const noexcept
{
    return kCUSTOM_OP_PLUGIN_VERSION;
}

PluginFieldCollection const* CustomLn3dEps6PluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* CustomLn3dEps6PluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        auto* plugin = new CustomLn3dEps6Plugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* CustomLn3dEps6PluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* plugin = new CustomLn3dEps6Plugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void CustomLn3dEps6PluginCreator::validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, 
    PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        requiredFieldNames.erase(fc->fields[i].name);
    }
    if (!requiredFieldNames.empty())
    {
        std::stringstream msg{};
        msg << "PluginFieldCollection missing required fields: {";
        char const* separator = "";
        for (auto const& field : requiredFieldNames)
        {
            msg << separator << field;
            separator = ", ";
        }
        msg << "}";
        std::string msg_str = msg.str();
        std::cout << msg_str << std::endl;
    }
}
