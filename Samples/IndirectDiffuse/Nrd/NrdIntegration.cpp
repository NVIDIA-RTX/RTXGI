/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "NrdIntegration.h"

static_assert(NRD_VERSION_MAJOR > 4 || (NRD_VERSION_MAJOR == 4 && NRD_VERSION_MINOR >= 18), "Unsupported NRD version!");

#include "../RenderTargets.h"
#include <nvrhi/utils.h>
#include <donut/core/math/math.h>
#include <donut/engine/View.h>
#include <donut/engine/ShaderFactory.h>
#include <algorithm>
#include <sstream>
#include <donut/core/log.h>

static nvrhi::Format GetNvrhiFormat(nrd::Format format)
{
    switch (format)
    {
    case nrd::Format::R8_UNORM:
        return nvrhi::Format::R8_UNORM;
    case nrd::Format::R8_SNORM:
        return nvrhi::Format::R8_SNORM;
    case nrd::Format::R8_UINT:
        return nvrhi::Format::R8_UINT;
    case nrd::Format::R8_SINT:
        return nvrhi::Format::R8_SINT;
    case nrd::Format::RG8_UNORM:
        return nvrhi::Format::RG8_UNORM;
    case nrd::Format::RG8_SNORM:
        return nvrhi::Format::RG8_SNORM;
    case nrd::Format::RG8_UINT:
        return nvrhi::Format::RG8_UINT;
    case nrd::Format::RG8_SINT:
        return nvrhi::Format::RG8_SINT;
    case nrd::Format::RGBA8_UNORM:
        return nvrhi::Format::RGBA8_UNORM;
    case nrd::Format::RGBA8_SNORM:
        return nvrhi::Format::RGBA8_SNORM;
    case nrd::Format::RGBA8_UINT:
        return nvrhi::Format::RGBA8_UINT;
    case nrd::Format::RGBA8_SINT:
        return nvrhi::Format::RGBA8_SINT;
    case nrd::Format::RGBA8_SRGB:
        return nvrhi::Format::SRGBA8_UNORM;
    case nrd::Format::R16_UNORM:
        return nvrhi::Format::R16_UNORM;
    case nrd::Format::R16_SNORM:
        return nvrhi::Format::R16_SNORM;
    case nrd::Format::R16_UINT:
        return nvrhi::Format::R16_UINT;
    case nrd::Format::R16_SINT:
        return nvrhi::Format::R16_SINT;
    case nrd::Format::R16_SFLOAT:
        return nvrhi::Format::R16_FLOAT;
    case nrd::Format::RG16_UNORM:
        return nvrhi::Format::RG16_UNORM;
    case nrd::Format::RG16_SNORM:
        return nvrhi::Format::RG16_SNORM;
    case nrd::Format::RG16_UINT:
        return nvrhi::Format::RG16_UINT;
    case nrd::Format::RG16_SINT:
        return nvrhi::Format::RG16_SINT;
    case nrd::Format::RG16_SFLOAT:
        return nvrhi::Format::RG16_FLOAT;
    case nrd::Format::RGBA16_UNORM:
        return nvrhi::Format::RGBA16_UNORM;
    case nrd::Format::RGBA16_SNORM:
        return nvrhi::Format::RGBA16_SNORM;
    case nrd::Format::RGBA16_UINT:
        return nvrhi::Format::RGBA16_UINT;
    case nrd::Format::RGBA16_SINT:
        return nvrhi::Format::RGBA16_SINT;
    case nrd::Format::RGBA16_SFLOAT:
        return nvrhi::Format::RGBA16_FLOAT;
    case nrd::Format::R32_UINT:
        return nvrhi::Format::R32_UINT;
    case nrd::Format::R32_SINT:
        return nvrhi::Format::R32_SINT;
    case nrd::Format::R32_SFLOAT:
        return nvrhi::Format::R32_FLOAT;
    case nrd::Format::RG32_UINT:
        return nvrhi::Format::RG32_UINT;
    case nrd::Format::RG32_SINT:
        return nvrhi::Format::RG32_SINT;
    case nrd::Format::RG32_SFLOAT:
        return nvrhi::Format::RG32_FLOAT;
    case nrd::Format::RGB32_UINT:
        return nvrhi::Format::RGB32_UINT;
    case nrd::Format::RGB32_SINT:
        return nvrhi::Format::RGB32_SINT;
    case nrd::Format::RGB32_SFLOAT:
        return nvrhi::Format::RGB32_FLOAT;
    case nrd::Format::RGBA32_UINT:
        return nvrhi::Format::RGBA32_UINT;
    case nrd::Format::RGBA32_SINT:
        return nvrhi::Format::RGBA32_SINT;
    case nrd::Format::RGBA32_SFLOAT:
        return nvrhi::Format::RGBA32_FLOAT;
    case nrd::Format::R10_G10_B10_A2_UNORM:
        return nvrhi::Format::R10G10B10A2_UNORM;
    case nrd::Format::R10_G10_B10_A2_UINT:
        return nvrhi::Format::UNKNOWN; // not representable and not used
    case nrd::Format::R11_G11_B10_UFLOAT:
        return nvrhi::Format::R11G11B10_FLOAT;
    case nrd::Format::R9_G9_B9_E5_UFLOAT:
        return nvrhi::Format::UNKNOWN; // not representable and not used
    default:
        return nvrhi::Format::UNKNOWN;
    }
}

// NRD reports each pipeline's shader as "fileName|macro1=value1|macro2=value2..."; the macros select the
// ShaderMake permutation compiled from NRD's own Shaders.cfg.
static std::string ParseShaderIdentifier(const char* shaderIdentifier, std::vector<donut::engine::ShaderMacro>& macros)
{
    std::stringstream stream(shaderIdentifier);

    std::string fileName;
    std::getline(stream, fileName, '|');

    std::string macro;
    while (std::getline(stream, macro, '|'))
    {
        const size_t separator = macro.find('=');
        if (separator == std::string::npos)
            macros.emplace_back(macro, "1");
        else
            macros.emplace_back(macro.substr(0, separator), macro.substr(separator + 1));
    }

    return fileName;
}

NrdIntegration::NrdIntegration(nvrhi::IDevice* device, nrd::Denoiser denoiser)
    : m_device(device), m_initialized(false), m_instance(nullptr), m_denoiser(denoiser), m_bindingCache(device), m_identifier(0)
{
}

NrdIntegration::~NrdIntegration()
{
    if (m_instance)
        nrd::DestroyInstance(*m_instance);
}

bool NrdIntegration::Initialize(uint32_t width, uint32_t height, donut::engine::ShaderFactory& shaderFactory)
{
    const nrd::DenoiserDesc denoiserDescs[] = { { m_identifier, m_denoiser } };

    nrd::InstanceCreationDesc instanceCreationDesc = {};
    instanceCreationDesc.denoisers = denoiserDescs;
    instanceCreationDesc.denoisersNum = _countof(denoiserDescs);

    nrd::Result result = nrd::CreateInstance(instanceCreationDesc, m_instance);
    if (result != nrd::Result::SUCCESS)
        return false;

    const nrd::InstanceDesc& instanceDesc = *nrd::GetInstanceDesc(*m_instance);

    const nvrhi::BufferDesc constantBufferDesc =
        nvrhi::utils::CreateVolatileConstantBufferDesc(instanceDesc.constantBufferMaxDataSize, "NrdConstantBuffer", instanceDesc.descriptorPoolDesc.setsMaxNum * 4);

    m_constantBuffer = m_device->createBuffer(constantBufferDesc);

    for (uint32_t samplerIndex = 0; samplerIndex < instanceDesc.samplersNum; samplerIndex++)
    {
        const nrd::Sampler& samplerMode = instanceDesc.samplers[samplerIndex];

        nvrhi::SamplerAddressMode addressMode = nvrhi::SamplerAddressMode::Wrap;
        bool filter = false;

        switch (samplerMode)
        {
        case nrd::Sampler::NEAREST_CLAMP:
            addressMode = nvrhi::SamplerAddressMode::Clamp;
            filter = false;
            break;
        case nrd::Sampler::LINEAR_CLAMP:
            addressMode = nvrhi::SamplerAddressMode::Clamp;
            filter = true;
            break;
        default:
            assert(!"Unknown NRD sampler mode");
            break;
        }

        auto samplerDesc = nvrhi::SamplerDesc().setAllAddressModes(addressMode).setAllFilters(filter);

        const nvrhi::SamplerHandle sampler = m_device->createSampler(samplerDesc);

        if (!sampler)
        {
            assert(!"Cannot create an NRD sampler");
            return false;
        }

        m_samplers.push_back(sampler);
    }

    // NRD keeps the constant buffer and samplers in a register space of their own, shared by every pipeline.
    // Mapping register spaces onto descriptor sets matches how DXC lays out the SPIR-V.
    nvrhi::BindingLayoutDesc sharedLayoutDesc;
    sharedLayoutDesc.visibility = nvrhi::ShaderType::Compute;
    sharedLayoutDesc.registerSpace = instanceDesc.constantBufferAndSamplersSpaceIndex;
    sharedLayoutDesc.registerSpaceIsDescriptorSet = true;
    sharedLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::VolatileConstantBuffer(instanceDesc.constantBufferRegisterIndex));

    nvrhi::BindingSetDesc sharedSetDesc;
    sharedSetDesc.bindings.push_back(nvrhi::BindingSetItem::ConstantBuffer(instanceDesc.constantBufferRegisterIndex, m_constantBuffer));

    for (uint32_t samplerIndex = 0; samplerIndex < instanceDesc.samplersNum; samplerIndex++)
    {
        const uint32_t slot = instanceDesc.samplersBaseRegisterIndex + samplerIndex;
        sharedLayoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Sampler(slot));
        sharedSetDesc.bindings.push_back(nvrhi::BindingSetItem::Sampler(slot, m_samplers[samplerIndex]));
    }

    m_sharedBindingLayout = m_device->createBindingLayout(sharedLayoutDesc);

    if (!m_sharedBindingLayout)
    {
        assert(!"Cannot create the NRD shared binding layout");
        return false;
    }

    m_sharedBindingSet = m_device->createBindingSet(sharedSetDesc, m_sharedBindingLayout);

    if (!m_sharedBindingSet)
    {
        assert(!"Cannot create the NRD shared binding set");
        return false;
    }

    for (uint32_t pipelineIndex = 0; pipelineIndex < instanceDesc.pipelinesNum; pipelineIndex++)
    {
        const nrd::PipelineDesc& nrdPipelineDesc = instanceDesc.pipelines[pipelineIndex];

        std::vector<donut::engine::ShaderMacro> macros;
        const std::string fileName = "nrd/" + ParseShaderIdentifier(nrdPipelineDesc.shaderIdentifier, macros);

        NrdPipeline pipeline;
        pipeline.shader = shaderFactory.CreateShader(fileName.c_str(), instanceDesc.shaderEntryPoint, &macros, nvrhi::ShaderType::Compute);

        if (!pipeline.shader)
        {
            assert(!"Cannot create an NRD shader");
            return false;
        }

        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.registerSpace = instanceDesc.resourcesSpaceIndex;
        layoutDesc.registerSpaceIsDescriptorSet = true;

        for (uint32_t rangeIndex = 0; rangeIndex < nrdPipelineDesc.resourceRangesNum; rangeIndex++)
        {
            const nrd::ResourceRangeDesc& range = nrdPipelineDesc.resourceRanges[rangeIndex];

            for (uint32_t descriptorOffset = 0; descriptorOffset < range.descriptorsNum; descriptorOffset++)
            {
                const uint32_t slot = instanceDesc.resourcesBaseRegisterIndex + descriptorOffset;

                switch (range.descriptorType)
                {
                case nrd::DescriptorType::TEXTURE:
                    layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_SRV(slot));
                    break;
                case nrd::DescriptorType::STORAGE_TEXTURE:
                    layoutDesc.bindings.push_back(nvrhi::BindingLayoutItem::Texture_UAV(slot));
                    break;
                default:
                    assert(!"Unknown NRD descriptor type");
                    return false;
                }
            }
        }

        pipeline.bindingLayout = m_device->createBindingLayout(layoutDesc);

        if (!pipeline.bindingLayout)
        {
            assert(!"Cannot create an NRD binding layout");
            return false;
        }

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { pipeline.bindingLayout, m_sharedBindingLayout };
        pipelineDesc.CS = pipeline.shader;
        pipeline.pipeline = m_device->createComputePipeline(pipelineDesc);

        if (!pipeline.pipeline)
        {
            assert(!"Cannot create an NRD pipeline");
            return false;
        }

        m_pipelines.push_back(pipeline);
    }

    const uint32_t poolSize = instanceDesc.permanentPoolSize + instanceDesc.transientPoolSize;

    for (uint32_t i = 0; i < poolSize; i++)
    {
        const bool isPermanent = (i < instanceDesc.permanentPoolSize);

        const nrd::TextureDesc& nrdTextureDesc = isPermanent ? instanceDesc.permanentPool[i] : instanceDesc.transientPool[i - instanceDesc.permanentPoolSize];

        const nvrhi::Format format = GetNvrhiFormat(nrdTextureDesc.format);
        if (format == nvrhi::Format::UNKNOWN)
        {
            assert(!"Unknown or unsupported NRD format");
            return false;
        }

        std::stringstream ss;
        ss << "NRD " << (isPermanent ? "Permanent" : "Transient") << "Texture [" << (isPermanent ? i : i - instanceDesc.permanentPoolSize) << "]";

        const uint32_t downsampleFactor = std::max<uint32_t>(nrdTextureDesc.downsampleFactor, 1);

        nvrhi::TextureDesc textureDesc;
        textureDesc.width = (width + downsampleFactor - 1) / downsampleFactor;
        textureDesc.height = (height + downsampleFactor - 1) / downsampleFactor;
        textureDesc.format = format;
        textureDesc.dimension = nvrhi::TextureDimension::Texture2D;
        textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        textureDesc.keepInitialState = true;
        textureDesc.isUAV = true;
        textureDesc.debugName = ss.str();

        const nvrhi::TextureHandle texture = m_device->createTexture(textureDesc);

        if (!texture)
        {
            assert(!"Cannot create an NRD texture");
            return false;
        }

        if (isPermanent)
            m_permanentTextures.push_back(texture);
        else
            m_transientTextures.push_back(texture);
    }

    m_initialized = true;

    return true;
}

const nrd::Denoiser NrdIntegration::GetDenoiser() const
{
    return m_denoiser;
}

bool NrdIntegration::IsAvailable() const
{
    return m_initialized;
}

static inline void MatrixToNrd(float* dest, const dm::float4x4& m)
{
    memcpy(dest, &m, sizeof(m));
}

void NrdIntegration::RunDenoiserPasses(
    nvrhi::ICommandList* commandList,
    const RenderTargets& renderTargets,
    int pass,
    const donut::engine::PlanarView& view,
    const donut::engine::PlanarView& viewPrev,
    uint32_t frameIndex,
    float disocclusionThreshold,
    float disocclusionThresholdAlternate,
    bool useDisocclusionThresholdAlternateMix,
    bool enableValidation,
    const void* methodSettings,
    bool reset)
{
    if (methodSettings)
        nrd::SetDenoiserSettings(*m_instance, m_identifier, methodSettings);

    nrd::CommonSettings commonSettings;
    MatrixToNrd(commonSettings.worldToViewMatrix, dm::affineToHomogeneous(view.GetViewMatrix()));
    MatrixToNrd(commonSettings.worldToViewMatrixPrev, dm::affineToHomogeneous(viewPrev.GetViewMatrix()));
    MatrixToNrd(commonSettings.viewToClipMatrix, view.GetProjectionMatrix(false));
    MatrixToNrd(commonSettings.viewToClipMatrixPrev, viewPrev.GetProjectionMatrix(false));

    dm::float2 pixelOffset = view.GetPixelOffset();
    dm::float2 prevPixelOffset = viewPrev.GetPixelOffset();
    commonSettings.isMotionVectorInWorldSpace = false;
    commonSettings.motionVectorScale[0] = 1.0f / view.GetViewExtent().width();
    commonSettings.motionVectorScale[1] = 1.0f / view.GetViewExtent().height();
    // Not passing Donut G-buffer .z directly with a scale of 1.0.
    // Donut stores device-depth delta there, while NRD interprets nonzero .z as view-Z delta.
    commonSettings.motionVectorScale[2] = 0.0f;
    commonSettings.cameraJitter[0] = pixelOffset.x;
    commonSettings.cameraJitter[1] = pixelOffset.y;
    commonSettings.cameraJitterPrev[0] = prevPixelOffset.x;
    commonSettings.cameraJitterPrev[1] = prevPixelOffset.y;
    commonSettings.resourceSize[0] = view.GetViewExtent().width();
    commonSettings.resourceSize[1] = view.GetViewExtent().height();
    commonSettings.resourceSizePrev[0] = viewPrev.GetViewExtent().width();
    commonSettings.resourceSizePrev[1] = viewPrev.GetViewExtent().height();
    commonSettings.rectSize[0] = view.GetViewExtent().width();
    commonSettings.rectSize[1] = view.GetViewExtent().height();
    commonSettings.rectSizePrev[0] = viewPrev.GetViewExtent().width();
    commonSettings.rectSizePrev[1] = viewPrev.GetViewExtent().height();
    commonSettings.frameIndex = frameIndex;
    commonSettings.enableValidation = enableValidation;
    commonSettings.disocclusionThreshold = disocclusionThreshold;
    commonSettings.disocclusionThresholdAlternate = disocclusionThresholdAlternate;
    commonSettings.isDisocclusionThresholdMixAvailable = useDisocclusionThresholdAlternateMix;
    commonSettings.accumulationMode = reset ? nrd::AccumulationMode::RESTART : nrd::AccumulationMode::CONTINUE;

    nrd::SetCommonSettings(*m_instance, commonSettings);

    const nrd::DispatchDesc* dispatchDescs = nullptr;
    uint32_t dispatchDescNum = 0;
    nrd::GetComputeDispatches(*m_instance, &m_identifier, 1, dispatchDescs, dispatchDescNum);

    const nrd::InstanceDesc& instanceDesc = *nrd::GetInstanceDesc(*m_instance);

    for (uint32_t dispatchIndex = 0; dispatchIndex < dispatchDescNum; dispatchIndex++)
    {
        const nrd::DispatchDesc& dispatchDesc = dispatchDescs[dispatchIndex];

        if (dispatchDesc.name)
            commandList->beginMarker(dispatchDesc.name);

        if (dispatchDesc.constantBufferDataSize > 0)
        {
            assert(m_constantBuffer);
            commandList->writeBuffer(m_constantBuffer, dispatchDesc.constantBufferData, dispatchDesc.constantBufferDataSize);
        }

        const nrd::PipelineDesc& nrdPipelineDesc = instanceDesc.pipelines[dispatchDesc.pipelineIndex];

        nvrhi::BindingSetDesc setDesc;
        uint32_t resourceIndex = 0;

        for (uint32_t rangeIndex = 0; rangeIndex < nrdPipelineDesc.resourceRangesNum; rangeIndex++)
        {
            const nrd::ResourceRangeDesc& range = nrdPipelineDesc.resourceRanges[rangeIndex];

            for (uint32_t descriptorOffset = 0; descriptorOffset < range.descriptorsNum; descriptorOffset++)
            {
                assert(resourceIndex < dispatchDesc.resourcesNum);
                const nrd::ResourceDesc& resource = dispatchDesc.resources[resourceIndex];

                nvrhi::TextureHandle texture;
                switch (resource.type)
                {
                case nrd::ResourceType::IN_MV:
                    texture = renderTargets.motionVectors;
                    break;
                case nrd::ResourceType::IN_NORMAL_ROUGHNESS:
                    texture = renderTargets.normalRoughness;
                    break;
                case nrd::ResourceType::IN_VIEWZ:
                    texture = renderTargets.depth;
                    break;
                case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST:
                    texture = renderTargets.inSpecRadianceHitDist;
                    break;
                case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST:
                    texture = renderTargets.inDiffRadianceHitDist;
                    break;
                case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST:
                    texture = renderTargets.outSpecRadianceHitDist;
                    break;
                case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST:
                    texture = renderTargets.outDiffRadianceHitDist;
                    break;
                case nrd::ResourceType::TRANSIENT_POOL:
                    texture = m_transientTextures[resource.indexInPool];
                    break;
                case nrd::ResourceType::PERMANENT_POOL:
                    texture = m_permanentTextures[resource.indexInPool];
                    break;
                default:
                    assert(!"Unavailable resource type");
                    break;
                }

                assert(texture);

                const uint32_t slot = instanceDesc.resourcesBaseRegisterIndex + descriptorOffset;

                if (range.descriptorType == nrd::DescriptorType::TEXTURE)
                    setDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_SRV(slot, texture));
                else
                    setDesc.bindings.push_back(nvrhi::BindingSetItem::Texture_UAV(slot, texture));

                resourceIndex++;
            }
        }

        assert(resourceIndex == dispatchDesc.resourcesNum);

        const NrdPipeline& pipeline = m_pipelines[dispatchDesc.pipelineIndex];

        nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(setDesc, pipeline.bindingLayout);

        nvrhi::ComputeState state;
        state.bindings = { bindingSet, m_sharedBindingSet };
        state.pipeline = pipeline.pipeline;
        commandList->setComputeState(state);

        commandList->dispatch(dispatchDesc.gridWidth, dispatchDesc.gridHeight);

        if (dispatchDesc.name)
            commandList->endMarker();
    }
}
