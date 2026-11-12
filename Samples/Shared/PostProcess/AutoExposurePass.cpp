/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "AutoExposurePass.h"

#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <nvrhi/utils.h>

#include <algorithm>
#include <bit>

using namespace donut::math;

#include "Shaders/AutoExposureCb.h"

AutoExposurePass::AutoExposurePass(nvrhi::IDevice* device) : m_device(device), m_bindingCache(device)
{
    m_constantBuffer = device->createBuffer(
        nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(AutoExposureConstants), "AutoExposureConstants", donut::engine::c_MaxRenderPassConstantBufferVersions));

    nvrhi::BufferDesc bufferDesc;
    bufferDesc.byteSize = AUTO_EXPOSURE_HISTOGRAM_BINS * sizeof(uint32_t);
    bufferDesc.format = nvrhi::Format::R32_UINT;
    bufferDesc.canHaveUAVs = true;
    bufferDesc.canHaveTypedViews = true;
    bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    bufferDesc.keepInitialState = true;
    bufferDesc.debugName = "AutoExposureHistogram";
    m_histogramBuffer = device->createBuffer(bufferDesc);

    bufferDesc.byteSize = sizeof(uint32_t);
    bufferDesc.debugName = "AutoExposureAdaptedLuminance";
    m_exposureBuffer = device->createBuffer(bufferDesc);

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(0),
        };
        m_histogramBindingLayout = device->createBindingLayout(layoutDesc);
    }

    {
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(1),
            nvrhi::BindingLayoutItem::TypedBuffer_UAV(1),
        };
        m_adaptationBindingLayout = device->createBindingLayout(layoutDesc);

        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
            nvrhi::BindingSetItem::TypedBuffer_SRV(1, m_histogramBuffer),
            nvrhi::BindingSetItem::TypedBuffer_UAV(1, m_exposureBuffer),
        };
        m_adaptationBindingSet = device->createBindingSet(bindingSetDesc, m_adaptationBindingLayout);
    }
}

void AutoExposurePass::CreatePipelines(donut::engine::ShaderFactory& shaderFactory)
{
    nvrhi::ComputePipelineDesc pipelineDesc;

    pipelineDesc.CS = shaderFactory.CreateShader("app/Shaders/AutoExposure.hlsl", "luminanceHistogram", nullptr, nvrhi::ShaderType::Compute);
    pipelineDesc.bindingLayouts = { m_histogramBindingLayout };
    m_histogramPipeline = m_device->createComputePipeline(pipelineDesc);

    pipelineDesc.CS = shaderFactory.CreateShader("app/Shaders/AutoExposure.hlsl", "exposureAdaptation", nullptr, nvrhi::ShaderType::Compute);
    pipelineDesc.bindingLayouts = { m_adaptationBindingLayout };
    m_adaptationPipeline = m_device->createComputePipeline(pipelineDesc);
}

void AutoExposurePass::ResetBindingCache()
{
    m_bindingCache.Clear();
}

void AutoExposurePass::Render(nvrhi::ICommandList* commandList, nvrhi::ITexture* sourceTexture, uint2 sourceSize, const Settings& settings, float frameTime, bool resetAdaptation)
{
    commandList->beginMarker("AutoExposure");

    // Keep the percentile window and the metering range well formed whatever the UI hands in.
    const float lowPercentile = std::clamp(settings.lowPercentile, 0.0f, 0.99f);
    const float highPercentile = std::clamp(settings.highPercentile, lowPercentile + 0.01f, 1.0f);
    const float logLuminanceRange = std::max(settings.maxLogLuminance - settings.minLogLuminance, 1.0f);

    AutoExposureConstants constants = {};
    constants.sourceSize = sourceSize;
    constants.minLogLuminance = settings.minLogLuminance;
    constants.logLuminanceRange = logLuminanceRange;
    constants.recipLogLuminanceRange = 1.0f / logLuminanceRange;
    constants.lowPercentile = lowPercentile;
    constants.highPercentile = highPercentile;
    constants.adaptationSpeedUp = std::max(settings.adaptationSpeedUp, 0.0f);
    constants.adaptationSpeedDown = std::max(settings.adaptationSpeedDown, 0.0f);
    constants.frameTime = std::max(frameTime, 0.0f);
    constants.resetAdaptation = resetAdaptation;
    commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));

    commandList->clearBufferUInt(m_histogramBuffer, 0);

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
        nvrhi::BindingSetItem::Texture_SRV(0, sourceTexture),
        nvrhi::BindingSetItem::TypedBuffer_UAV(0, m_histogramBuffer),
    };

    nvrhi::ComputeState state;
    state.pipeline = m_histogramPipeline;
    state.bindings = { m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_histogramBindingLayout) };
    commandList->setComputeState(state);

    const uint32_t groupsX = (sourceSize.x + AUTO_EXPOSURE_GROUP_SIZE - 1) / AUTO_EXPOSURE_GROUP_SIZE;
    const uint32_t groupsY = (sourceSize.y + AUTO_EXPOSURE_GROUP_SIZE - 1) / AUTO_EXPOSURE_GROUP_SIZE;
    commandList->dispatch(groupsX, groupsY);

    state.pipeline = m_adaptationPipeline;
    state.bindings = { m_adaptationBindingSet };
    commandList->setComputeState(state);
    commandList->dispatch(1);

    commandList->endMarker();
}

void AutoExposurePass::SetAdaptedLuminance(nvrhi::ICommandList* commandList, float luminance)
{
    commandList->clearBufferUInt(m_exposureBuffer, std::bit_cast<uint32_t>(luminance));
}

nvrhi::IBuffer* AutoExposurePass::GetExposureBuffer() const
{
    return m_exposureBuffer;
}

float AutoExposurePass::GetDlssExposureScale(float exposureScale)
{
    // Donut turns the buffer into the DLSS exposure as midGrey / (adaptedLuminance * (1 - midGrey)) * exposureScale
    // (dlss_exposure_cs.hlsl, after the DLSS Programming Guide). The tonemapper applies key / adaptedLuminance *
    // exposureScale with the same 0.18 middle grey, so folding (1 - midGrey) into the scale makes the two agree.
    return exposureScale * (1.0f - AUTO_EXPOSURE_KEY);
}
