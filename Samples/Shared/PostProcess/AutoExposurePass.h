/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <donut/core/math/math.h>
#include <donut/engine/BindingCache.h>
#include <nvrhi/nvrhi.h>

namespace donut::engine
{
class ShaderFactory;
}

// Meters the luminance of an HDR colour texture and keeps the adapted luminance, the scene luminance the tonemapper maps
// to middle grey, in a one-element buffer. The buffer is an R32_UINT holding a float bit pattern, the layout Donut's
// DLSS integration consumes as its exposure buffer, so one value drives both the tonemapper and DLSS.
// The shaders are loaded through the sample's "app/Shaders" mount, so the sample's Shaders.cfg must compile
// AutoExposure.hlsl.
class AutoExposurePass
{
public:
    struct Settings
    {
        float minLogLuminance = -12.0f;   // Metering range in log2 scene luminance (EV)
        float maxLogLuminance = 12.0f;
        float lowPercentile = 0.4f;       // Fraction of the darkest pixels ignored (shadows, black pixels)
        float highPercentile = 0.95f;     // Fraction above which pixels are ignored (emitters, fireflies)
        float adaptationSpeedUp = 3.0f;   // Toward a brighter scene: the remaining EV gap halves every 1/speed seconds
        float adaptationSpeedDown = 1.0f; // Toward a darker scene
    };

    AutoExposurePass(nvrhi::IDevice* device);

    // Creates the compute pipelines; call again after the shader factory cache was cleared to reload shaders.
    void CreatePipelines(donut::engine::ShaderFactory& shaderFactory);

    // The histogram binding set references the source texture, so drop it when the render targets are recreated.
    void ResetBindingCache();

    // Meters sourceTexture and moves the adapted luminance toward the result; resetAdaptation snaps to it instead.
    void Render(nvrhi::ICommandList* commandList, nvrhi::ITexture* sourceTexture, dm::uint2 sourceSize, const Settings& settings, float frameTime, bool resetAdaptation);

    // Replaces the adapted luminance with a fixed value, used while the exposure is set manually.
    void SetAdaptedLuminance(nvrhi::ICommandList* commandList, float luminance);

    nvrhi::IBuffer* GetExposureBuffer() const;

    // Converts the tonemapper's exposure compensation into DLSS::EvaluateParameters::exposureScale so that DLSS applies
    // the same exposure as the tonemapper.
    static float GetDlssExposureScale(float exposureScale);

private:
    nvrhi::DeviceHandle m_device;
    nvrhi::BufferHandle m_constantBuffer;
    nvrhi::BufferHandle m_histogramBuffer;
    nvrhi::BufferHandle m_exposureBuffer;

    nvrhi::BindingLayoutHandle m_histogramBindingLayout;
    nvrhi::BindingLayoutHandle m_adaptationBindingLayout;
    nvrhi::BindingSetHandle m_adaptationBindingSet;
    nvrhi::ComputePipelineHandle m_histogramPipeline;
    nvrhi::ComputePipelineHandle m_adaptationPipeline;

    donut::engine::BindingCache m_bindingCache;
};
