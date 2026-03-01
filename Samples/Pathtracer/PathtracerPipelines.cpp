/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "PathtracerUi.h" // TODO: we do this to get the NRC and SHARC defines. Move to Defines.h
#include "PathtracerPipelines.h"

#include <algorithm>

using namespace donut::engine;

void PathtracerPipelines::CreateCommon(nvrhi::IDevice* device, ShaderFactory& shaderFactory, nvrhi::GraphicsAPI api, const BindingLayouts& layouts)
{
#if ENABLE_SHARC
    {
        nvrhi::ComputePipelineDesc pipelineDesc;
        if (api == nvrhi::GraphicsAPI::D3D12)
            pipelineDesc.bindingLayouts = { layouts.global, layouts.sharc };
        else
            pipelineDesc.bindingLayouts = { layouts.global, layouts.dummy[1], layouts.dummy[2], layouts.sharc };

        m_sharcResolveCS = shaderFactory.CreateShader("app/SharcResolve.hlsl", "sharcResolve", nullptr, nvrhi::ShaderType::Compute);
        pipelineDesc.CS = m_sharcResolveCS;
        m_sharcResolvePSO = device->createComputePipeline(pipelineDesc);
    }
#endif // ENABLE_SHARC

#if ENABLE_NRD
    std::vector<ShaderMacro> denoiseMacros = { ShaderMacro("NRD_NORMAL_ENCODING", "2"), ShaderMacro("NRD_ROUGHNESS_ENCODING", "1") };

    {
        m_denoiserReblurPackCS = shaderFactory.CreateShader("app/Denoiser.hlsl", "reblurPackData", &denoiseMacros, nvrhi::ShaderType::Compute);
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { layouts.global, layouts.denoiser };
        pipelineDesc.CS = m_denoiserReblurPackCS;
        m_denoiserReblurPackPSO = device->createComputePipeline(pipelineDesc);
    }

#if ENABLE_NRC
    {
        std::vector<ShaderMacro> denoiseMacrosNRC = denoiseMacros;
        denoiseMacrosNRC.push_back(ShaderMacro("ENABLE_NRC", "1"));

        m_denoiserReblurPack_NRC_CS = shaderFactory.CreateShader("app/Denoiser.hlsl", "reblurPackData", &denoiseMacrosNRC, nvrhi::ShaderType::Compute);
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { layouts.global, layouts.denoiser };
        pipelineDesc.CS = m_denoiserReblurPack_NRC_CS;
        m_denoiserReblurPack_NRC_PSO = device->createComputePipeline(pipelineDesc);
    }
#endif // ENABLE_NRC

    {
        m_denoiserResolveCS = shaderFactory.CreateShader("app/Denoiser.hlsl", "resolve", &denoiseMacros, nvrhi::ShaderType::Compute);
        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = { layouts.global, layouts.denoiser };
        pipelineDesc.CS = m_denoiserResolveCS;
        m_denoiserResolvePSO = device->createComputePipeline(pipelineDesc);
    }
#endif // ENABLE_NRD

    m_tonemappingPS = shaderFactory.CreateShader("app/Tonemapping.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
}

bool PathtracerPipelines::CreateRayTracing(nvrhi::IDevice* device, ShaderFactory& shaderFactory, const BindingLayouts& layouts, bool enableNrd)
{
    char const* enableDenoiserStr = enableNrd ? "1" : "0";

    std::vector<ShaderMacro> macros[Type::Count];
    macros[Type::DefaultPathTracing].push_back(ShaderMacro("REFERENCE", "1"));
    macros[Type::DefaultPathTracing].push_back(ShaderMacro("ENABLE_NRD", enableDenoiserStr));

#if ENABLE_NRC
    macros[Type::NRC_Update].push_back(ShaderMacro("NRC_UPDATE", "1"));
    macros[Type::NRC_Query].push_back(ShaderMacro("NRC_QUERY", "1"));
    macros[Type::NRC_Query].push_back(ShaderMacro("ENABLE_NRD", enableDenoiserStr));
#endif // ENABLE_NRC

#if ENABLE_SHARC
    macros[Type::Sharc_Update].push_back(ShaderMacro("SHARC_UPDATE", "1"));
    macros[Type::Sharc_Query].push_back(ShaderMacro("SHARC_QUERY", "1"));
    macros[Type::Sharc_Query].push_back(ShaderMacro("ENABLE_NRD", enableDenoiserStr));
#endif // ENABLE_SHARC

    for (uint32_t i = 0; i < Type::Count; ++i)
    {
        if (!CreatePermutation(device, shaderFactory, m_permutations[i], macros[i], layouts))
            return false;
    }

    return true;
}

PathtracerPipelines::Permutation& PathtracerPipelines::GetPermutation(Type t)
{
    return m_permutations[t];
}

#if ENABLE_SHARC
nvrhi::ComputePipelineHandle PathtracerPipelines::GetSharcResolvePSO() const
{
    return m_sharcResolvePSO;
}
#endif // ENABLE_SHARC

#if ENABLE_NRD
nvrhi::ComputePipelineHandle PathtracerPipelines::GetDenoiserReblurPackPSO() const
{
    return m_denoiserReblurPackPSO;
}

#if ENABLE_NRC
nvrhi::ComputePipelineHandle PathtracerPipelines::GetDenoiserReblurPack_NRC_PSO() const
{
    return m_denoiserReblurPack_NRC_PSO;
}
#endif // ENABLE_NRC

nvrhi::ComputePipelineHandle PathtracerPipelines::GetDenoiserResolvePSO() const
{
    return m_denoiserResolvePSO;
}
#endif // ENABLE_NRD

nvrhi::ShaderHandle PathtracerPipelines::GetTonemappingPS() const
{
    return m_tonemappingPS;
}

bool PathtracerPipelines::CreatePermutation(
    nvrhi::IDevice* device,
    ShaderFactory& shaderFactory,
    PathtracerPipelines::Permutation& permutation,
    std::vector<ShaderMacro>& macros,
    const PathtracerPipelines::BindingLayouts& layouts)
{
    nvrhi::ShaderLibraryHandle shaderLibrary = shaderFactory.CreateShaderLibrary("app/Pathtracer.hlsl", &macros);
    if (!shaderLibrary)
        return false;

    permutation.shaderLibrary = shaderLibrary;

    auto macroDefined = [&macros](const std::string& token) {
        auto it = std::find_if(macros.begin(), macros.end(), [&token](const ShaderMacro& macro) { return macro.name.find(token) != std::string::npos; });
        return (it != macros.end()) && (it->definition == "1");
    };

    nvrhi::rt::PipelineDesc pipelineDesc;
    for (int i = 0; i < DescriptorSetIDs::COUNT; ++i)
        pipelineDesc.globalBindingLayouts.push_back(layouts.dummy[i]);

    pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Globals] = layouts.global;
    pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Bindless] = layouts.bindless;
    pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Denoiser] = layouts.denoiser;

    pipelineDesc.shaders = {
        { "", shaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration), nullptr },
        { "", shaderLibrary->getShader("Miss", nvrhi::ShaderType::Miss), nullptr },
        { "", shaderLibrary->getShader("ShadowMiss", nvrhi::ShaderType::Miss), nullptr },
    };

    pipelineDesc.hitGroups = {
        {
            "HitGroup",
            shaderLibrary->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit),
            shaderLibrary->getShader("AnyHit", nvrhi::ShaderType::AnyHit),
            nullptr, // intersectionShader
            nullptr, // bindingLayout
            false    // isProceduralPrimitive
        },
        {
            "HitGroupShadow",
            shaderLibrary->getShader("ClosestHitShadow", nvrhi::ShaderType::ClosestHit),
            shaderLibrary->getShader("AnyHitShadow", nvrhi::ShaderType::AnyHit),
            nullptr, // intersectionShader
            nullptr, // bindingLayout
            false    // isProceduralPrimitive
        },
    };

    pipelineDesc.maxPayloadSize = sizeof(float) * 6;

#if ENABLE_NRC
    if (macroDefined("NRC_"))
        pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Nrc] = layouts.nrc;
#endif // ENABLE_NRC

#if ENABLE_SHARC
    if (macroDefined("SHARC_"))
        pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Sharc] = layouts.sharc;
#endif // ENABLE_SHARC

    permutation.pipeline = device->createRayTracingPipeline(pipelineDesc);
    permutation.shaderTable = permutation.pipeline->createShaderTable();

    permutation.shaderTable->setRayGenerationShader("RayGen");
    permutation.shaderTable->addHitGroup("HitGroup");
    permutation.shaderTable->addHitGroup("HitGroupShadow");
    permutation.shaderTable->addMissShader("Miss");
    permutation.shaderTable->addMissShader("ShadowMiss");

    return true;
}