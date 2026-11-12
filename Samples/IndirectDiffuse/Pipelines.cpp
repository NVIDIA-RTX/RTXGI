/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "Application/Defines.h"
#include "Pipelines.h"

#include <algorithm>

using namespace donut::engine;

void Pipelines::CreateCommon(nvrhi::IDevice* device, ShaderFactory& shaderFactory, nvrhi::GraphicsAPI api, const BindingLayouts& layouts)
{
#if ENABLE_SHARC
    {
        nvrhi::ComputePipelineDesc pipelineDesc;
        if (api == nvrhi::GraphicsAPI::D3D12)
            pipelineDesc.bindingLayouts = { layouts.global, layouts.sharc };
        else
            pipelineDesc.bindingLayouts = { layouts.global, layouts.dummy[1], layouts.dummy[2], layouts.sharc };

        m_sharcResolveCS = shaderFactory.CreateShader("app/Shaders/SharcResolve.hlsl", "sharcResolve", nullptr, nvrhi::ShaderType::Compute);
        pipelineDesc.CS = m_sharcResolveCS;
        m_sharcResolvePSO = device->createComputePipeline(pipelineDesc);
    }
#endif // ENABLE_SHARC

#if ENABLE_NRD
    std::vector<ShaderMacro> denoiseMacros = { ShaderMacro("NRD_NORMAL_ENCODING", "2"), ShaderMacro("NRD_ROUGHNESS_ENCODING", "1") };

#if ENABLE_NRC || ENABLE_SHARC
    {
        m_nrdIndirectDiffusePackCS = shaderFactory.CreateShader("app/Shaders/NrdIndirectDiffuse.hlsl", "PackNrdIndirectDiffuse", &denoiseMacros, nvrhi::ShaderType::Compute);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = {
            layouts.global,  layouts.denoiser, layouts.dummy[DescriptorSetIDs::Nrc], layouts.dummy[DescriptorSetIDs::Sharc], layouts.dummy[DescriptorSetIDs::Bindless],
            layouts.gBuffer,
        };
        pipelineDesc.CS = m_nrdIndirectDiffusePackCS;

        m_nrdIndirectDiffusePackPSO = device->createComputePipeline(pipelineDesc);
    }

    {
        m_nrdIndirectDiffuseResolveCS = shaderFactory.CreateShader("app/Shaders/NrdIndirectDiffuse.hlsl", "ResolveNrdIndirectDiffuse", &denoiseMacros, nvrhi::ShaderType::Compute);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc.bindingLayouts = {
            layouts.global,  layouts.denoiser, layouts.dummy[DescriptorSetIDs::Nrc], layouts.dummy[DescriptorSetIDs::Sharc], layouts.dummy[DescriptorSetIDs::Bindless],
            layouts.gBuffer,
        };
        pipelineDesc.CS = m_nrdIndirectDiffuseResolveCS;

        m_nrdIndirectDiffuseResolvePSO = device->createComputePipeline(pipelineDesc);
    }
#endif // ENABLE_NRC || ENABLE_SHARC
#endif // ENABLE_NRD

    m_tonemappingPS = shaderFactory.CreateShader("app/Shaders/Tonemapping.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);
}

bool Pipelines::CreateRayTracing(nvrhi::IDevice* device, ShaderFactory& shaderFactory, const BindingLayouts& layouts, bool enableNrd, bool enableSer)
{
    char const* enableDenoiserStr = enableNrd ? "1" : "0";

    std::vector<ShaderMacro> macros[Type::Count];
#if ENABLE_NRC
    // The standard NRC_UPDATE and NRC_QUERY macros must remain because Nrc.hlsli uses them to select its mode.
    // NRC_INDIRECT_DIFFUSE only selects the application's raygen implementation.
    macros[Type::NrcIndirectDiffuseUpdate].push_back(ShaderMacro("NRC_UPDATE", "1"));
    macros[Type::NrcIndirectDiffuseUpdate].push_back(ShaderMacro("NRC_INDIRECT_DIFFUSE", "1"));

    macros[Type::NrcIndirectDiffuseQuery].push_back(ShaderMacro("NRC_QUERY", "1"));
    macros[Type::NrcIndirectDiffuseQuery].push_back(ShaderMacro("NRC_INDIRECT_DIFFUSE", "1"));
    macros[Type::NrcIndirectDiffuseQuery].push_back(ShaderMacro("ENABLE_NRD", enableDenoiserStr));
#endif // ENABLE_NRC

#if ENABLE_SHARC
    macros[Type::SharcUpdate].push_back(ShaderMacro("SHARC_UPDATE", "1"));
    macros[Type::SharcUpdate].push_back(ShaderMacro("SHARC_INDIRECT_DIFFUSE", "1"));

    macros[Type::SharcQuery].push_back(ShaderMacro("SHARC_QUERY", "1"));
    macros[Type::SharcQuery].push_back(ShaderMacro("SHARC_INDIRECT_DIFFUSE", "1"));
    macros[Type::SharcQuery].push_back(ShaderMacro("ENABLE_NRD", enableDenoiserStr));
#endif // ENABLE_SHARC

    // Selects the SER permutations (see Shaders.cfg) that trace through a hit object with thread reordering
    if (enableSer)
    {
        for (uint32_t i = 0; i < Type::Count; ++i)
            macros[i].push_back(ShaderMacro("ENABLE_SER", "1"));
    }

    for (uint32_t i = 0; i < Type::Count; ++i)
    {
        if (!CreatePermutation(device, shaderFactory, m_permutations[i], macros[i], layouts))
            return false;
    }

    return true;
}

Pipelines::Permutation& Pipelines::GetPermutation(Type t)
{
    return m_permutations[t];
}

#if ENABLE_SHARC
nvrhi::ComputePipelineHandle Pipelines::GetSharcResolvePSO() const
{
    return m_sharcResolvePSO;
}
#endif // ENABLE_SHARC

#if ENABLE_NRD
#if ENABLE_NRC || ENABLE_SHARC
nvrhi::ComputePipelineHandle Pipelines::GetNrdIndirectDiffusePackPSO() const
{
    return m_nrdIndirectDiffusePackPSO;
}

nvrhi::ComputePipelineHandle Pipelines::GetNrdIndirectDiffuseResolvePSO() const
{
    return m_nrdIndirectDiffuseResolvePSO;
}
#endif // ENABLE_NRC || ENABLE_SHARC
#endif // ENABLE_NRD

nvrhi::ShaderHandle Pipelines::GetTonemappingPS() const
{
    return m_tonemappingPS;
}

bool Pipelines::CreatePermutation(
    nvrhi::IDevice* device, ShaderFactory& shaderFactory, Pipelines::Permutation& permutation, std::vector<ShaderMacro>& macros, const Pipelines::BindingLayouts& layouts)
{
    nvrhi::ShaderLibraryHandle shaderLibrary = shaderFactory.CreateShaderLibrary("app/Shaders/Pathtracer.hlsl", &macros);
    if (!shaderLibrary)
        return false;

    permutation.shaderLibrary = shaderLibrary;

    auto isMacroEnabled = [&macros](const char* name) {
        const auto it = std::find_if(macros.begin(), macros.end(), [name](const ShaderMacro& macro) { return macro.name == name; });

        return it != macros.end() && it->definition == "1";
    };

    const bool isNrcUpdate = isMacroEnabled("NRC_UPDATE");
    const bool isNrcQuery = isMacroEnabled("NRC_QUERY");
    const bool isNrcIndirectDiffuse = isMacroEnabled("NRC_INDIRECT_DIFFUSE");
    const bool isSharcIndirectDiffuse = isMacroEnabled("SHARC_INDIRECT_DIFFUSE");
    const bool isSharcUpdate = isMacroEnabled("SHARC_UPDATE");
    const bool isSharcQuery = isMacroEnabled("SHARC_QUERY");
    const char* rayGenerationShaderName = isNrcIndirectDiffuse ? "NrcIndirectDiffuseRayGen" : "SharcIndirectDiffuseRayGen";

    nvrhi::rt::PipelineDesc pipelineDesc;
    for (int i = 0; i < DescriptorSetIDs::Count; ++i)
        pipelineDesc.globalBindingLayouts.push_back(layouts.dummy[i]);

    pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Globals] = layouts.global;
    pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Bindless] = layouts.bindless;
    pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Denoiser] = layouts.denoiser;

    pipelineDesc.shaders = {
        { "", shaderLibrary->getShader(rayGenerationShaderName, nvrhi::ShaderType::RayGeneration), nullptr },
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
    if (isNrcUpdate || isNrcQuery)
        pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Nrc] = layouts.nrc;

    if (isNrcIndirectDiffuse)
        pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Gbuffer] = layouts.gBuffer;
#endif // ENABLE_NRC

#if ENABLE_NRC || ENABLE_SHARC
    if (isNrcIndirectDiffuse || isSharcIndirectDiffuse)
        pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Gbuffer] = layouts.gBuffer;
#endif // ENABLE_NRC || ENABLE_SHARC

#if ENABLE_SHARC
    if (isSharcUpdate || isSharcQuery)
        pipelineDesc.globalBindingLayouts[DescriptorSetIDs::Sharc] = layouts.sharc;
#endif // ENABLE_SHARC

    permutation.pipeline = device->createRayTracingPipeline(pipelineDesc);
    permutation.shaderTable = permutation.pipeline->createShaderTable();

    permutation.shaderTable->setRayGenerationShader(rayGenerationShaderName);
    permutation.shaderTable->addHitGroup("HitGroup");
    permutation.shaderTable->addHitGroup("HitGroupShadow");
    permutation.shaderTable->addMissShader("Miss");
    permutation.shaderTable->addMissShader("ShadowMiss");

    return true;
}