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

#include <donut/engine/ShaderFactory.h>
#include <nvrhi/nvrhi.h>

#include <vector>

// Descriptor set slot assignments shared between pipeline creation and rendering.
struct DescriptorSetIDs
{
    enum
    {
        Globals = 0,
        Denoiser = 1,
        Nrc = 2,
        Sharc = 3,
        Bindless = 4,
        Gbuffer = 5,
        Count = 6
    };
};

class Pipelines
{
public:
    struct Permutation
    {
        nvrhi::ShaderLibraryHandle shaderLibrary;
        nvrhi::rt::PipelineHandle pipeline;
        nvrhi::rt::ShaderTableHandle shaderTable;
    };

    enum Type
    {
#if ENABLE_NRC
        NrcIndirectDiffuseUpdate,
        NrcIndirectDiffuseQuery,
#endif // ENABLE_NRC
#if ENABLE_SHARC
        SharcUpdate,
        SharcQuery,
#endif // ENABLE_SHARC
        Count
    };

    // Bundles all binding layouts required for pipeline construction.
    struct BindingLayouts
    {
        nvrhi::BindingLayoutHandle global;
        nvrhi::BindingLayoutHandle denoiser;
        nvrhi::BindingLayoutHandle bindless;
#if ENABLE_NRC
        nvrhi::BindingLayoutHandle nrc;
#endif
#if ENABLE_NRC || ENABLE_SHARC
        nvrhi::BindingLayoutHandle gBuffer;
#endif
#if ENABLE_SHARC
        nvrhi::BindingLayoutHandle sharc;
#endif
        nvrhi::BindingLayoutHandle dummy[DescriptorSetIDs::Count];
    };

    void CreateCommon(nvrhi::IDevice* device, donut::engine::ShaderFactory& shaderFactory, nvrhi::GraphicsAPI api, const BindingLayouts& layouts);

    bool CreateRayTracing(nvrhi::IDevice* device, donut::engine::ShaderFactory& shaderFactory, const BindingLayouts& layouts, bool enableNrd, bool enableSer);

    Permutation& GetPermutation(Type t);

#if ENABLE_SHARC
    nvrhi::ComputePipelineHandle GetSharcResolvePSO() const;
#endif // ENABLE_SHARC

#if ENABLE_NRD
#if ENABLE_NRC || ENABLE_SHARC
    nvrhi::ComputePipelineHandle GetNrdIndirectDiffusePackPSO() const;
    nvrhi::ComputePipelineHandle GetNrdIndirectDiffuseResolvePSO() const;
#endif // ENABLE_NRC || ENABLE_SHARC
#endif // ENABLE_NRD

    nvrhi::ShaderHandle GetTonemappingPS() const;

private:
    bool CreatePermutation(
        nvrhi::IDevice* device, donut::engine::ShaderFactory& shaderFactory, Permutation& permutation, std::vector<donut::engine::ShaderMacro>& macros, const BindingLayouts& layouts);

    Permutation m_permutations[Type::Count];

#if ENABLE_SHARC
    nvrhi::ShaderHandle m_sharcResolveCS;
    nvrhi::ComputePipelineHandle m_sharcResolvePSO;
#endif // ENABLE_SHARC

#if ENABLE_NRD
#if ENABLE_NRC || ENABLE_SHARC
    nvrhi::ShaderHandle m_nrdIndirectDiffusePackCS;
    nvrhi::ComputePipelineHandle m_nrdIndirectDiffusePackPSO;

    nvrhi::ShaderHandle m_nrdIndirectDiffuseResolveCS;
    nvrhi::ComputePipelineHandle m_nrdIndirectDiffuseResolvePSO;
#endif // ENABLE_NRC || ENABLE_SHARC
#endif // ENABLE_NRD

    nvrhi::ShaderHandle m_tonemappingPS;
};