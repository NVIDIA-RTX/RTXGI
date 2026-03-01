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
        Globals,
        Denoiser,
        Nrc,
        Sharc,
        Bindless,
        COUNT
    };
};

class PathtracerPipelines
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
        DefaultPathTracing,
#if ENABLE_NRC
        NRC_Update,
        NRC_Query,
#endif // ENABLE_NRC
#if ENABLE_SHARC
        Sharc_Update,
        Sharc_Query,
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
#endif // ENABLE_NRC
#if ENABLE_SHARC
        nvrhi::BindingLayoutHandle sharc;
#endif // ENABLE_SHARC
        nvrhi::BindingLayoutHandle dummy[DescriptorSetIDs::COUNT];
    };

    void CreateCommon(nvrhi::IDevice* device, donut::engine::ShaderFactory& shaderFactory, nvrhi::GraphicsAPI api, const BindingLayouts& layouts);

    bool CreateRayTracing(nvrhi::IDevice* device, donut::engine::ShaderFactory& shaderFactory, const BindingLayouts& layouts, bool enableNrd);

    Permutation& GetPermutation(Type t);

#if ENABLE_SHARC
    nvrhi::ComputePipelineHandle GetSharcResolvePSO() const;
#endif // ENABLE_SHARC

#if ENABLE_NRD
    nvrhi::ComputePipelineHandle GetDenoiserReblurPackPSO() const;

#if ENABLE_NRC
    nvrhi::ComputePipelineHandle GetDenoiserReblurPack_NRC_PSO() const;
#endif // ENABLE_NRC

    nvrhi::ComputePipelineHandle GetDenoiserResolvePSO() const;
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
    nvrhi::ShaderHandle m_denoiserReblurPackCS;
    nvrhi::ComputePipelineHandle m_denoiserReblurPackPSO;

#if ENABLE_NRC
    nvrhi::ShaderHandle m_denoiserReblurPack_NRC_CS;
    nvrhi::ComputePipelineHandle m_denoiserReblurPack_NRC_PSO;
#endif // ENABLE_NRC

    nvrhi::ShaderHandle m_denoiserResolveCS;
    nvrhi::ComputePipelineHandle m_denoiserResolvePSO;
#endif // ENABLE_NRD

    nvrhi::ShaderHandle m_tonemappingPS;
};