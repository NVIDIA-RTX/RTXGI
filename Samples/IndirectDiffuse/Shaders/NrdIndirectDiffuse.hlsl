/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma pack_matrix(row_major)

#include "../GlobalCb.h"
#include "../LightingCb.h"

#include <donut/shaders/gbuffer.hlsli>

#include "NRD.hlsli"
#include "IndirectDiffuseCommon.hlsli"

#define BLOCK_SIZE 16

// Nota bene: 
//  - Don't subtract primary emissive here. The radiance cache output contains indirect diffuse only.
//  - Don't account for emissive during resolve- deferred lighting already supplies primary emissive.
//  - Demodulate/modulate only diffuseAlbedo.
//  - Preserve the Fresnel attenuation already contained in the sampled primary diffuse throughput.

ConstantBuffer<LightingConstants> g_Lighting : register(b0, space0);
ConstantBuffer<GlobalConstants> g_Global : register(b1, space0);

// NRC/SHARC resolve output. This contains reflected indirect diffuse.
RWTexture2D<float4> u_IndirectDiffuse : register(u0, space0);

// Keep the existing denoiser UAV slot assignments for now.
RWTexture2D<float4> u_DiffuseRadianceHitDistance : register(u0, space1);
RWTexture2D<float> u_ViewZ : register(u2, space1);
RWTexture2D<float4> u_NormalRoughness : register(u3, space1);
RWTexture2D<float4> u_MotionVectors : register(u4, space1);

// Shared G-buffer descriptor set.
Texture2D<float> t_GBufferDepth : register(t0, space5);
Texture2D<float4> t_GBufferDiffuse : register(t1, space5);
Texture2D<float4> t_GBufferNormals : register(t3, space5);
Texture2D<float4> t_GBufferMotionVectors : register(t5, space5);

bool IsOutsideRenderArea(uint2 pixelPosition)
{
    uint width;
    uint height;
    t_GBufferDepth.GetDimensions(width, height);

    return any(pixelPosition >= uint2(width, height));
}

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)] void PackNrdIndirectDiffuse(uint2 pixelPosition : SV_DispatchThreadID)
{
    if (IsOutsideRenderArea(pixelPosition))
        return;

    const float deviceDepth = t_GBufferDepth[pixelPosition];

    // Reverse-Z clear value: no primary surface.
    if (deviceDepth <= 0.0f)
    {
        u_DiffuseRadianceHitDistance[pixelPosition] = 0.0f;
        u_ViewZ[pixelPosition] = 0.0f;
        u_NormalRoughness[pixelPosition] = 0.0f;
        u_MotionVectors[pixelPosition] = 0.0f;

        return;
    }

    const float firstBounceHitDistance = u_DiffuseRadianceHitDistance[pixelPosition].w;
    const float3 reflectedIndirectDiffuse = u_IndirectDiffuse[pixelPosition].xyz;
    const float3 diffuseAlbedo = t_GBufferDiffuse[pixelPosition].xyz;

    // The cached output already contains the primary diffuse throughput. Remove only the primary diffuse albedo for
    // denoising; preserve directional Fresnel. The irradiance-only output view never applied the albedo, so pass it through.
    const float3 demodulatedIndirectDiffuse = g_Global.indirectDiffuseIrradianceOnly ? reflectedIndirectDiffuse : DemodulateIndirectDiffuse(reflectedIndirectDiffuse, diffuseAlbedo);
    const float3 viewPosition = ReconstructViewPosition(g_Lighting.view, float2(pixelPosition) + 0.5f, deviceDepth);
    const float viewZ = viewPosition.z;
    const float normalizedHitDistance = REBLUR_FrontEnd_GetNormHitDist(firstBounceHitDistance, viewZ, g_Global.nrdHitDistanceParams.xyz, 1.0f);

    u_DiffuseRadianceHitDistance[pixelPosition] = REBLUR_FrontEnd_PackRadianceAndNormHitDist(demodulatedIndirectDiffuse, normalizedHitDistance, true);

    const float4 normalRoughness = t_GBufferNormals[pixelPosition];

    u_NormalRoughness[pixelPosition] = NRD_FrontEnd_PackNormalAndRoughness(normalRoughness.xyz, normalRoughness.w, 0.0f);

    // Donut's z motion is device-depth delta, not NRD view-Z delta.
    // Start with 2D motion and explicitly discard z.
    const float2 motion = t_GBufferMotionVectors[pixelPosition].xy;

    u_MotionVectors[pixelPosition] = float4(motion, 0.0f, 0.0f);

    u_ViewZ[pixelPosition] = viewZ;
}

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)] void ResolveNrdIndirectDiffuse(uint2 pixelPosition : SV_DispatchThreadID)
{
    if (IsOutsideRenderArea(pixelPosition))
        return;

    if (t_GBufferDepth[pixelPosition] <= 0.0f)
    {
        u_IndirectDiffuse[pixelPosition] = 0.0f;
        return;
    }

    float4 diffuseData = u_DiffuseRadianceHitDistance[pixelPosition];
    diffuseData = REBLUR_BackEnd_UnpackRadianceAndNormHitDist(diffuseData);
    const float3 diffuseAlbedo = t_GBufferDiffuse[pixelPosition].xyz;

    // Donut's default DeferredLightingPass expects reflected indirect diffuse. The irradiance-only output view bypasses it
    // and displays the denoised signal as is.
    const float3 modulation = g_Global.indirectDiffuseIrradianceOnly ? 1.0f : diffuseAlbedo;
    u_IndirectDiffuse[pixelPosition] = float4(diffuseData.xyz * modulation, 1.0f);
}