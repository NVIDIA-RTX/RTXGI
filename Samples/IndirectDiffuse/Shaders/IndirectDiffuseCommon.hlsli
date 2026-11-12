/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef INDIRECT_DIFFUSE_COMMON_HLSLI
#define INDIRECT_DIFFUSE_COMMON_HLSLI

static const float c_IndirectDiffuseAlbedoMin = 1e-4f;

// Removes the primary diffuse albedo carried by the cached output. The Fresnel attenuation and
// occlusion sampled with the primary diffuse event stay in the signal.
float3 DemodulateIndirectDiffuse(float3 reflectedIndirectDiffuse, float3 diffuseAlbedo)
{
    return reflectedIndirectDiffuse / max(diffuseAlbedo, c_IndirectDiffuseAlbedoMin);
}

#endif // INDIRECT_DIFFUSE_COMMON_HLSLI
