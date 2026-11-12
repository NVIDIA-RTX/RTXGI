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

#include <donut/shaders/gbuffer_cb.h>

#define MATERIAL_REGISTER_SPACE    GBUFFER_SPACE_MATERIAL
#define MATERIAL_CB_SLOT           GBUFFER_BINDING_MATERIAL_CONSTANTS
#define MATERIAL_DIFFUSE_SLOT      GBUFFER_BINDING_MATERIAL_DIFFUSE_TEXTURE
#define MATERIAL_SPECULAR_SLOT     GBUFFER_BINDING_MATERIAL_SPECULAR_TEXTURE
#define MATERIAL_NORMALS_SLOT      GBUFFER_BINDING_MATERIAL_NORMAL_TEXTURE
#define MATERIAL_EMISSIVE_SLOT     GBUFFER_BINDING_MATERIAL_EMISSIVE_TEXTURE
#define MATERIAL_OCCLUSION_SLOT    GBUFFER_BINDING_MATERIAL_OCCLUSION_TEXTURE
#define MATERIAL_TRANSMISSION_SLOT GBUFFER_BINDING_MATERIAL_TRANSMISSION_TEXTURE
#define MATERIAL_OPACITY_SLOT      GBUFFER_BINDING_MATERIAL_OPACITY_TEXTURE

#define MATERIAL_SAMPLER_REGISTER_SPACE GBUFFER_SPACE_VIEW
#define MATERIAL_SAMPLER_SLOT           GBUFFER_BINDING_MATERIAL_SAMPLER

#include <donut/shaders/scene_material.hlsli>
#include <donut/shaders/material_bindings.hlsli>
#include <donut/shaders/motion_vectors.hlsli>
#include <donut/shaders/forward_vertex.hlsli>
#include <donut/shaders/binding_helpers.hlsli>
#include <donut/shaders/utils.hlsli>

DECLARE_CBUFFER(GBufferFillConstants, c_GBuffer, GBUFFER_BINDING_VIEW_CONSTANTS, GBUFFER_SPACE_VIEW);

float3 ComputeGeometricNormal(float3 worldPosition, float3 fallbackNormal)
{
    // Perspective-correct interpolation keeps worldPosition on the triangle
    // plane, so both derivatives are tangent to that plane.
    float3 geometryNormal = cross(ddx(worldPosition), ddy(worldPosition));

    const float normalLengthSquared = dot(geometryNormal, geometryNormal);

    if (normalLengthSquared <= 1e-20f)
    {
        return normalize(fallbackNormal);
    }

    return geometryNormal * rsqrt(normalLengthSquared);
}

void main(
    in float4 i_position : SV_Position,
    in SceneVertex i_vtx,
    in bool i_isFrontFace : SV_IsFrontFace,
    out float4 o_channel0 : SV_Target0,
    out float4 o_channel1 : SV_Target1,
    out float4 o_channel2 : SV_Target2,
    out float4 o_channel3 : SV_Target3,
#if MOTION_VECTORS
    out float3 o_motion : SV_Target4,
    out float4 o_geometryNormalAndPlaneDistance : SV_Target5
#else
    out float4 o_geometryNormalAndPlaneDistance : SV_Target4
#endif
)
{
    // Calculate derivatives before the alpha-test discard.
    const float3 geometryNormal = ComputeGeometricNormal(i_vtx.pos, i_vtx.normal);

    MaterialTextureSample textures = SampleMaterialTexturesAuto(i_vtx.texCoord, g_Material.normalTextureTransformScale);

    MaterialSample surface = EvaluateSceneMaterial(i_vtx.normal, i_vtx.tangent, g_Material, textures);

#if ALPHA_TESTED
    if (g_Material.domain != MaterialDomain_Opaque)
        clip(surface.opacity - g_Material.alphaCutoff);
#endif

    if (!i_isFrontFace)
    {
        surface.shadingNormal = -surface.shadingNormal;
    }

    o_channel0.xyz = surface.diffuseAlbedo;
    o_channel0.w = surface.opacity;

    o_channel1.xyz = surface.specularF0;
    o_channel1.w = surface.occlusion;

    o_channel2.xyz = surface.shadingNormal;
    o_channel2.w = surface.roughness;

    o_channel3.xyz = surface.emissiveColor;
    // Donut leaves this channel unused. The cache update passes need the metalness to evaluate the visible surface
    // with the same material model as traced hits (see CreateRootSurface in Pathtracer.hlsl): 0 means the material
    // has no metal-rough parameters (also the clear value), otherwise the value is 1 + metalness.
    o_channel3.w = surface.hasMetalRoughParams ? 1.0f + saturate(surface.metalness) : 0.0f;

#if MOTION_VECTORS
    o_motion = GetMotionVector(i_position.xyz, i_vtx.prevPos, c_GBuffer.view, c_GBuffer.viewPrev);
#endif

    o_geometryNormalAndPlaneDistance = float4(geometryNormal, dot(geometryNormal, i_vtx.pos));
}
