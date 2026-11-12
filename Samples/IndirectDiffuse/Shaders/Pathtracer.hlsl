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

#include <donut/shaders/bindless.h>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/gbuffer.hlsli>
#include <donut/shaders/binding_helpers.hlsli>
#include <donut/shaders/packing.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/lighting.hlsli>
#include <donut/shaders/scene_material.hlsli> 

#define ENABLE_NRC 1
#include "Nrc.hlsli"

#include "Brdf.hlsli"
#include "../GlobalCb.h"
#include "../LightingCb.h"
#include "PathtracerUtils.hlsli"

#define SHARC_MATERIAL_DEMODULATION     1

#include "SharcCommon.h"

#define BOUNCES_MIN                     3
#define RIS_CANDIDATES_LIGHTS           8 // Number of candidates used for resampling of analytical lights
#define SHADOW_RAY_IN_RIS               1 // Enable this to cast shadow rays for each candidate during resampling. This is expensive but increases quality
#define SHADOW_RAY_INDEX                1 // Hit group of the shadow rays when SHADOW_RAY_USE_RAY_QUERY (GlobalCb.h) is 0
#define TRACING_DISTANCE                1000.0f
#define MEDIUM_STACK_DEPTH              4 // Max nesting depth of overlapping transmissive materials tracked per path
#define SHARC_ENABLE_DEBUG              1

// Shader Execution Reordering: the ENABLE_SER permutations (see Shaders.cfg) reorder threads by hit before invoking
// the closest-hit/miss shaders. D3D12 uses the Shader Model 6.9 HitObject API; DXC cannot lower that API to SPIR-V,
// so Vulkan uses the hit object instructions of VK_NV_ray_tracing_invocation_reorder (HitObjectNV.hlsli).
// TARGET_D3D12 is defined by the DXIL shader build.
#ifndef ENABLE_SER
#define ENABLE_SER                      0
#endif
#ifndef TARGET_D3D12
#define TARGET_D3D12                    0
#endif
#if ENABLE_SER && !TARGET_D3D12
#include "HitObjectNV.hlsli"
#endif

#ifndef NRC_INDIRECT_DIFFUSE
#define NRC_INDIRECT_DIFFUSE 0
#endif

#ifndef SHARC_INDIRECT_DIFFUSE
#define SHARC_INDIRECT_DIFFUSE 0
#endif

struct [raypayload] RayPayload
{
    float hitDistance       : read(caller) : write(closesthit, miss);
    uint instanceID         : read(caller) : write(closesthit);
    uint primitiveIndex     : read(caller) : write(closesthit);
    uint geometryIndex      : read(caller) : write(closesthit);
    float2 barycentrics     : read(caller) : write(closesthit);

    bool Hit() { return hitDistance > 0.0f; }
    bool IsFrontFacing() { return asuint(hitDistance) & 0x1; }
};

struct [raypayload] ShadowRayPayload
{
    float3 visibility       : read(caller, anyhit) : write(caller, closesthit, anyhit);
};

ConstantBuffer<LightingConstants>           g_Lighting                      : register(b0, space0);
ConstantBuffer<GlobalConstants>             g_Global                        : register(b1, space0);

RaytracingAccelerationStructure             SceneBVH                        : register(t0, space0);
StructuredBuffer<InstanceData>              t_InstanceData                  : register(t1, space0);
StructuredBuffer<GeometryData>              t_GeometryData                  : register(t2, space0);
StructuredBuffer<MaterialConstants>         t_MaterialConstants             : register(t3, space0);

RWTexture2D<float4>                         u_Output                        : register(u0, space0);
#if ENABLE_NRD
RWTexture2D<float4>                         u_OutputDiffuseHitDistance      : register(u0, space1);
#endif
SamplerState                                s_MaterialSampler               : register(s0, space0);

// reg, dset
VK_BINDING(0, VK_BINDLESS_DESCRIPTOR_SET) ByteAddressBuffer  t_BindlessBuffers[]  : register(t0, space1);
VK_BINDING(1, VK_BINDLESS_DESCRIPTOR_SET) Texture2D          t_BindlessTextures[] : register(t0, space2);

#if NRC_INDIRECT_DIFFUSE || SHARC_INDIRECT_DIFFUSE
Texture2D<float>                            t_GBufferDepth                  : register(t0, space5);
Texture2D<float4>                           t_GBufferDiffuse                : register(t1, space5);
Texture2D<float4>                           t_GBufferSpecular               : register(t2, space5);
Texture2D<float4>                           t_GBufferNormals                : register(t3, space5);
Texture2D<float4>                           t_GBufferEmissive               : register(t4, space5);
Texture2D<float4>                           t_GBufferGeometryNormalAndPlaneDistance : register(t6, space5);
#endif

RWStructuredBuffer<NrcPackedQueryPathInfo>      queryPathInfo               : register(u0, space2); // Misc path info (vertexCount, queryIndex)
RWStructuredBuffer<NrcPackedTrainingPathInfo>   trainingPathInfo            : register(u1, space2); // Misc path info (vertexCount, queryIndex)
RWStructuredBuffer<NrcPackedPathVertex>         trainingPathVertices        : register(u2, space2); // Path vertex data used to train the neural radiance cache
RWStructuredBuffer<NrcRadianceParams>           queryRadianceParams         : register(u3, space2);
RWStructuredBuffer<uint>                        countersData                : register(u4, space2);

#define WRITE_TRAINING_DEBUG_PARAMS
#define WRITE_TRAINING_OUTPUT_DEBUG_PARAMS
#define WRITE_QUERY_OUTPUT_DEBUG_PARAMS

RWStructuredBuffer<HashGridKey>             u_SharcHashEntriesBuffer        : register(u0, space3);
RWStructuredBuffer<uint>                    u_SharcLockBuffer               : register(u1, space3);
RWStructuredBuffer<SharcAccumulationData>   u_SharcAccumulationBuffer       : register(u2, space3);
RWStructuredBuffer<SharcPackedData>         u_SharcResolvedBuffer           : register(u3, space3);

// Evaluates a non-opaque shadow-ray candidate: alpha-tested materials occlude when they pass the cutoff, other
// materials tint the visibility by their transmission. Returns true when the light is occluded and traversal can stop.
// Shared by the inline RayQuery traversal and the AnyHitShadow fallback so both paths shade identically.
bool EvaluateShadowHit(uint instanceID, uint primitiveIndex, uint geometryIndex, float2 barycentrics, inout float3 visibility)
{
    GeometrySample geometry = GetGeometryFromHit(instanceID, primitiveIndex, geometryIndex, barycentrics, GeomAttr_TexCoord, t_InstanceData, t_GeometryData, t_MaterialConstants, t_BindlessBuffers);
    MaterialSample material = SampleGeometryMaterial(geometry, 0, 0, 0, MatAttr_All, s_MaterialSampler, t_BindlessTextures);

    switch (geometry.material.domain)
    {
        case MaterialDomain_AlphaTested:
        case MaterialDomain_TransmissiveAlphaTested:
            return material.opacity >= geometry.material.alphaCutoff;

        default:
            // Modulate the visibility by the material's transmission
            visibility *= (1.0f - material.opacity) * material.baseColor;
            return dot(visibility, 0.333f) <= 0.001f;
    }
}

// Casts a shadow ray and returns the light visibility: 1 when unoccluded, 0 when fully occluded, tinted by transmissive occluders.
// Opaque geometry is flagged as such in the acceleration structure, so only non-opaque candidates reach EvaluateShadowHit.
float3 CastShadowRay(float3 hitPosition, float3 surfaceNormal, float3 directionToLight, float tracingDistance)
{
    // No need to cast shadow rays for back-facing primitives if there is no transmission
    if (!g_Global.enableTransmission && dot(surfaceNormal, directionToLight) < 0)
    {
        return float3(0, 0, 0);
    }

    RayDesc ray;
    ray.Origin = OffsetRay(hitPosition, surfaceNormal);
    ray.Direction = directionToLight;
    ray.TMin = 0.0f;
    ray.TMax = tracingDistance;

#if SHADOW_RAY_USE_RAY_QUERY
    // Inline traversal: the template flag commits the first opaque candidate and ends the search, while non-opaque
    // candidates that do not occlude are left uncommitted (the equivalent of IgnoreHit) and traversal continues.
    float3 visibility = float3(1.0f, 1.0f, 1.0f);

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);

    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            EvaluateShadowHit(query.CandidateInstanceID(), query.CandidatePrimitiveIndex(), query.CandidateGeometryIndex(), query.CandidateTriangleBarycentrics(), visibility))
        {
            return float3(0.0f, 0.0f, 0.0f);
        }
    }

    return query.CommittedStatus() == COMMITTED_NOTHING ? visibility : float3(0.0f, 0.0f, 0.0f);
#else // !SHADOW_RAY_USE_RAY_QUERY
    // Dedicated hit group with simpler shaders for shadow rays
    ShadowRayPayload payload;
    payload.visibility = float3(1.0f, 1.0f, 1.0f);

    uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    TraceRay(SceneBVH, rayFlags, 0xFF, SHADOW_RAY_INDEX, 0, SHADOW_RAY_INDEX, ray, payload);

    return payload.visibility;
#endif // SHADOW_RAY_USE_RAY_QUERY
}

// Light sample selected by SampleLightRIS. The sampled light point and its shadow ray result travel with the sample so
// the shading step evaluates exactly the point that passed resampling. Re-drawing the point at shading time would apply
// the visibility twice in penumbrae (once through the candidate surviving, once at shading) and darken soft shadows.
struct LightSample
{
    LightConstants light;
    float3 incidentVector; // Light to surface, as returned by GetLightData
    float lightDistance;
    float irradiance;
    float3 visibility; // Resolved during resampling when SHADOW_RAY_IN_RIS, otherwise 1 and the caller casts the shadow ray
    float weight; // RIS contribution weight: (1 / M) * sum(w_i) / pHat(selected)
};

// Samples a random light from the pool of all lights using RIS (Resampled Importance Sampling)
bool SampleLightRIS(inout uint rngState, float3 hitPosition, float3 surfaceNormal, out LightSample lightSample)
{
    lightSample = (LightSample)0;
    lightSample.visibility = float3(1.0f, 1.0f, 1.0f);
    lightSample.weight = 1.0f;

    if (g_Lighting.lightCount == 0)
    {
        return false;
    }

    if (g_Lighting.lightCount == 1)
    {
        lightSample.light = g_Lighting.lights[0];
        float2 rand2 = float2(Rand(rngState), Rand(rngState));
        GetLightData(lightSample.light, hitPosition, rand2, g_Global.enableSoftShadows, lightSample.incidentVector, lightSample.lightDistance, lightSample.irradiance);

#if SHADOW_RAY_IN_RIS
        lightSample.visibility = CastShadowRay(hitPosition, surfaceNormal, -lightSample.incidentVector, lightSample.lightDistance);
        return any(lightSample.visibility > 0.0f);
#else
        return true;
#endif
    }

    float totalWeights = 0.0f;
    float samplePdfG = 0.0f;

    // A target light outside the uploaded range is ignored rather than read out of bounds
    const bool forceTargetLight = g_Global.targetLight >= 0 && g_Global.targetLight < int(g_Lighting.lightCount);

    // Reciprocal of the proposal PDF, the RIS weight of every candidate: the uniform draw has PDF 1 / lightCount, so the
    // weight is lightCount. A forced target light is drawn with probability one, so its weight is 1; keeping lightCount
    // there would render the light lightCount times brighter than its isolated contribution.
    const float candidateWeight = forceTargetLight ? 1.0f : float(g_Lighting.lightCount);

    const uint candidateMax = min(g_Lighting.lightCount, RIS_CANDIDATES_LIGHTS);
    for (uint i = 0; i < candidateMax; i++)
    {
        uint randomLightIndex = forceTargetLight ? uint(g_Global.targetLight) : min(g_Lighting.lightCount - 1, uint(Rand(rngState) * g_Lighting.lightCount));
        LightConstants candidate = g_Lighting.lights[randomLightIndex];

        float3 incidentVector;
        float lightDistance;
        float irradiance;
        float2 rand2 = float2(Rand(rngState), Rand(rngState));
        GetLightData(candidate, hitPosition, rand2, g_Global.enableSoftShadows, incidentVector, lightDistance, irradiance);

        float3 visibility = float3(1.0f, 1.0f, 1.0f);
#if SHADOW_RAY_IN_RIS
        // Casting a shadow ray for all candidates here is expensive, but can significantly decrease noise:
        // occluded candidates get a zero target weight and can never be selected.
        visibility = CastShadowRay(hitPosition, surfaceNormal, -incidentVector, lightDistance);
        if (!any(visibility > 0.0f))
        {
            continue;
        }
#endif

        const float candidatePdfG = irradiance;
        const float candidateRISWeight = candidatePdfG * candidateWeight;

        totalWeights += candidateRISWeight;
        if (Rand(rngState) < (candidateRISWeight / totalWeights))
        {
            lightSample.light = candidate;
            lightSample.incidentVector = incidentVector;
            lightSample.lightDistance = lightDistance;
            lightSample.irradiance = irradiance;
            lightSample.visibility = visibility;
            samplePdfG = candidatePdfG;
        }
    }

    if (totalWeights == 0.0f)
    {
        return false;
    }

    lightSample.weight = (totalWeights / float(candidateMax)) / samplePdfG;

    return true;
}

// Calculates probability of selecting BRDF (specular or diffuse) using the approximate Fresnel term
float GetSpecularBrdfProbability(MaterialSample material, float3 viewVector, float3 shadingNormal)
{
#if ENABLE_SPECULAR_LOBE
    // Fast path for mirrors
    if (material.metalness == 1.0f && material.roughness == 0.0f)
        return 1.0f;

    // Evaluate Fresnel term using the shading normal
    // Note: we use the shading normal instead of the microfacet normal (half-vector) for Fresnel term here. That's suboptimal for rough surfaces at grazing angles, but half-vector is yet unknown at this point
    float specularF0 = luminance(material.specularF0);
    float diffuseReflectance = luminance(material.diffuseAlbedo);

    float fresnel = saturate(luminance(evalFresnel(specularF0, shadowedF90(specularF0), max(0.0f, dot(viewVector, shadingNormal)))));

    // Approximate relative contribution of BRDFs using the Fresnel term
    float specular = fresnel;
    float diffuse = diffuseReflectance * (1.0f - fresnel); //< If diffuse term is weighted by Fresnel, apply it here as well

    // Return probability of selecting specular BRDF over diffuse BRDF
    float probability = (specular / max(0.0001f, (specular + diffuse)));

    // Clamp probability to avoid undersampling of less prominent BRDF
    return clamp(probability, 0.1f, 0.9f);
#else // !ENABLE_SPECULAR_LOBE
    return 0.0f;
#endif // ENABLE_SPECULAR_LOBE
}

struct IndirectDiffusePrimarySurface
{
    MaterialSample material;
    float3 positionWorld;
    float3 geometryNormal;
    float3 viewVector;

    // NRC needs this as the hit distance of its vertex 0 (the G-buffer surface). SHARC does not,
    // but keeping it here lets both cache paths share the surface.
    float cameraPathLength;

    // Metal-rough parameters of the rasterized material, see GBufferGeometryNormal.hlsl. They are kept out of
    // 'material' so that the primary keeps being shaded with the G-buffer's diffuseAlbedo and specularF0, like the
    // deferred pass does. CreateRootSurface() uses them to evaluate the surface like a traced hit.
    float metalness;
    bool hasMetalRoughParams;

    bool valid;
};

struct IndirectDiffuseSecondarySurface
{
    MaterialSample material;
    float3 positionWorld;
    float3 geometryNormal;
    float3 shadingNormal;
    float3 viewVector;

    // Hit the front face of the triangle: a transmissive bounce from here enters a medium rather than leaving one.
    bool isFrontFacing;
};

// IORs of the media the path is currently nested inside (air is implicit below the bottom of the stack). Tracked
// through front/back-face hits so that overlapping and nested transmissive objects refract correctly.
struct MediumStack
{
    float iors[MEDIUM_STACK_DEPTH];
    int depth;
};

// Surfaces that are pass-through for the radiance caches, see PASS_THROUGH_TRANSMISSION. NRC and SHaRC must agree on
// this so that both produce the same image through glass.
bool IsCachePassThrough(IndirectDiffuseSecondarySurface surface)
{
    return surface.material.transmission >= PASS_THROUGH_TRANSMISSION;
}

// Finalizes a material for the path: applies the roughness and metalness overrides of the UI and derives the BRDF
// inputs from the result. Every surface the caches see goes through here, traced hits and the G-buffer root alike, so
// that one cache entry or training target never mixes two material models.
// The BRDF inputs follow Donut's convention (the diffuse albedo of a dielectric is 0.96 * baseColor), which is what the
// G-buffer and the deferred pass use. Clearing hasMetalRoughParams makes the BRDF code take them verbatim; otherwise
// prepareBRDFData() would rederive the diffuse reflectance from baseColor without that factor. baseColor and metalness
// stay valid for the code that reads them directly.
void ApplyMaterialOverrides(inout MaterialSample material)
{
    if (!material.hasMetalRoughParams)
        return;

    material.roughness = lerp(g_Global.roughnessMin, g_Global.roughnessMax, material.roughness);
    material.metalness = lerp(g_Global.metalnessMin, g_Global.metalnessMax, material.metalness);
    material.diffuseAlbedo = lerp(material.baseColor * (1.0f - c_DielectricSpecular), 0.0f, material.metalness);
    material.specularF0 = lerp(c_DielectricSpecular, material.baseColor, material.metalness);
    material.hasMetalRoughParams = false;
}

IndirectDiffusePrimarySurface LoadIndirectDiffusePrimarySurface(uint2 pixelPosition)
{
    IndirectDiffusePrimarySurface surface = (IndirectDiffusePrimarySurface)0;

    const float depth = t_GBufferDepth[pixelPosition];
    if (depth <= 0.0f) // Donut uses reverse-Z.
        return surface;

    float4 gBufferChannels[4];
    gBufferChannels[0] = t_GBufferDiffuse[pixelPosition];
    gBufferChannels[1] = t_GBufferSpecular[pixelPosition];
    gBufferChannels[2] = t_GBufferNormals[pixelPosition];
    gBufferChannels[3] = t_GBufferEmissive[pixelPosition];

    surface.material = DecodeGBuffer(gBufferChannels);
    surface.hasMetalRoughParams = gBufferChannels[3].w >= 0.5f;
    surface.metalness = saturate(gBufferChannels[3].w - 1.0f);
    surface.positionWorld = ReconstructWorldPosition(g_Lighting.view, float2(pixelPosition) + 0.5f, depth);

    const float4 geometryNormalAndPlaneDistance = t_GBufferGeometryNormalAndPlaneDistance[pixelPosition];
    surface.geometryNormal = normalize(geometryNormalAndPlaneDistance.xyz);

    // Inverse projection can place the reconstructed point on either side of
    // the rasterized triangle. Project it back onto that triangle's plane
    // before applying the floating-point ray-origin offset.
    const float signedPlaneDistance = dot(surface.geometryNormal, surface.positionWorld) - geometryNormalAndPlaneDistance.w;
    surface.positionWorld -= surface.geometryNormal * signedPlaneDistance;

    const float3 viewIncident = GetIncidentVector(g_Lighting.view.cameraDirectionOrPosition, surface.positionWorld);
    surface.viewVector = -viewIncident;

    // The encoded derivative normal has an arbitrary sign. Orient it toward the camera-facing side of the rasterized surface.
    if (dot(surface.geometryNormal, surface.viewVector) < 0.0f)
    {
        surface.geometryNormal = -surface.geometryNormal;
    }

    // Keep the normal-mapped shading normal in the same hemisphere as the true triangle normal.
    if (dot(surface.geometryNormal, surface.material.shadingNormal) < 0.0f)
    {
        surface.material.shadingNormal = -surface.material.shadingNormal;
    }

    // Donut's DecodeGBuffer aliases geometryNormal to shadingNormal. Replace that alias with the true triangle normal.
    surface.material.geometryNormal = surface.geometryNormal;

    const float3 cameraPosition = g_Lighting.view.matViewToWorld[3].xyz;
    surface.cameraPathLength = length(surface.positionWorld - cameraPosition);
    surface.valid = true;

    return surface;
}

// Used by the update (training) passes of the caches to index the GBuffer since the update dispatch size is smaller compared to the GBuffer.
uint2 SelectSparseGBufferPixel(uint2 dispatchIndex, uint2 dispatchDimensions, uint2 frameDimensions, inout uint rngState)
{
    const uint2 safeDispatchDimensions = max(dispatchDimensions, uint2(1u, 1u));
    const uint2 safeFrameDimensions = max(frameDimensions, uint2(1u, 1u));
    const float2 randomOffset = float2(Rand(rngState), Rand(rngState));
    const float2 dispatchToFrameScale = float2(safeFrameDimensions) / float2(safeDispatchDimensions);
    const uint2 pixelPosition = uint2((float2(dispatchIndex) + randomOffset) * dispatchToFrameScale);

    return min(pixelPosition, safeFrameDimensions - 1u);
}

IndirectDiffuseSecondarySurface LoadIndirectDiffuseSecondarySurface(RayDesc ray, RayPayload payload)
{
    IndirectDiffuseSecondarySurface surface = (IndirectDiffuseSecondarySurface)0;

    const GeometrySample geometry = GetGeometryFromHit(
        payload.instanceID, payload.primitiveIndex, payload.geometryIndex, payload.barycentrics, GeomAttr_All, t_InstanceData, t_GeometryData, t_MaterialConstants, t_BindlessBuffers);

    surface.material = SampleGeometryMaterial(geometry, 0, 0, 0, MatAttr_All, s_MaterialSampler, t_BindlessTextures);
    surface.material.emissiveColor = g_Global.enableEmissives ? surface.material.emissiveColor : 0.0f;
    // With the Transmission toggle off, path rays never reach a transmissive surface (see AnyHit), so the material's
    // transmission needs no override here
    surface.isFrontFacing = payload.IsFrontFacing();

    ApplyMaterialOverrides(surface.material);

    surface.viewVector = -ray.Direction;
    surface.geometryNormal = geometry.flatNormal;
    if (dot(surface.geometryNormal, surface.viewVector) < 0.0f)
        surface.geometryNormal = -surface.geometryNormal;

    surface.shadingNormal = surface.material.shadingNormal;
    if (dot(surface.geometryNormal, surface.shadingNormal) < 0.0f)
        surface.shadingNormal = -surface.shadingNormal;

    surface.material.geometryNormal = surface.geometryNormal;
    surface.material.shadingNormal = surface.shadingNormal;

    surface.positionWorld = ray.Origin + ray.Direction * payload.hitDistance;

    return surface;
}

// Presents the G-buffer surface as a path vertex, so that the code handling traced hits (cache hit data, direct
// lighting, lobe sampling) can also run at the root of a path. The material goes through ApplyMaterialOverrides() like
// the material of a traced hit, so both routes evaluate a surface identically. For that the metal-rough parameters are
// restored first: the metalness comes from the G-buffer, and the base color is solved from the G-buffer's diffuse
// albedo (0.96 * baseColor * (1 - metalness)) or specular F0 (lerp(0.04, baseColor, metalness)), whichever is the
// better conditioned for this metalness. Transmission is zero because the G-buffer holds no transmissive geometry.
IndirectDiffuseSecondarySurface CreateRootSurface(IndirectDiffusePrimarySurface primarySurface)
{
    IndirectDiffuseSecondarySurface surface = (IndirectDiffuseSecondarySurface)0;

    surface.material = primarySurface.material;

    if (primarySurface.hasMetalRoughParams)
    {
        const float metalness = primarySurface.metalness;
        const float3 baseColorFromDiffuse = surface.material.diffuseAlbedo / ((1.0f - c_DielectricSpecular) * max(1.0f - metalness, 1e-4f));
        const float3 baseColorFromSpecular = (surface.material.specularF0 - c_DielectricSpecular * (1.0f - metalness)) / max(metalness, 1e-4f);

        surface.material.baseColor = saturate(metalness < 0.5f ? baseColorFromDiffuse : baseColorFromSpecular);
        surface.material.metalness = metalness;
        surface.material.hasMetalRoughParams = true;
    }

    ApplyMaterialOverrides(surface.material);

    surface.material.emissiveColor = g_Global.enableEmissives ? surface.material.emissiveColor : 0.0f;
    surface.positionWorld = primarySurface.positionWorld;
    surface.geometryNormal = primarySurface.geometryNormal;
    surface.shadingNormal = primarySurface.material.shadingNormal;
    surface.viewVector = primarySurface.viewVector;
    surface.isFrontFacing = true;

    return surface;
}

// The primary event is always diffuse. Secondary and later events may
// still select diffuse or specular lobes.
bool GeneratePrimaryDiffuseRay(IndirectDiffusePrimarySurface surface, inout uint rngState, out RayDesc ray, out float3 throughput, out float brdfPdf)
{
    ray = (RayDesc)0;
    throughput = 0.0f;
    brdfPdf = 0.0f;

    float3 rayDirection = 0.0f;
    float3 sampleWeight = 0.0f;

    // The irradiance-only output view leaves the primary diffuse albedo out of the path throughput. The cached techniques then
    // return (1 - F) * incident indirect radiance directly, which stays exact in channels where the albedo is zero; dividing the
    // albedo back out of the reflected signal cannot do that. Donut's G-buffer decode leaves hasMetalRoughParams false, so
    // prepareBRDFData takes diffuseAlbedo verbatim.
    MaterialSample material = surface.material;
    if (g_Global.indirectDiffuseIrradianceOnly)
        material.diffuseAlbedo = 1.0f;

    const bool sampled = evalIndirectCombinedBRDF(
        float2(Rand(rngState), Rand(rngState)),
        material.shadingNormal,
        surface.geometryNormal,
        surface.viewVector,
        material,
        DIFFUSE_TYPE,
        1.0f,
        rayDirection,
        sampleWeight,
        brdfPdf);

    if (!sampled)
        return false;

    // This contains the primary diffuse albedo (unless the irradiance-only view is active) and directional Fresnel attenuation.
    // Deferred lighting must not apply diffuse albedo again.
    throughput = sampleWeight;

    if (g_Global.enableOcclusion)
        throughput *= surface.material.occlusion;

    ray.Origin = OffsetRay(surface.positionWorld, surface.geometryNormal);
    ray.Direction = rayDirection;
    ray.TMin = 0.0f;
    ray.TMax = TRACING_DISTANCE;

    return true;
}

// Samples the specular, transmissive or diffuse lobe of a secondary surface, like the Pathtracer sample does. The
// primary diffuse event is forced, but the paths behind it are regular paths: without the transmissive lobe, emitters
// enclosed by glass (lamp covers, windows) would be unreachable, since emissive geometry has no next-event estimation.
bool GenerateSecondaryIndirectDiffuseRay(
    IndirectDiffuseSecondarySurface surface,
    inout uint rngState,
    inout float3 throughput,
    inout MediumStack mediumStack,
    out RayDesc ray,
    out int launchLobeType,
    out float launchLobeRoughness,
    out float brdfPdf)
{
    ray = (RayDesc)0;
    launchLobeType = DIFFUSE_TYPE;
    launchLobeRoughness = 1.0f;
    brdfPdf = 0.0f;

    const float specularProbability = GetSpecularBrdfProbability(surface.material, surface.viewVector, surface.shadingNormal);
    const float transmission = surface.material.transmission;

    if (Rand(rngState) < specularProbability)
    {
        launchLobeType = SPECULAR_TYPE;
        throughput /= max(specularProbability, 1e-4f);
    }
    else if (Rand(rngState) < transmission)
    {
        launchLobeType = TRANSMISSIVE_TYPE;
        throughput /= max((1.0f - specularProbability) * transmission, 1e-4f);
    }
    else
    {
        launchLobeType = DIFFUSE_TYPE;
        throughput /= max((1.0f - specularProbability) * (1.0f - transmission), 1e-4f);
    }

    // For a transmissive bounce, derive eta (the IOR of the medium the path is in over the IOR of the medium it is
    // about to enter) from the medium stack; the hit's facing tells entering a new medium apart from leaving one
    float refractiveIndex = 1.0f;
    if (launchLobeType == TRANSMISSIVE_TYPE)
    {
        // A back face reached with an empty stack is an exit without a recorded entry: the path started inside the
        // medium. The G-buffer holds no transmissive geometry, so the root of a path can lie inside a glass enclosure.
        // The path is then leaving glass for air, and not air for air, which would remove refraction and total
        // internal reflection. Paths that entered through a front face have their medium on the stack
        const bool isUnmatchedExit = !surface.isFrontFacing && mediumStack.depth == 0;
        const float currentIor = (mediumStack.depth > 0) ? mediumStack.iors[mediumStack.depth - 1] : (isUnmatchedExit ? g_Global.transmissionIor : 1.0f);
        const float targetIor = surface.isFrontFacing ? g_Global.transmissionIor : ((mediumStack.depth > 1) ? mediumStack.iors[mediumStack.depth - 2] : 1.0f);
        refractiveIndex = currentIor / targetIor;
    }

    float3 nextDirection = 0.0f;
    float3 brdfWeight = 0.0f;

    const bool sampled = evalIndirectCombinedBRDF(
        float2(Rand(rngState), Rand(rngState)), surface.shadingNormal, surface.geometryNormal, surface.viewVector, surface.material, launchLobeType, refractiveIndex, nextDirection, brdfWeight, brdfPdf);

    if (!sampled)
        return false;

    // The transmissive lobe only returns a direction that crosses the surface when it was refracted (total internal
    // reflection stays on the incident side), so the medium changes exactly when the direction crosses
    const bool crossesSurface = dot(surface.geometryNormal, nextDirection) <= 0.0f;
    if (launchLobeType == TRANSMISSIVE_TYPE && crossesSurface)
    {
        if (surface.isFrontFacing)
        {
            if (mediumStack.depth < MEDIUM_STACK_DEPTH)
                mediumStack.iors[mediumStack.depth++] = g_Global.transmissionIor;
        }
        else if (mediumStack.depth > 0)
        {
            mediumStack.depth--;
        }
    }

    if (g_Global.enableOcclusion)
        throughput *= surface.material.occlusion;

    throughput *= brdfWeight;

    launchLobeRoughness = launchLobeType == DIFFUSE_TYPE ? 1.0f : surface.material.roughness;

    // Refraction requires the ray offset to go to the other side of the surface
    ray.Origin = OffsetRay(surface.positionWorld, crossesSurface ? -surface.geometryNormal : surface.geometryNormal);
    ray.Direction = nextDirection;
    ray.TMin = 0.0f;
    ray.TMax = TRACING_DISTANCE;

    return true;
}

// This is locally direct lighting at a secondary surface, but it is indirect lighting relative to the camera
float3 EvaluateIndirectDirectLighting(IndirectDiffuseSecondarySurface surface, inout uint rngState)
{
    if (!g_Global.enableLighting)
        return 0.0f;

    LightSample lightSample;
    if (!SampleLightRIS(rngState, surface.positionWorld, surface.geometryNormal, lightSample))
        return 0.0f;

    const float3 directionToLight = -lightSample.incidentVector;

#if !SHADOW_RAY_IN_RIS
    // Cast shadow ray towards the selected light (with SHADOW_RAY_IN_RIS the visibility was resolved during resampling)
    lightSample.visibility = CastShadowRay(surface.positionWorld, surface.geometryNormal, directionToLight, lightSample.lightDistance);
#endif
    if (!any(lightSample.visibility > 0.0f))
        return 0.0f;

    return evalCombinedBRDF(surface.shadingNormal, directionToLight, surface.viewVector, surface.material) * lightSample.light.color * lightSample.irradiance * lightSample.weight * lightSample.visibility;
}

struct Attributes
{
    float2 uv;
};

[shader("miss")]
void Miss(inout RayPayload payload : SV_RayPayload)
{
    payload.hitDistance = -1.0f;
}

[shader("closesthit")]
void ClosestHit(inout RayPayload payload : SV_RayPayload, in Attributes attrib : SV_IntersectionAttributes)
{
    payload.instanceID = InstanceID();
    payload.primitiveIndex = PrimitiveIndex();
    payload.geometryIndex = GeometryIndex();
    payload.barycentrics = attrib.uv;

    uint packedDistance = asuint(RayTCurrent()) & (~0x1u);
    packedDistance |= HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? 0x1 : 0x0;
    payload.hitDistance = asfloat(packedDistance);
}

[shader("anyhit")]
void AnyHit(inout RayPayload payload : SV_RayPayload, in Attributes attrib : SV_IntersectionAttributes)
{
    GeometrySample geometry = GetGeometryFromHit(InstanceID(), PrimitiveIndex(), GeometryIndex(), attrib.uv, GeomAttr_TexCoord, t_InstanceData, t_GeometryData, t_MaterialConstants, t_BindlessBuffers);
    MaterialSample material = SampleGeometryMaterial(geometry, 0, 0, 0, MatAttr_All, s_MaterialSampler, t_BindlessTextures);

    // The G-buffer does not rasterize transmissive geometry, so the primary surfaces lie behind it. With the
    // Transmission toggle off, path rays skip it as well: treating the glass as opaque instead would seal off what the
    // G-buffer sees through it (an emitter behind a lamp cover would light nothing), and the caches would disagree on
    // the result. With the toggle on, the glass is hit and the path refracts through it.
    const int domain = geometry.material.domain;
    const bool isTransmissive =
        domain == MaterialDomain_Transmissive || domain == MaterialDomain_TransmissiveAlphaTested || domain == MaterialDomain_TransmissiveAlphaBlended;
    if (!g_Global.enableTransmission && isTransmissive)
        IgnoreHit();

    switch (domain)
    {
        case MaterialDomain_AlphaTested:
        case MaterialDomain_TransmissiveAlphaTested:
        {
            if (material.opacity < geometry.material.alphaCutoff)
                IgnoreHit();

            break;
        }

        default:
            break;
    }
    // AcceptHit but continue looking for the closest hit
}

// Traces a path segment through the main hit group. With SER the trace is split so threads can be reordered by their
// hit (shader and geometry) before the closest-hit/miss shaders run, which improves coherence of the shading work.
void TracePathRay(RayDesc ray, uint rayFlags, inout RayPayload payload)
{
#if ENABLE_SER && TARGET_D3D12
    dx::HitObject hit = dx::HitObject::TraceRay(SceneBVH, rayFlags, 0xFF, 0, 0, 0, ray, payload);
    dx::MaybeReorderThread(hit);
    dx::HitObject::Invoke(hit, payload);
#elif ENABLE_SER // Vulkan
    TraceRayReorderNV(SceneBVH, rayFlags, 0xFF, 0, 0, 0, ray, payload);
#else // !ENABLE_SER
    TraceRay(SceneBVH, rayFlags, 0xFF, 0, 0, 0, ray, payload);
#endif // ENABLE_SER
}

// TODO: Delete this
[shader("miss")]
void ShadowMiss(inout ShadowRayPayload payload : SV_RayPayload)
{
}

[shader("closesthit")]
void ClosestHitShadow(inout ShadowRayPayload payload : SV_RayPayload, in Attributes attrib : SV_IntersectionAttributes)
{
    payload.visibility = float3(0.0f, 0.0f, 0.0f);
}

[shader("anyhit")]
void AnyHitShadow(inout ShadowRayPayload payload : SV_RayPayload, in Attributes attrib : SV_IntersectionAttributes)
{
    if (!EvaluateShadowHit(InstanceID(), PrimitiveIndex(), GeometryIndex(), attrib.uv, payload.visibility))
    {
        IgnoreHit();
    }

    AcceptHitAndEndSearch();
}

#if NRC_INDIRECT_DIFFUSE && SHARC_INDIRECT_DIFFUSE
#error NRC and SHARC indirect-diffuse implementations are mutually exclusive.
#elif NRC_INDIRECT_DIFFUSE
#include "NrcIndirectDiffuse.hlsli"
#elif SHARC_INDIRECT_DIFFUSE
#include "SharcIndirectDiffuse.hlsli"
#else
#error An indirect-diffuse cache implementation must be selected.
#endif
