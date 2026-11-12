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
#define SHARC_ENABLE_DEBUG              1
#define MEDIUM_STACK_DEPTH              4 // Max nesting depth of overlapping transmissive materials tracked per path

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
SamplerState                                s_MaterialSampler               : register(s0, space0);

// reg, dset
VK_BINDING(0, 4) ByteAddressBuffer          t_BindlessBuffers[]             : register(t0, space1);
VK_BINDING(1, 4) Texture2D                  t_BindlessTextures[]            : register(t0, space2);

RWTexture2D<float4>                         u_OutputDiffuseHitDistance      : register(u0, space1);
RWTexture2D<float4>                         u_OutputSpecularHitDistance     : register(u1, space1);
RWTexture2D<float>                          u_OutputDepth                   : register(u2, space1);
RWTexture2D<float4>                         u_OutputNormalRoughness         : register(u3, space1);
RWTexture2D<float4>                         u_OutputMotionVectors           : register(u4, space1);
RWTexture2D<float4>                         u_OutputEmissive                : register(u5, space1);
RWTexture2D<float4>                         u_OutputDiffuseAlbedo           : register(u6, space1);
RWTexture2D<float4>                         u_OutputSpecularAlbedo          : register(u7, space1);

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

RayDesc GeneratePinholeCameraRay(float2 normalizedDeviceCoordinate, float4x4 viewToWorld, float4x4 viewToClip)
{
    // Set up the ray
    RayDesc ray;
    ray.Origin = viewToWorld[3].xyz;
    ray.TMin = 0.0f;
    ray.TMax = TRACING_DISTANCE;

    // Extract the aspect ratio and fov from the projection matrix
    float aspect = viewToClip[1][1] / viewToClip[0][0];
    float tanHalfFovY = 1.0f / viewToClip[1][1];

    // Compute the ray direction
    ray.Direction = normalize(
        ((normalizedDeviceCoordinate.x * 2.f - 1.f) * viewToWorld[0].xyz * tanHalfFovY * aspect) -
        ((normalizedDeviceCoordinate.y * 2.f - 1.f) * viewToWorld[1].xyz * tanHalfFovY) +
        viewToWorld[2].xyz);

    return ray;
}

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
        return float3(0, 0, 0);

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
        return false;

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
            continue;
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
        return false;

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

    switch (geometry.material.domain)
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

struct AccumulatedSampleData
{
    float3 radiance;
#if ENABLE_NRD
    float hitDistance;
#if ENABLE_SPECULAR_LOBE
    uint diffuseSampleNum;
    float3 specularRadiance;
    float specularHitDistance;
#endif // ENABLE_SPECULAR_LOBE
#endif // ENABLE_NRD
};

void UpdateSampleData(inout AccumulatedSampleData accumulatedSampleData, float3 sampleRadiance, bool isDiffusePath, float hitDistance)
{
#if ENABLE_NRD
#if ENABLE_SPECULAR_LOBE
    if (!isDiffusePath)
    {
        accumulatedSampleData.specularRadiance += sampleRadiance;
        accumulatedSampleData.specularHitDistance += hitDistance;
        return;
    }
    accumulatedSampleData.diffuseSampleNum++;
#endif // ENABLE_SPECULAR_LOBE
    accumulatedSampleData.hitDistance += hitDistance;
#endif // ENABLE_NRD

    accumulatedSampleData.radiance += sampleRadiance;
}

void ResolveSampleData(inout AccumulatedSampleData accumulatedSampleData, uint sampleNum, float intensityScale)
{
#if ENABLE_NRD
#if ENABLE_SPECULAR_LOBE
    uint specularSampleNum = sampleNum - accumulatedSampleData.diffuseSampleNum;
    if (specularSampleNum)
    {
        accumulatedSampleData.specularRadiance *= (intensityScale / specularSampleNum);
        accumulatedSampleData.specularHitDistance *= (1.0f / specularSampleNum);
    }
    u_OutputSpecularHitDistance[DispatchRaysIndex().xy] = float4(accumulatedSampleData.specularRadiance, accumulatedSampleData.specularHitDistance);
    uint diffuseSampleNum = accumulatedSampleData.diffuseSampleNum;
#else // !ENABLE_SPECULAR_LOBE
    uint diffuseSampleNum = sampleNum;
#endif // ENABLE_SPECULAR_LOBE
    if (diffuseSampleNum)
    {
        accumulatedSampleData.radiance *= (intensityScale / diffuseSampleNum);
        accumulatedSampleData.hitDistance *= (1.0f / diffuseSampleNum);
    }
    u_OutputDiffuseHitDistance[DispatchRaysIndex().xy] = float4(accumulatedSampleData.radiance, accumulatedSampleData.hitDistance);
#else // !ENABLE_NRD
    accumulatedSampleData.radiance *= (intensityScale / sampleNum);
    u_Output[DispatchRaysIndex().xy] = float4(accumulatedSampleData.radiance, 1.0f);
#endif // ENABLE_NRD
}

void PathTraceRays()
{
#if ENABLE_NRD
    const bool enableNrd = true;
#else // !ENABLE_NRD
    const bool enableNrd = false;
#endif // ENABLE_NRD

#if (NRC_UPDATE || SHARC_UPDATE)
    const bool isUpdatePass = true;
#else // !(NRC_UPDATE || SHARC_UPDATE)
    const bool isUpdatePass = false;
#endif // (NRC_UPDATE || SHARC_UPDATE)

    // NRC training paths are limited by the context's maxPathVertices, not by the rendering bounce count: the training
    // vertex buffer holds exactly maxPathVertices entries per path, and NrcUpdateOnHit() only checks the limit at vertices
    // it records. A delta vertex (mirror, pass-through glass) at the last allowed bounce skips that check, and a path that
    // continues past it would write into the next path's entries. The loop below enforces the limit for every bounce
#if NRC_UPDATE
    const int bouncesMax = int(g_Lighting.nrcConstants.maxPathVertices);
#else // !NRC_UPDATE
    const int bouncesMax = g_Global.bouncesMax;
#endif // NRC_UPDATE

    const uint2 launchIndex = DispatchRaysIndex().xy;
    const uint2 launchDimensions = DispatchRaysDimensions().xy;
    uint rngState = InitRNG(launchIndex, launchDimensions, g_Global.frameIndex);

    AccumulatedSampleData accumulatedSampleData = (AccumulatedSampleData)0;
    float3 debugColor = float3(0.0f, 0.0f, 0.0f);

    // Initialize common resources required by the NRC
    NrcBuffers buffers;
    buffers.queryPathInfo = queryPathInfo;
    buffers.trainingPathInfo = trainingPathInfo;
    buffers.trainingPathVertices = trainingPathVertices;
    buffers.queryRadianceParams = queryRadianceParams;
    buffers.countersData = countersData;

    // Create NrcContext
    NrcContext nrcContext = NrcCreateContext(g_Lighting.nrcConstants, buffers, launchIndex);

    // Initialize SHARC parameters
    SharcParameters sharcParameters;
    {
        sharcParameters.hashGridParameters.cameraPosition = g_Lighting.sharcCameraPosition.xyz;
        sharcParameters.hashGridParameters.sceneScale = g_Lighting.sharcSceneScale;
        sharcParameters.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        sharcParameters.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

        sharcParameters.hashGridData.capacity = g_Lighting.sharcEntriesNum;
        sharcParameters.hashGridData.hashEntriesBuffer = u_SharcHashEntriesBuffer;

#if !HASH_GRID_ENABLE_64_BIT_ATOMICS
        sharcParameters.hashGridData.lockBuffer = u_SharcLockBuffer;
#endif // !HASH_GRID_ENABLE_64_BIT_ATOMICS

        sharcParameters.accumulationBuffer = u_SharcAccumulationBuffer;
        sharcParameters.resolvedBuffer = u_SharcResolvedBuffer;
        sharcParameters.radianceScale = SHARC_RADIANCE_SCALE;
    }

    const uint samplesPerPixel = isUpdatePass ? 1 : g_Global.samplesPerPixel;
    for (uint sampleIndex = 0; sampleIndex < samplesPerPixel; sampleIndex++)
    {
        // Initialize NRC data for path and sample index traced in this thread
        NrcPathState nrcPathState = NrcCreatePathState(g_Lighting.nrcConstants, Rand(rngState));
        NrcSetSampleIndex(nrcContext, sampleIndex);

        // Initialize SHARC state
        SharcState sharcState;
        SharcInit(sharcState);

        float2 pixel = launchIndex + float2(0.5f, 0.5f) - g_Lighting.view.pixelOffset;
        RayDesc ray = GeneratePinholeCameraRay(pixel / launchDimensions,
            isUpdatePass ? g_Lighting.updatePassView.matViewToWorld : g_Lighting.view.matViewToWorld,
            isUpdatePass ? g_Lighting.updatePassView.matViewToClip : g_Lighting.view.matViewToClip);

        float3 sampleRadiance = float3(0.0f, 0.0f, 0.0f);
        float3 throughput = float3(1.0f, 1.0f, 1.0f);

        float materialRoughnessPrev = 0.0f;
        bool isDiffusePath = true; // Used by denoiser
        float hitDistance = 0.0f; // Used by denoiser

        // Stack of IORs for the media the path is currently nested inside (bottom of stack is air, implicit).
        // Tracked via front/back-face hits so overlapping/nested transmissive objects refract correctly.
        float mediumIorStack[MEDIUM_STACK_DEPTH];
        int mediumStackDepth = 0;

        int bounce;
        for (bounce = 0; true/* break from the middle */; bounce++)
        {
            uint rayFlags = RAY_FLAG_NONE;
            RayPayload payload;
            TracePathRay(ray, rayFlags, payload);

#if SHARC_UPDATE
            // SHaRC handles radiance propagation per segment; no path throughput needed.
            // Reset throughput to 1 at each bounce.
            sampleRadiance = float3(0.0f, 0.0f, 0.0f);
            throughput = float3(1.0f, 1.0f, 1.0f);
#endif // SHARC_UPDATE

            if (enableNrd && bounce == 1)
                hitDistance = payload.Hit() ? payload.hitDistance : TRACING_DISTANCE;

            // On a miss, load the sky value and break out of the ray tracing loop
            if (!payload.Hit())
            {
                float3 skyValue = g_Lighting.skyColor.rgb;

                SharcUpdateMiss(sharcParameters, sharcState, skyValue);

                NrcUpdateOnMiss(nrcPathState);

                sampleRadiance += skyValue * throughput;

                if (enableNrd && bounce == 0)
                    u_Output[launchIndex] = float4(sampleRadiance, 1.0f);

                break;
            }

            GeometrySample geometry = GetGeometryFromHit(payload.instanceID, payload.primitiveIndex, payload.geometryIndex, payload.barycentrics, GeomAttr_All, t_InstanceData, t_GeometryData, t_MaterialConstants, t_BindlessBuffers);
            MaterialSample material = SampleGeometryMaterial(geometry, 0, 0, 0, MatAttr_All, s_MaterialSampler, t_BindlessTextures);
            material.emissiveColor = g_Global.enableEmissives ? material.emissiveColor : 0;
            // Zeroing transmission here makes the toggle apply everywhere the material is used: lobe selection,
            // indirect sample weights, direct lighting BRDF evaluation and SHaRC pass-through
            material.transmission = g_Global.enableTransmission ? material.transmission : 0.0f;

            if (material.hasMetalRoughParams)
            {
                // Remap roughness and metalness according to the UI sliders
                material.roughness = lerp(g_Global.roughnessMin, g_Global.roughnessMax, material.roughness);
                material.metalness = lerp(g_Global.metalnessMin, g_Global.metalnessMax, material.metalness);
                material.diffuseAlbedo = lerp(material.baseColor * (1.0 - c_DielectricSpecular), 0.0, material.metalness);
                material.specularF0 = lerp(c_DielectricSpecular, material.baseColor.rgb, material.metalness);
            }

            // Flip normals towards the incident ray direction (needed for backfacing triangles)
            float3 viewVector = -ray.Direction;

            // Flip the triangle normal, based on positional data, NOT the provided vertex normal
            float3 geometryNormal = geometry.flatNormal;
            if (dot(geometryNormal, viewVector) < 0.0f)
                geometryNormal = -geometryNormal;

            // Flip the shading normal, based on texture
            float3 shadingNormal = material.shadingNormal;
            if (dot(geometryNormal, shadingNormal) < 0.0f)
                shadingNormal = -shadingNormal;

            float3 hitPosition = ray.Origin + ray.Direction * payload.hitDistance;

            // Transmissive surfaces are pass-through for both caches (see PASS_THROUGH_TRANSMISSION): the glass is neither
            // cached nor queried and the path continues to the surfaces behind it. Both caches must agree on this so that
            // NRC and SHaRC produce the same image through a window
            const bool isCachePassThrough = material.transmission >= PASS_THROUGH_TRANSMISSION;

            // Construct NRCSurfaceData structure needed for creating a query point at this hit location
            NrcSurfaceAttributes surfaceAttributes = (NrcSurfaceAttributes)0;
            surfaceAttributes.encodedPosition = NrcEncodePosition(hitPosition, g_Lighting.nrcConstants);
            surfaceAttributes.roughness = material.roughness;
            surfaceAttributes.specularF0 = material.specularF0;
            surfaceAttributes.diffuseReflectance = material.diffuseAlbedo;
            surfaceAttributes.shadingNormal = shadingNormal;
            surfaceAttributes.viewVector = viewVector;
            // NRC skips vertices flagged as delta lobes: no training vertex is written and no query is created there, the
            // path's throughput and radiance keep accumulating into the previous segment, the vertex does not count towards
            // the path's vertex count and the path spread is not grown across it. That is how NRC looks through mirrors, and
            // it is exactly the pass-through behavior the transmissive surfaces need, so flag them the same way. Without
            // this NRC caches and terminates on the glass itself, which the network cannot represent, and the image
            // behind a window turns into a blur. Note that NRC only honors the flag with the termination heuristic
            // enabled (every resolve mode but DirectCacheView) or with skipDeltaVertices set
            surfaceAttributes.isDeltaLobe = (material.metalness == 1.0f && material.roughness == 0.0f) /* perfect mirrors */ || isCachePassThrough;

            NrcProgressState nrcProgressState = NrcUpdateOnHit(nrcContext, nrcPathState, surfaceAttributes, payload.hitDistance, bounce, throughput, sampleRadiance);
            if (nrcProgressState == NrcProgressState::TerminateImmediately)
                break;

            // Construct SharcHitData structure needed for creating a query point at this hit location
            SharcHitData sharcHitData;
            sharcHitData.positionWorld = hitPosition;
            sharcHitData.normalWorld = geometryNormal;
#if SHARC_ENABLE_SH_ENCODING
            sharcHitData.radianceDirectionWorld = viewVector;
            sharcHitData.radianceDirectionWeight = 0;
#endif // SHARC_ENABLE_SH_ENCODING
#if SHARC_MATERIAL_DEMODULATION
            sharcHitData.materialDemodulation = float3(1.0f, 1.0f, 1.0f);
            if (g_Global.sharcMaterialDemodulation)
            {
                float3 specularFAvg = material.specularF0 + (1.0f - material.specularF0) * 0.047619; // 1/21
                sharcHitData.materialDemodulation = max(material.diffuseAlbedo, 0.05f) + max(material.specularF0, 0.02f) * luminance(specularFAvg);
            }
#endif // SHARC_MATERIAL_DEMODULATION
#if SHARC_SEPARATE_EMISSIVE
            sharcHitData.emissive = material.emissiveColor;
#endif // SHARC_SEPARATE_EMISSIVE

#if SHARC_UPDATE
            material.roughness = max(g_Lighting.sharcRoughnessThreshold, material.roughness);
#endif // SHARC_UPDATE

#if SHARC_QUERY
            {
                uint gridLevel = HashGridGetLevel(hitPosition, sharcParameters.hashGridParameters);
                float voxelSize = HashGridGetVoxelSize(gridLevel, sharcParameters.hashGridParameters);
                bool isValidHit = !isCachePassThrough && payload.hitDistance > voxelSize * sqrt(3.0f);

                materialRoughnessPrev = min(materialRoughnessPrev, 0.99f);
                float alpha = materialRoughnessPrev * materialRoughnessPrev;
                float footprint = payload.hitDistance * sqrt(0.5f * alpha * alpha / (1.0f - alpha * alpha));
                isValidHit &= footprint > voxelSize;

                float3 sharcRadiance;
                if (isValidHit && SharcGetCachedRadiance(sharcParameters, sharcHitData, sharcRadiance, false))
                {
                    sampleRadiance += sharcRadiance * throughput;

                    break; // Terminate the path once we've looked up into the cache
                }

#if SHARC_ENABLE_DEBUG
                if (g_Global.sharcDebug)
                {
                    float3 debugColor;
                    SharcGetCachedRadiance(sharcParameters, sharcHitData, debugColor, true);
                    //debugColor = HashGridDebugOccupancy(launchIndex, launchDimensions, sharcParameters.hashGridData);
                    //debugColor = HashGridDebugHashCollisions(sharcHitData.positionWorld, sharcHitData.normalWorld, sharcParameters.hashGridParameters, sharcParameters.hashGridData);

                    u_Output[DispatchRaysIndex().xy] = float4(debugColor, 1.0f);

                    return;
                }
#endif // SHARC_ENABLE_DEBUG
            }
#endif // SHARC_QUERY

            if (g_Global.enableLighting)
            {
                // Evaluate direct light (next event estimation), start by sampling one light
                LightSample lightSample;
                if (SampleLightRIS(rngState, hitPosition, geometryNormal, lightSample))
                {
                    const float3 vectorToLight = -lightSample.incidentVector;

#if !SHADOW_RAY_IN_RIS
                    // Cast shadow ray towards the selected light (with SHADOW_RAY_IN_RIS the visibility was resolved during resampling)
                    lightSample.visibility = CastShadowRay(hitPosition, geometryNormal, vectorToLight, lightSample.lightDistance);
#endif
                    if (any(lightSample.visibility > 0.0f))
                    {
                        // If light is not in shadow, evaluate BRDF and accumulate its contribution into radiance
                        // This is an entry point for evaluation of all other BRDFs based on selected configuration (for direct light)
                        float3 lightContribution = evalCombinedBRDF(shadingNormal, vectorToLight, viewVector, material) * lightSample.light.color * lightSample.irradiance * lightSample.weight * lightSample.visibility;
                        sampleRadiance += lightContribution * throughput;
                    }
                }
            }

            // Emissive geometry has no next-event estimation (SampleLightRIS only samples analytic lights), so the
            // BRDF-sampled hit is its only estimator. It must be counted at every vertex, including the last one,
            // which is why this is placed before the max-bounce termination below.
#if !(SHARC_UPDATE && SHARC_SEPARATE_EMISSIVE)
            sampleRadiance += material.emissiveColor * throughput;
#endif

            // Terminate the loop early on the last bounce (we don't need to sample the BRDF)
            if (bounce == bouncesMax - 1)
            {
                NrcSetDebugPathTerminationReason(nrcPathState, NrcDebugPathTerminationReason::MaxPathVertices);
                break;
            }

            // Terminate the loop after the emissives and direct light contribution has been added if NRC CreateQuery call 
            // requested delayed termination. In case direct lighting is not being cached (radianceCacheDirect paramter is 
            //  false), we need to add direct lighting on hit where we query NRC before terminating the loop.
            if (nrcProgressState == NrcProgressState::TerminateAfterDirectLighting)
                break;

            if (isCachePassThrough)
            {
                // SHaRC has no notion of a delta vertex, so the pass-through is spelled out here: no cache entry for the
                // glass itself, but the radiance found at this vertex (direct lighting on the glass, emissive) still
                // belongs to the vertices before it. SharcUpdateMiss() propagates the radiance to the previous vertices
                // without inserting an entry/modifying state. SharcSetThroughput() then keeps the chain intact, so those
                // vertices also receive what's behind the glass.
#if SHARC_UPDATE && SHARC_SEPARATE_EMISSIVE
                // With separate emissive, SharcUpdateHit() adds sharcHitData.emissive to the propagated radiance itself
                // (and keeps it out of the vertex's own entry). SharcUpdateMiss() does not, and so we account for it here.
                sampleRadiance += material.emissiveColor * throughput;
#endif
                SharcUpdateMiss(sharcParameters, sharcState, sampleRadiance);
            }
            else if (!SharcUpdateHit(sharcParameters, sharcState, sharcHitData, sampleRadiance, Rand(rngState)))
            {
                break;
            }

            // Russian roulette
            if (g_Global.enableRussianRoulette && NrcCanUseRussianRoulette(nrcPathState) && (bounce > BOUNCES_MIN))
            {
                float rrProbability = min(0.95f, luminance(throughput));
                const bool terminate = (rrProbability < Rand(rngState));
                if (terminate)
                    break;
                else
                    throughput /= rrProbability;
            }

            // Sample BRDF to generate the next ray
            // First, figure out whether to sample diffuse or specular BRDF
            int brdfType = DIFFUSE_TYPE;

            float specularBrdfProbability = GetSpecularBrdfProbability(material, viewVector, shadingNormal);
            if (Rand(rngState) < specularBrdfProbability)
            {
                brdfType = SPECULAR_TYPE;
                throughput /= specularBrdfProbability;
            }
            else if (Rand(rngState) < material.transmission)
            {
                brdfType = TRANSMISSIVE_TYPE;
                throughput /= (1.0f - specularBrdfProbability) * material.transmission;
            }
            else
            {
                brdfType = DIFFUSE_TYPE;
                throughput /= (1.0f - specularBrdfProbability) * (1.0f - material.transmission);
            }

#if SHARC_QUERY
            materialRoughnessPrev += brdfType == DIFFUSE_TYPE ? 1.0f : material.roughness;
#endif // SHARC_QUERY

#if !SHARC_UPDATE && !NRC_UPDATE
            if (bounce == 0)
            {
                isDiffusePath = brdfType == DIFFUSE_TYPE;
                // Fill GBuffer data
                if (sampleIndex == 0)
                {
                    u_OutputDepth[launchIndex] = dot(hitPosition - g_Lighting.view.matViewToWorld[3].xyz, g_Lighting.view.matViewToWorld[2].xyz);
                    u_OutputNormalRoughness[launchIndex] = float4(shadingNormal, material.roughness);
                    // Motion vectors: reproject where this surface point was on the previous frame (previous skinned
                    // position and previous instance transform) with the previous view, so animated geometry
                    // contributes its own motion and not only the camera's
                    {
                        float3 hitPositionPrev = ComputePreviousHitPosition(geometry, payload.primitiveIndex, payload.barycentrics, t_BindlessBuffers);

                        float4 positionClip = mul(float4(hitPosition, 1.0f), g_Lighting.view.matWorldToClipNoOffset);
                        positionClip.xyz /= positionClip.w;
                        float4 positionClipPrev = mul(float4(hitPositionPrev, 1.0f), g_Lighting.viewPrev.matWorldToClipNoOffset);

                        // The depth delta is valid even when the point was behind the previous camera, and a negative
                        // previous depth lets the denoiser reject history for it
                        float3 motionVector;
                        motionVector.z = positionClipPrev.w - positionClip.w;

                        if (positionClipPrev.w > 0.0f)
                        {
                            positionClipPrev.xy /= positionClipPrev.w;
                            motionVector.xy = (positionClipPrev.xy - positionClip.xy) * g_Lighting.view.clipToWindowScale;
                        }
                        else
                        {
                            // No previous screen position exists; a zero vector would claim "same pixel, same depth" and
                            // let unrelated history through, so point two screen sizes off-screen instead. The magnitude
                            // stays finite and representable in the RGBA16_FLOAT target
                            motionVector.xy = -2.0f * g_Lighting.view.viewportSize;
                        }

                        u_OutputMotionVectors[launchIndex] = float4(motionVector, 0.0f);
                        u_OutputEmissive[launchIndex] = float4(sampleRadiance, 0.0f);
                        float3 envBRDF = EnvBRDFApprox2(material.specularF0, material.roughness * material.roughness, dot(shadingNormal, ray.Direction));
                        u_OutputDiffuseAlbedo[launchIndex] = float4(material.diffuseAlbedo * (1.0f - envBRDF), isDiffusePath ? 1.0f : 0.0f);
                        u_OutputSpecularAlbedo[launchIndex] = float4(envBRDF, 0.0f);
                    }
                }
            }
#endif // !SHARC_UPDATE && !NRC_UPDATE

            // Run importance sampling of selected BRDF to generate reflecting ray direction
            float3 brdfWeight = float3(0.0f, 0.0f, 0.0f);
            float brdfPdf = 0.0f;
            float refractiveIndex = 1.0f;

            // For a transmissive bounce, derive eta (ratio of the IOR of the medium the path is
            // currently in over the IOR of the medium it's about to enter) from the medium stack,
            // using the hit's front/back-facing state to tell entering a new medium apart from
            // leaving the current one. This correctly handles nested/overlapping transmissive
            // objects, where a single flag to check whether we're dealing with the inside of the glass would get out of sync.
            bool enteringMedium = payload.IsFrontFacing();
            if (brdfType == TRANSMISSIVE_TYPE)
            {
                float currentIor = (mediumStackDepth > 0) ? mediumIorStack[mediumStackDepth - 1] : 1.0f;
                float targetIor = enteringMedium
                    ? g_Global.transmissionIor
                    : ((mediumStackDepth > 1) ? mediumIorStack[mediumStackDepth - 2] : 1.0f);
                refractiveIndex = currentIor / targetIor;
            }

            // Generates a new ray direction
            float2 rand2 = float2(Rand(rngState), Rand(rngState));
            if (!evalIndirectCombinedBRDF(rand2, shadingNormal, geometryNormal, viewVector, material, brdfType, refractiveIndex, ray.Direction, brdfWeight, brdfPdf))
            {
                NrcSetDebugPathTerminationReason(nrcPathState, NrcDebugPathTerminationReason::BRDFAbsorption);
                break; // Ray was eaten by the surface :(
            }

            NrcSetBrdfPdf(nrcPathState, brdfPdf);

            // Refraction requires the ray offset to go in the opposite direction
            bool transition = dot(geometryNormal, ray.Direction) <= 0.0f;
            ray.Origin = OffsetRay(hitPosition, transition ? -geometryNormal : geometryNormal);

            // Only update the medium stack when the sampled direction actually crossed the surface. The transmissive
            // lobe in evalIndirectCombinedBRDF() discards samples whose direction disagrees with the refraction/total
            // internal reflection outcome using this same test, so for a surviving ray, 'transition' is true exactly
            // when it was refracted; the check here is kept as a guard.
            if (brdfType == TRANSMISSIVE_TYPE && transition)
            {
                if (enteringMedium)
                {
                    if (mediumStackDepth < MEDIUM_STACK_DEPTH)
                        mediumIorStack[mediumStackDepth++] = g_Global.transmissionIor;
                }
                else if (mediumStackDepth > 0)
                {
                    mediumStackDepth--;
                }

                //if (g_Global.enableAbsorption)
                //    throughput *= exp(-0.5f * payload.hitDistance); // Beers law of attenuation
            }

            if (g_Global.enableOcclusion)
                throughput *= material.occlusion;

            // Account for surface properties using the BRDF "weight"
            throughput *= brdfWeight;

            SharcSetRadianceDirectionWeight(sharcState, (brdfType == SPECULAR_TYPE) ? saturate(1.0f - material.roughness * material.roughness) : 0.0f);
            SharcSetThroughput(sharcState, throughput);

            if (!isUpdatePass && luminance(throughput) < g_Global.throughputThreshold)
                break;
        }

        NrcWriteFinalPathInfo(nrcContext, nrcPathState, throughput, sampleRadiance);

#if !NRC_UPDATE && !SHARC_UPDATE
        UpdateSampleData(accumulatedSampleData, sampleRadiance, isDiffusePath, hitDistance);

        if (g_Global.debugOutputMode == 9 /* Bounce Heatmap */)
            debugColor = BounceHeatmap(bounce);
#endif // !NRC_UPDATE && !SHARC_UPDATE
    }

#if !NRC_UPDATE && !SHARC_UPDATE
    // Write radiance to output buffer
    ResolveSampleData(accumulatedSampleData, g_Global.samplesPerPixel, 1.0f);

    // Debug output calculation
    if (g_Global.debugOutputMode != 0)
    {
        float2 pixel = float2(DispatchRaysIndex().xy) + 0.5.xx;
        RayDesc ray = GeneratePinholeCameraRay(pixel / float2(launchDimensions), g_Lighting.view.matViewToWorld, g_Lighting.view.matViewToClip);
        RayPayload payload;
        TracePathRay(ray, RAY_FLAG_NONE, payload);

        if (payload.Hit())
        {
            GeometrySample geometry = GetGeometryFromHit(payload.instanceID, payload.primitiveIndex, payload.geometryIndex, payload.barycentrics, GeomAttr_All, t_InstanceData, t_GeometryData, t_MaterialConstants, t_BindlessBuffers);

            if (g_Global.debugOutputMode == 1 /* DiffuseReflectance */)
            {
                debugColor = u_OutputDiffuseAlbedo[launchIndex].xyz;
            }
            else if (g_Global.debugOutputMode == 2 /* SpecularReflectance */)
            {
                debugColor = u_OutputSpecularAlbedo[launchIndex].xyz;
            }
            else if (g_Global.debugOutputMode == 3 /* WorldSpaceNormals */)
            {
                debugColor = u_OutputNormalRoughness[launchIndex].xyz * 0.5f + 0.5f;
            }
            else if (g_Global.debugOutputMode == 4 /* WorldSpacePosition */)
            {
                debugColor = mul(geometry.instance.transform, float4(geometry.objectSpacePosition, 1.0f)).xyz;
            }
            else if (g_Global.debugOutputMode == 5 /* Barycentrics */)
            {
                debugColor = float3(1 - payload.barycentrics.x - payload.barycentrics.y, payload.barycentrics.x, payload.barycentrics.y);
            }
            else if (g_Global.debugOutputMode == 6 /* HitT */)
            {
                debugColor = u_OutputDepth[launchIndex].xxx;
            }
            else if (g_Global.debugOutputMode == 7 /* InstanceID */)
            {
                debugColor = HashAndColor(payload.instanceID);
            }
            else if (g_Global.debugOutputMode == 8 /* Emissives */)
            {
                MaterialSample material = SampleGeometryMaterial(geometry, 0, 0, 0, MatAttr_All, s_MaterialSampler, t_BindlessTextures);
                debugColor = material.emissiveColor;
            }
            else if (g_Global.debugOutputMode == 9 /* Heat map */)
            {
                // Already set
            }
        }

        u_Output[launchIndex] = float4(debugColor, 1.0f);
    }
#endif // !NRC_UPDATE && !SHARC_UPDATE
}

[shader("raygeneration")]
void RayGen()
{
    PathTraceRays();
}