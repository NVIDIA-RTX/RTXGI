/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// TODO: query phase only needs (reads) the hash and the resolved buffers. Potential to optimise?
SharcParameters CreateSharcParameters()
{
    SharcParameters params;
    params.hashGridParameters.cameraPosition = g_Lighting.sharcCameraPosition.xyz;
    params.hashGridParameters.sceneScale = g_Lighting.sharcSceneScale;
    params.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    params.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
    params.hashGridData.capacity = g_Lighting.sharcEntriesNum;
    params.hashGridData.hashEntriesBuffer = u_SharcHashEntriesBuffer;

#if !HASH_GRID_ENABLE_64_BIT_ATOMICS && !HASH_GRID_COMPACT
    params.hashGridData.lockBuffer = u_SharcLockBuffer;
#endif

    params.accumulationBuffer = u_SharcAccumulationBuffer;
    params.resolvedBuffer = u_SharcResolvedBuffer;
    params.radianceScale = SHARC_RADIANCE_SCALE;

    return params;
}

SharcHitData CreateSharcHitData(IndirectDiffuseSecondarySurface surface)
{
    SharcHitData hitData = (SharcHitData) 0;
    hitData.positionWorld = surface.positionWorld;
    hitData.normalWorld = surface.geometryNormal;

#if SHARC_ENABLE_SH_ENCODING
    // Direction from the queried hit toward the preceding path vertex.
    hitData.radianceDirectionWorld = surface.viewVector;
    hitData.radianceDirectionWeight = 0.0f;
#endif

#if SHARC_MATERIAL_DEMODULATION
    hitData.materialDemodulation = 1.0f;

    if (g_Global.sharcMaterialDemodulation)
    {
        const float3 specularFAvg = surface.material.specularF0 + (1.0f - surface.material.specularF0) * 0.047619f; // TODO: const the magic numbers.
        hitData.materialDemodulation = max(surface.material.diffuseAlbedo, 0.05f) + max(surface.material.specularF0, 0.02f) * luminance(specularFAvg);
    }
#endif

#if SHARC_SEPARATE_EMISSIVE
    hitData.emissive = surface.material.emissiveColor;
#endif

    return hitData;
}

// Note:
// Eligibility describes the path segment that was traced, NOT the corresponding hit material.
// g_Lighting.sharcRoughnessThreshold is not used here. That setting modifies update-path material roughness; it must not affect query eligibility.
bool CanQuerySharc(SharcParameters sharcParams, float3 hitPositionWorld, float segmentLength, int launchLobeType, float launchLobeRoughness)
{
    const uint gridLevel = HashGridGetLevel(hitPositionWorld, sharcParams.hashGridParameters);

    const float voxelSize = HashGridGetVoxelSize(gridLevel, sharcParams.hashGridParameters);

    // Be conservative: require the segment to exceed the voxel diagonal.
    if (segmentLength <= voxelSize * sqrt(3.0f))
        return false;

    // The primary is forced diffuse, so the first secondary hit can be queried after the first path segment.
    if (launchLobeType == DIFFUSE_TYPE)
        return true;

    // Do not query through a transmission event.
    if (launchLobeType != SPECULAR_TYPE)
        return false;

    const float roughness = min(launchLobeRoughness, 0.99f); // TODO: const magic number.
    const float alpha = roughness * roughness;
    const float alphaSquared = alpha * alpha;

    const float footprint = 2.0f * segmentLength * sqrt(0.5f * alphaSquared / max(1.0f - alphaSquared, 1e-6f)); // TODO: const magic number.

    return footprint > voxelSize;
}

// FLow:
// Trace path segment -> build secondary surface -> try SHARC -> 
//  if hit: multiply cached radiance by path throughput and stop.
//  if miss: evaluate localdirect/emissive and continue tracing.
void TraceSharcIndirectDiffuseQueryRays()
{
    const uint2 pixelPosition = DispatchRaysIndex().xy;
    const uint2 dispatchDimensions = DispatchRaysDimensions().xy;

    uint rngState = InitRNG(pixelPosition, dispatchDimensions, g_Global.frameIndex);

    const IndirectDiffusePrimarySurface primarySurface = LoadIndirectDiffusePrimarySurface(pixelPosition);

    const uint samplesPerPixel = uint(g_Global.samplesPerPixel);

    float3 accumulatedRadiance = 0.0f;
    float accumulatedHitDistance = 0.0f;

    if (!primarySurface.valid)
    {
        u_Output[pixelPosition] = 0.0f;

#if ENABLE_NRD
        u_OutputDiffuseHitDistance[pixelPosition] = 0.0f;
#endif

        return;
    }

    const SharcParameters sharcParams = CreateSharcParameters();

    for (uint sampleIndex = 0; sampleIndex < samplesPerPixel; ++sampleIndex)
    {
        float3 throughput = 0.0f;
        float3 sampleRadiance = 0.0f;
        float firstIndirectHitDistance = 0.0f;

        RayDesc ray;
        float primaryBrdfPdf = 0.0f; // Not required by SHaRC
        if (!GeneratePrimaryDiffuseRay(primarySurface, rngState, ray, throughput, primaryBrdfPdf))
            continue;

        // Primary from G-buffer to diffuse continuation.
        int launchLobeType = DIFFUSE_TYPE;
        float launchLobeRoughness = 1.0f;

        MediumStack mediumStack = (MediumStack)0;

        const uint maxBounces = uint(g_Global.bouncesMax);
        for (uint bounce = 1; bounce < maxBounces; ++bounce)
        {
            RayPayload payload = (RayPayload) 0;

            TracePathRay(ray, RAY_FLAG_NONE, payload);

            if (bounce == 1)
                firstIndirectHitDistance = payload.Hit() ? payload.hitDistance : TRACING_DISTANCE;

            if (!payload.Hit())
            {
                sampleRadiance += g_Lighting.skyColor.rgb * throughput;
                break;
            }

            const IndirectDiffuseSecondarySurface surface = LoadIndirectDiffuseSecondarySurface(ray, payload);
            
            const SharcHitData sharcHitData = CreateSharcHitData(surface);

            // Debug cache contents at the first traced indirect surface and bypasses deferred composition
            if (g_Global.sharcDebug)
            {
                float3 debugRadiance = 0.0f;
                SharcGetCachedRadiance(sharcParams, sharcHitData, debugRadiance, true);
                sampleRadiance = debugRadiance;
                break;
            }

            // Transmissive surfaces are pass-through for the cache: the glass is neither queried nor (in the update
            // pass) cached, and the path continues to the surfaces behind it. Querying the glass would return whatever
            // shares its voxel, e.g. an emitter right behind a lamp cover
            const bool isCachePassThrough = IsCachePassThrough(surface);

            if (!isCachePassThrough && CanQuerySharc(sharcParams, surface.positionWorld, payload.hitDistance, launchLobeType, launchLobeRoughness))
            {
                float3 cachedRadiance = 0.0f;

                if (SharcGetCachedRadiance(sharcParams, sharcHitData, cachedRadiance, false))
                {
                    // cachedRadiance is outgoing radiance at the secondary surface; throughput includes the primary diffuse event.
                    sampleRadiance += cachedRadiance * throughput;
                    break;
                }
            }

            const float3 localRadiance = EvaluateIndirectDirectLighting(surface, rngState) + surface.material.emissiveColor;
            sampleRadiance += localRadiance * throughput;

            if (bounce == maxBounces - 1)
                break;

            float brdfPdf = 0.0f; // Not required by SHaRC
            int sampledLobeType = DIFFUSE_TYPE;
            float sampledLobeRoughness = 1.0f;
            if (!GenerateSecondaryIndirectDiffuseRay(surface, rngState, throughput, mediumStack, ray, sampledLobeType, sampledLobeRoughness, brdfPdf))
                break;

            // A pass-through vertex does not change how the next hit may be queried: the lobe that launched the path
            // into the glass still describes the spread of the path behind it
            if (!isCachePassThrough)
            {
                launchLobeType = sampledLobeType;
                launchLobeRoughness = sampledLobeRoughness;
            }

            if (luminance(throughput) < g_Global.throughputThreshold)
                break;
        }

        accumulatedRadiance += sampleRadiance;
        accumulatedHitDistance += firstIndirectHitDistance;
    }

    const float inverseSampleCount = rcp(float(samplesPerPixel));

    u_Output[pixelPosition] = float4(accumulatedRadiance * inverseSampleCount, 1.0f);

#if ENABLE_NRD
    // NrdIndirectDiffuse reads only .w here. Radiance stays in u_Output.
    u_OutputDiffuseHitDistance[pixelPosition] = float4(0.0f, 0.0f, 0.0f, accumulatedHitDistance * inverseSampleCount);
#endif
}

#if SHARC_UPDATE
void TraceSharcIndirectDiffuseUpdateRays()
{
    const uint2 dispatchIndex = DispatchRaysIndex().xy;
    const uint2 dispatchDimensions = DispatchRaysDimensions().xy;
    const uint2 frameDimensions = max(uint2(g_Lighting.view.viewportSize), uint2(1u, 1u));

    uint selectionRngState = InitRNG(dispatchIndex, dispatchDimensions, g_Global.frameIndex);

    const uint2 pixelPosition = SelectSparseGBufferPixel(dispatchIndex, dispatchDimensions, frameDimensions, selectionRngState);

    uint rngState = InitRNG(pixelPosition, frameDimensions, g_Global.frameIndex);

    const IndirectDiffusePrimarySurface primarySurface = LoadIndirectDiffusePrimarySurface(pixelPosition);

    // Background pixels do not provide a surface from which to launch an
    // indirect path. Update must not write any display or denoiser output.
    if (!primarySurface.valid)
        return;

    const SharcParameters sharcParams = CreateSharcParameters();

    SharcState sharcState;
    SharcInit(sharcState);

    const uint maxBounces = uint(g_Global.bouncesMax);

    MediumStack mediumStack = (MediumStack)0;

    // The G-buffer surface is the root of the update path (bounce 0) and a cache vertex like every traced hit: it gets
    // an entry holding its own direct and emissive light, demodulated with the G-buffer material, and it receives the
    // radiance propagated back from the vertices after it. No ray is needed to reach it, so every update path adds a
    // sample on a visible surface, and SHaRC's cache resampling, which counts the vertices of the path, can end the
    // path one ray earlier. The Query pass still never reads the cache at the G-buffer surface itself, only at the
    // hits traced from it
    IndirectDiffuseSecondarySurface surface = CreateRootSurface(primarySurface);

    RayDesc ray = (RayDesc)0;

    for (uint bounce = 0; bounce < maxBounces; ++bounce)
    {
        // Bounce 0 is the root; every later bounce traces the segment sampled at the previous vertex
        if (bounce > 0)
        {
            RayPayload payload = (RayPayload)0;

            TracePathRay(ray, RAY_FLAG_NONE, payload);

            if (!payload.Hit())
            {
                SharcUpdateMiss(sharcParams, sharcState, g_Lighting.skyColor.rgb);
                break;
            }

            surface = LoadIndirectDiffuseSecondarySurface(ray, payload);
        }

        // Update-only roughness clamping reduces unstable, highly localized
        // specular signals. Do not use this setting in Query eligibility.
        surface.material.roughness = max(surface.material.roughness, g_Lighting.sharcRoughnessThreshold);

        const SharcHitData sharcHitData = CreateSharcHitData(surface);

        // At the root this is the direct lighting of the visible surface. It only goes into the cache entry; the
        // image gets the primary direct lighting from the deferred pass
        float3 localRadiance = EvaluateIndirectDirectLighting(surface, rngState);

#if !SHARC_SEPARATE_EMISSIVE
        localRadiance += surface.material.emissiveColor;
#endif

        if (IsCachePassThrough(surface))
        {
            // SHaRC has no notion of a delta vertex, so the pass-through is spelled out: no cache entry for the glass
            // itself, but the radiance found at this vertex still belongs to the vertices before it. SharcUpdateMiss()
            // propagates it to them without inserting an entry, and SharcSetThroughput() below keeps the chain intact
            // so that they also receive what is behind the glass
#if SHARC_SEPARATE_EMISSIVE
            localRadiance += surface.material.emissiveColor;
#endif
            SharcUpdateMiss(sharcParams, sharcState, localRadiance);
        }
        else if (!SharcUpdateHit(sharcParams, sharcState, sharcHitData, localRadiance, Rand(rngState)))
        {
            break;
        }

        if (bounce == maxBounces - 1u)
            break;

        // SHARC propagates every segment independently. This is deliberately
        // reset to one rather than carrying primary/path throughput.
        // The root samples all of its lobes like any other vertex, and not only the diffuse lobe that the Query pass
        // uses for the indirect diffuse signal: its entry has to hold the full outgoing radiance of the surface, the
        // same quantity that update paths arriving at this surface from elsewhere store in it
        float3 segmentThroughput = 1.0f;
        int launchLobeType = DIFFUSE_TYPE;
        float launchLobeRoughness = 1.0f;
        
        float brdfPdf = 0.0f; // Not required by SHaRC.
        if (!GenerateSecondaryIndirectDiffuseRay(surface, rngState, segmentThroughput, mediumStack, ray, launchLobeType, launchLobeRoughness, brdfPdf))
            break;

        if (luminance(segmentThroughput) < g_Global.throughputThreshold)
            break;

        if (g_Global.enableRussianRoulette && bounce > BOUNCES_MIN)
        {
            const float continuationProbability = min(0.95f, luminance(segmentThroughput));

            if (continuationProbability <= 0.0f || Rand(rngState) > continuationProbability)
                break;

            segmentThroughput /= continuationProbability;
        }

        const float directionWeight = launchLobeType == SPECULAR_TYPE ? saturate(1.0f - launchLobeRoughness * launchLobeRoughness) : 0.0f;
        SharcSetRadianceDirectionWeight(sharcState, directionWeight);

        SharcSetThroughput(sharcState, segmentThroughput);
    }
}
#endif

#if SHARC_UPDATE && SHARC_QUERY
#error SHARC Update and Query must be separate permutations.
#endif

#if !SHARC_UPDATE && !SHARC_QUERY
#error SHARC indirect-diffuse ray generation requires Update or Query.
#endif

[shader("raygeneration")]
void SharcIndirectDiffuseRayGen()
{
#if SHARC_UPDATE
    TraceSharcIndirectDiffuseUpdateRays();
#else
    TraceSharcIndirectDiffuseQueryRays();
#endif
}

