/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


NrcSurfaceAttributes CreateNrcSurfaceAttributes(MaterialSample material, float3 positionWorld, float3 shadingNormal, float3 viewVector)
{
    NrcSurfaceAttributes attributes = (NrcSurfaceAttributes)0;
    attributes.encodedPosition = NrcEncodePosition(positionWorld, g_Lighting.nrcConstants);
    attributes.roughness = material.roughness;
    attributes.specularF0 = material.specularF0;
    attributes.diffuseReflectance = material.diffuseAlbedo;
    attributes.shadingNormal = shadingNormal;
    attributes.viewVector = viewVector;
    // Perfect mirrors and transmissive surfaces are skipped by NRC (neither trained on nor queried) and the path spread is not
    // grown across them, matching the pass-through rule the caches use in the Pathtracer sample. The path continues
    // through the glass (GenerateSecondaryIndirectDiffuseRay samples the transmissive lobe) and the first surface
    // behind it becomes the cache vertex. With the Transmission toggle off, path rays skip transmissive geometry
    // altogether (see AnyHit), so no glass vertex reaches this point
    attributes.isDeltaLobe = ((material.metalness == 1.0f) && (material.roughness == 0.0f)) || (material.transmission >= PASS_THROUGH_TRANSMISSION);

    return attributes;
}

NrcSurfaceAttributes CreateNrcSurfaceAttributes(IndirectDiffuseSecondarySurface surface)
{
    return CreateNrcSurfaceAttributes(surface.material, surface.positionWorld, surface.shadingNormal, surface.viewVector);
}

void TraceNrcIndirectDiffuseRays()
{   
    const uint2 dispatchIndex = DispatchRaysIndex().xy;
    const uint2 dispatchDimensions = DispatchRaysDimensions().xy;

    uint rngState = InitRNG(dispatchIndex, dispatchDimensions, g_Global.frameIndex);

    NrcBuffers nrcBuffers;
    nrcBuffers.queryPathInfo = queryPathInfo;
    nrcBuffers.trainingPathInfo = trainingPathInfo;
    nrcBuffers.trainingPathVertices = trainingPathVertices;
    nrcBuffers.queryRadianceParams = queryRadianceParams;
    nrcBuffers.countersData = countersData;

    NrcContext nrcContext = NrcCreateContext(g_Lighting.nrcConstants, nrcBuffers, dispatchIndex);

    uint2 gBufferPixel = dispatchIndex;
    if (NrcIsUpdateMode())
        gBufferPixel = SelectSparseGBufferPixel(dispatchIndex, g_Lighting.nrcConstants.trainingDimensions, g_Lighting.nrcConstants.frameDimensions, rngState);
    
    const IndirectDiffusePrimarySurface primarySurface = LoadIndirectDiffusePrimarySurface(gBufferPixel); //const NrcPrimarySurface primarySurface = LoadNrcPrimarySurface(gBufferPixel);
    const uint samplesPerPixel = NrcIsUpdateMode() ? 1 : g_Global.samplesPerPixel;
    float3 accumulatedRadiance = 0.0f;
    float accumulatedHitDistance = 0.0f; // for NRD
    
    for (uint sampleIndex = 0; sampleIndex < samplesPerPixel; ++sampleIndex)
    {
        NrcSetSampleIndex(nrcContext, sampleIndex);

        NrcPathState nrcPathState = NrcCreatePathState(g_Lighting.nrcConstants, Rand(rngState));

        float3 throughput = 1.0f;
        float3 sampleRadiance = 0.0f;
        float primaryDiffuseHitDistance = 0.0f; // for NRD
        
        // Record a G-buffer miss
        if (!primarySurface.valid)
        {
            NrcUpdateOnMiss(nrcPathState);
            NrcWriteFinalPathInfo(nrcContext, nrcPathState, throughput, sampleRadiance);
            continue;
        }

        // The G-buffer surface is NRC vertex 0 in both modes, as in the G-buffer variant of the NRC guide, so the
        // termination heuristic sees the primary scattering like in a path tracer that traced the primary ray.
        // Query paths write nothing here; with the heuristic enabled they can't create a query either, because the
        // cumulative spread is still zero, and the cache is queried at the first traced hit at the earliest.
        // Training paths record the G-buffer surface as their first training vertex. NRC resets throughput and
        // radiance at every training vertex, so what is accumulated below belongs to this vertex.
        // The vertex is described by the root surface, whose material is evaluated like that of a traced hit, so the
        // attributes the network sees for a visible surface match those of training paths arriving there.
        const IndirectDiffuseSecondarySurface rootSurface = CreateRootSurface(primarySurface);
        NrcSurfaceAttributes primaryAttributes = CreateNrcSurfaceAttributes(rootSurface);

        // Query paths scatter diffusely at the root whatever its material is (the irradiance-only view even does so on
        // a perfect mirror). A delta flag would make NRC skip the vertex and not grow the path spread over the first
        // traced segment, so the cache could not be queried at the first hit. Describe the event that is sampled:
        // not delta. Training paths sample the real lobes, and the direct cache view (the only query mode without the
        // termination heuristic) shows the cache for the real material, so both keep the material's classification.
        if (NrcIsQueryMode() && g_Lighting.nrcConstants.enableTerminationHeuristic)
            primaryAttributes.isDeltaLobe = false;

        const NrcProgressState primaryProgressState = NrcUpdateOnHit(nrcContext, nrcPathState, primaryAttributes, primarySurface.cameraPathLength, 0, throughput, sampleRadiance);

        // Anything but Continue is only reachable in query mode with the heuristic disabled (DirectCacheView), where
        // the resolve replaces the output anyway, or in update mode with a single training vertex per path.
        bool continuePath = primaryProgressState == NrcProgressState::Continue;

        RayDesc ray = (RayDesc)0;
        float primaryBrdfPdf = 0.0f;
        MediumStack mediumStack = (MediumStack)0;

        if (NrcIsUpdateMode())
        {
            if (primaryProgressState != NrcProgressState::TerminateImmediately)
            {
                // The training target of the root is the full outgoing radiance of the visible surface, the same
                // quantity training paths record when they arrive at this surface from elsewhere. So the root adds
                // its own direct and emissive light (which only reaches the network; the image gets the primary direct
                // lighting from the deferred pass) and samples all of its lobes, not only the diffuse lobe that the
                // query paths use for the indirect diffuse signal.
                const float3 rootRadiance = EvaluateIndirectDirectLighting(rootSurface, rngState) + rootSurface.material.emissiveColor;
                sampleRadiance += rootRadiance * throughput;

                if (continuePath)
                {
                    int rootLobeType = DIFFUSE_TYPE; // Not required by NRC
                    float rootLobeRoughness = 1.0f; // Not required by NRC
                    continuePath = GenerateSecondaryIndirectDiffuseRay(rootSurface, rngState, throughput, mediumStack, ray, rootLobeType, rootLobeRoughness, primaryBrdfPdf);

                    if (!continuePath)
                        NrcSetDebugPathTerminationReason(nrcPathState, NrcDebugPathTerminationReason::BRDFAbsorption);
                }
            }
        }
        else if (continuePath)
        {
            // In the direct cache view a path only continues past the root when NRC skipped it as a delta vertex
            // (skipDeltaVertices); any other root gets the cache query and terminates. The view then shows the cache
            // at what the delta surface reflects, so the path has to follow the material's actual lobes: the diffuse
            // lobe of a perfect mirror has no weight and sampling it would end the path without a query.
            const bool isSkippedDeltaRoot = !g_Lighting.nrcConstants.enableTerminationHeuristic && primaryAttributes.isDeltaLobe;

            if (isSkippedDeltaRoot)
            {
                int rootLobeType = DIFFUSE_TYPE; // Not required by NRC
                float rootLobeRoughness = 1.0f; // Not required by NRC
                continuePath = GenerateSecondaryIndirectDiffuseRay(rootSurface, rngState, throughput, mediumStack, ray, rootLobeType, rootLobeRoughness, primaryBrdfPdf);
            }
            else
            {
                // Sample only the diffuse BRDF.
                // Secondary direct lighting and emissives contribute to indirect diffuse; primary direct lighting does not.
                continuePath = GeneratePrimaryDiffuseRay(primarySurface, rngState, ray, throughput, primaryBrdfPdf);
            }

            if (!continuePath)
                NrcSetDebugPathTerminationReason(nrcPathState, NrcDebugPathTerminationReason::BRDFAbsorption);
        }

        if (!continuePath)
        {
            NrcWriteFinalPathInfo(nrcContext, nrcPathState, throughput, sampleRadiance);
            continue;
        }

        // The heuristic grows the path spread at the next hit from the PDF of this scatter.
        NrcSetBrdfPdf(nrcPathState, primaryBrdfPdf);

        const uint maxBounces = NrcIsUpdateMode() ? g_Lighting.nrcConstants.maxPathVertices : uint(g_Global.bouncesMax);

        for (uint bounce = 1; bounce < maxBounces; ++bounce)
        {
            RayPayload payload = (RayPayload)0;

            TracePathRay(ray, RAY_FLAG_NONE, payload);
            
            // For NRD
            if (bounce == 1)
                primaryDiffuseHitDistance = payload.Hit() ? payload.hitDistance : TRACING_DISTANCE;

            if (!payload.Hit())
            {
                NrcUpdateOnMiss(nrcPathState);
                
                sampleRadiance += g_Lighting.skyColor.rgb * throughput;

                break;
            }
            
            const IndirectDiffuseSecondarySurface secondarySurface = LoadIndirectDiffuseSecondarySurface(ray, payload);
            const NrcSurfaceAttributes surfaceAttributes = CreateNrcSurfaceAttributes(secondarySurface);

            // The G-buffer surface was submitted as vertex 0, so every traced hit reports its segment length.
            const NrcProgressState progressState = NrcUpdateOnHit(nrcContext, nrcPathState, surfaceAttributes, payload.hitDistance, bounce, throughput, sampleRadiance);

            if (progressState == NrcProgressState::TerminateImmediately)
                break;
            
            const float3 localRadiance = EvaluateIndirectDirectLighting(secondarySurface, rngState) + secondarySurface.material.emissiveColor;
            sampleRadiance += localRadiance * throughput;
            
            // With includeDirectLighting=true, NRC normally returns TerminateImmediately when it creates a query because the prediction already contains secondary direct lighting. 
            // With it disabled, NRC returns TerminateAfterDirectLighting, and the shader must explicitly evaluate that light before stopping.
            if (progressState == NrcProgressState::TerminateAfterDirectLighting)
                break;

            if (bounce == maxBounces - 1)
            {
                NrcSetDebugPathTerminationReason(nrcPathState, NrcDebugPathTerminationReason::MaxPathVertices);
                break;
            }
            

            int launchLobeType = DIFFUSE_TYPE; // NRC doesn't use this, but is required for SHaRC in GenerateSecondaryIndirectDiffuseRay
            float launchLobeRoughness = 1.0f; // NRC doesn't use this, but is required for SHaRC in GenerateSecondaryIndirectDiffuseRay
            float brdfPdf = 0.0f;

            if (!GenerateSecondaryIndirectDiffuseRay(secondarySurface, rngState, throughput, mediumStack, ray, launchLobeType, launchLobeRoughness, brdfPdf))
            {
                NrcSetDebugPathTerminationReason(nrcPathState, NrcDebugPathTerminationReason::BRDFAbsorption);
                break;
            }

            // Call this once after BRDF sampling and before tracing the next ray.
            NrcSetBrdfPdf(nrcPathState, brdfPdf);
     
            if (!NrcIsUpdateMode() && luminance(throughput) < g_Global.throughputThreshold)
                break;
        }// end bounce loop

        NrcWriteFinalPathInfo(nrcContext, nrcPathState, throughput, sampleRadiance);

        accumulatedRadiance += sampleRadiance;
        accumulatedHitDistance += primaryDiffuseHitDistance; // for NRD
    }// end samples loop
    
    // The query shader writes the explicitly traced part of the indirect signal.
    // Later, m_nrc->Resolve() call adds the cache prediction using NRC's stored prefix throughput.
    // That prefix contains the primary diffuse throughput because the path throughput carries it when the query is created.
    // In update mode (training), NRC resets throughput at every training vertex, starting with the G-buffer surface, so each vertex is trained on its own outgoing radiance.
    if (NrcIsQueryMode())
    {
        const float inverseSampleCount = rcp(float(samplesPerPixel));
        u_Output[dispatchIndex] = float4(accumulatedRadiance * inverseSampleCount, 1.0f);
#if ENABLE_NRD
    u_OutputDiffuseHitDistance[dispatchIndex] = float4(0.0f, 0.0f, 0.0f, accumulatedHitDistance * inverseSampleCount);
#endif
    }
}

[shader("raygeneration")]
void NrcIndirectDiffuseRayGen()
{
    TraceNrcIndirectDiffuseRays();
}