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

#include <nvrhi/nvrhi.h>

class RenderTargets
{
public:
    RenderTargets(nvrhi::IDevice* device, uint32_t width, uint32_t height);

    nvrhi::TextureHandle depth;
    nvrhi::TextureHandle normalRoughness;
    nvrhi::TextureHandle motionVectors;
    nvrhi::TextureHandle emissive;
    nvrhi::TextureHandle diffuseAlbedo;
    nvrhi::TextureHandle specularAlbedo;

    nvrhi::TextureHandle inDiffRadianceHitDist;
    nvrhi::TextureHandle inSpecRadianceHitDist;

    nvrhi::TextureHandle outDiffRadianceHitDist;
    nvrhi::TextureHandle outSpecRadianceHitDist;

    nvrhi::TextureHandle outPathTracer;
};