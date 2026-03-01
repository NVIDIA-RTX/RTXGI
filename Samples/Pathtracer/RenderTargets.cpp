/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "RenderTargets.h"

RenderTargets::RenderTargets(nvrhi::IDevice* device, uint32_t width, uint32_t height)
{
    auto CreateCommonTexture = [device, width, height](nvrhi::Format format, const char* debugName, nvrhi::TextureHandle& texture) {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.debugName = debugName;
        desc.isVirtual = false;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.isRenderTarget = false;
        desc.isUAV = true;
        desc.dimension = nvrhi::TextureDimension::Texture2D;
        desc.keepInitialState = true;
        desc.isTypeless = false;

        texture = device->createTexture(desc);
    };

    CreateCommonTexture(nvrhi::Format::R32_FLOAT, "depth", depth);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "motionVectors", motionVectors);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "normalRoughness", normalRoughness);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "emissive", emissive);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "diffuseAlbedo", diffuseAlbedo);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "specularAlbedo", specularAlbedo);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "inDiffRadianceHitDist", inDiffRadianceHitDist);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "inSpecRadianceHitDist", inSpecRadianceHitDist);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "outDiffRadianceHitDist", outDiffRadianceHitDist);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "outSpecRadianceHitDist", outSpecRadianceHitDist);
    CreateCommonTexture(nvrhi::Format::RGBA32_FLOAT, "outPathTracer", outPathTracer);
}