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

#include <donut/engine/FramebufferFactory.h>

RenderTargets::RenderTargets(nvrhi::IDevice* device, uint32_t width, uint32_t height)
{
    // TODO: verify if there is a better place for the following three variables.
    constexpr dm::uint sampleCount = 1;
    constexpr bool enableMotionVectors = true;
    constexpr bool useReverseProjection = true; // perspProjD3DStyleReverse is in use

    // Creates:gBuffer Depth, Diffuse, Specular, Normals, Emissive, MotionVectors, Framebuffer
    gBuffer.Init(device, dm::uint2(width, height), sampleCount, enableMotionVectors, useReverseProjection);

    // The default G-buffer prefers D24S8. Reconstructing a ray origin from fixed-point
    // reverse-Z depth loses precision rapidly with camera distance, causing self-hits.
    nvrhi::TextureDesc depthDesc = gBuffer.Depth->getDesc();
    depthDesc.format = nvrhi::Format::D32;
    depthDesc.debugName = "GBufferDepth32";
    gBuffer.Depth = device->createTexture(depthDesc);
    gBuffer.GBufferFramebuffer->DepthTarget = gBuffer.Depth;

    {
        nvrhi::TextureDesc geometryNormalDesc;
        geometryNormalDesc.width = width;
        geometryNormalDesc.height = height;
        geometryNormalDesc.sampleCount = sampleCount;
        geometryNormalDesc.format = nvrhi::Format::RGBA32_FLOAT;
        geometryNormalDesc.debugName = "GBufferGeometryNormalAndPlaneDistance";
        geometryNormalDesc.dimension = nvrhi::TextureDimension::Texture2D;
        geometryNormalDesc.initialState = nvrhi::ResourceStates::RenderTarget;
        geometryNormalDesc.keepInitialState = true;
        geometryNormalDesc.isRenderTarget = true;
        geometryNormalDesc.isVirtual = false;
        geometryNormalDesc.isTypeless = false;
        geometryNormalDesc.useClearValue = true;
        geometryNormalDesc.clearValue = nvrhi::Color(0.0f);

        gBufferGeometryNormalAndPlaneDistance = device->createTexture(geometryNormalDesc);
        gBuffer.GBufferFramebuffer->RenderTargets.push_back(gBufferGeometryNormalAndPlaneDistance);
    }

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
    CreateCommonTexture(nvrhi::Format::RGBA32_FLOAT, "reflectedIndirectDiffuse", reflectedIndirectDiffuse);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "compositedColor", compositedColor);
}
