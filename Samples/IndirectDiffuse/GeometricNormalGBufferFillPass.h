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

#include <donut/render/GBufferFillPass.h>

class GeometricNormalGBufferFillPass : public donut::render::GBufferFillPass
{
public:
    using donut::render::GBufferFillPass::GBufferFillPass;

protected:
    nvrhi::ShaderHandle CreatePixelShader(donut::engine::ShaderFactory& shaderFactory, const CreateParameters& params, bool alphaTested) override;
};
