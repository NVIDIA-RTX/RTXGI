/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "GeometricNormalGBufferFillPass.h"

#include <donut/engine/ShaderFactory.h>

#include <vector>

nvrhi::ShaderHandle GeometricNormalGBufferFillPass::CreatePixelShader(donut::engine::ShaderFactory& shaderFactory, const CreateParameters& params, bool alphaTested)
{
    std::vector<donut::engine::ShaderMacro> pixelShaderMacros = { donut::engine::ShaderMacro("MOTION_VECTORS", params.enableMotionVectors ? "1" : "0"),
                                                                  donut::engine::ShaderMacro("ALPHA_TESTED", alphaTested ? "1" : "0") };

    return shaderFactory.CreateShader("app/Shaders/GBufferGeometryNormal.hlsl", "main", &pixelShaderMacros, nvrhi::ShaderType::Pixel);
}
