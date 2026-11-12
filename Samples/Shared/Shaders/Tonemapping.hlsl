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

#include "TonemappingCb.h"
#include "AutoExposureCb.h"

ConstantBuffer<TonemappingConstants> g_Tonemapping              : register(b0);
Buffer<uint>                        adaptedLuminanceBuffer      : register(t0);
RWTexture2D<float4>                 pathTracingOutputBuffer     : register(u0);
RWTexture2D<float4>                 accumulationBuffer          : register(u1);

float CalculateLuminance(float3 color)
{
    // Rec. 709 weights: the renderer works in linear sRGB primaries, and AutoExposure.hlsl meters with the same.
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

float GetExposure()
{
    // The buffer holds the scene luminance that maps to middle grey, either metered by AutoExposure.hlsl or derived
    // from the manual EV on the CPU. Before the first write it reads as zero, which leaves the compensation alone.
    const float adaptedLuminance = asfloat(adaptedLuminanceBuffer[0]);
    float exposure = g_Tonemapping.exposureScale;
    if (adaptedLuminance > 0.0f)
        exposure *= AUTO_EXPOSURE_KEY / adaptedLuminance;
    return exposure;
}

float3 ToneMapLinear(float3 color)
{
    return color;
}

// Extended Reinhard (Reinhard et al. 2002, eq. 4) applied to luminance, so hue and saturation are preserved:
//   Ld = L * (1 + L / Lwhite^2) / (1 + L)
// Lwhite is the smallest luminance that maps to white. 1.0 makes the curve linear (clipped by the display),
// larger values compress more highlights and approach the simple L / (1 + L) curve.
float3 ToneMapReinhard(float3 color)
{
    const float luminance = max(CalculateLuminance(color), 0.0f);
    const float scale = (1.0f + luminance * g_Tonemapping.reinhardRecipWhitePointSquared) / (1.0f + luminance);
    return color * scale;
}

// Based on: https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl 
// ACES filmic (RRT + ODT), targeting the sRGB D65 100-nit output transform.
// Input is scene-referred linear sRGB/
// Output is display-referred linear sRGB, so the sRGB OETF is still applied afterwards. 
// Unlike the single-curve Narkowicz fit (from Falcor), this works in the AP1 space 
// with the RRT/ODT saturation steps folded into the matrices, which keeps
// hue and saturation of bright colors much closer to the reference transform.

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
static const float3x3 c_AcesInputMat = {
    { 0.59719f, 0.35458f, 0.04823f },
    { 0.07600f, 0.90834f, 0.01566f },
    { 0.02840f, 0.13383f, 0.83777f }
};

// ODT_SAT => XYZ => D60_2_D65 => sRGB
static const float3x3 c_AcesOutputMat = {
    {  1.60475f, -0.53108f, -0.07367f },
    { -0.10208f,  1.10813f, -0.00605f },
    { -0.00327f, -0.07276f,  1.07602f }
};

float3 AcesRrtAndOdtFit(float3 color)
{
    const float3 a = color * (color + 0.0245786f) - 0.000090537f;
    const float3 b = color * (0.983729f * color + 0.4329510f) + 0.238081f;
    return a / b;
}

float3 ToneMapAces(float3 color)
{
    color = mul(c_AcesInputMat, color);
    color = AcesRrtAndOdtFit(color);
    color = mul(c_AcesOutputMat, color);
    return saturate(color);
}

// Based on: https://github.com/sobotka/AgX, https://iolite-engine.com/blog_posts/minimal_agx_implementation 
// AgX base look without the optional "punchy"/"golden" grades.
// The sRGB primaries are inset into a wider working space before the per-channel sigmoid, so all three
// channels saturate together and highlights roll off toward white with far less hue skew than ACES.
// Covers ~16.5 stops (c_AgxMinEv..c_AgxMaxEv). The sigmoid uses Benjamin Wrensch's 6th-order polynomial
// fit of the reference curve (mean squared error 3.7e-6). 
// Output is display-referred linear sRGB.
// Note: the matrices below are the reference ones transposed for HLSL row-major mul(M, v); the rows
// sum to 1 so achromatic input stays achromatic.

// sRGB => AgX base (inset)
static const float3x3 c_AgxInsetMat = {
    { 0.842479062253094f,  0.0784335999999992f, 0.0792237451477643f },
    { 0.0423282422610123f, 0.878468636469772f,  0.0791661274605434f },
    { 0.0423756549057051f, 0.0784336f,          0.879142973793104f  }
};

// AgX base => sRGB (outset)
static const float3x3 c_AgxOutsetMat = {
    {  1.19687900512017f,   -0.0980208811401368f, -0.0990297440797205f },
    { -0.0528968517574562f,  1.15190312990417f,   -0.0989611768448433f },
    { -0.0529716355144438f, -0.0980434501171241f,  1.15107367264116f   }
};

static const float c_AgxMinEv = -12.47393f;
static const float c_AgxMaxEv = 4.026069f;

float3 AgxDefaultContrastApprox(float3 x)
{
    const float3 x2 = x * x;
    const float3 x4 = x2 * x2;
    return 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4 - 6.868f * x2 * x + 0.4298f * x2 + 0.1191f * x - 0.00232f;
}

// The "Golden" and "Punchy" looks of the AgX reference config: an ASC CDL grade (slope, power, saturation)
// applied in the AgX base encoding, i.e. after the sigmoid and before the outset matrix.
float3 ApplyAgxLook(float3 color)
{
    if (g_Tonemapping.agxLook == 0 /* Default */)
    {
        return color;
    }

    float3 slope = 1.0f;
    float3 power = 1.0f;
    float saturation = 1.0f;
    if (g_Tonemapping.agxLook == 1 /* Golden */)
    {
        slope = float3(1.0f, 0.9f, 0.5f);
        power = 0.8f;
        saturation = 0.8f;
    }
    else if (g_Tonemapping.agxLook == 2 /* Punchy */)
    {
        power = 1.35f;
        saturation = 1.4f;
    }

    // Per the ASC CDL spec the saturation step uses the luma of the slope/power result, so neutral input stays
    // neutral. Note that the iolite reference implementation takes it from the input instead, which skews greys.
    color = pow(color * slope, power);
    const float luma = CalculateLuminance(color);
    return luma + saturation * (color - luma);
}

float3 ToneMapAgx(float3 color)
{
    color = mul(c_AgxInsetMat, color);

    // Log2 encoding over the supported exposure range; guard against negative/zero input from denoisers.
    color = clamp(log2(max(color, 1e-10f)), c_AgxMinEv, c_AgxMaxEv);
    color = (color - c_AgxMinEv) / (c_AgxMaxEv - c_AgxMinEv);

    // The polynomial fit undershoots zero slightly at the black end; clamp so the look's pow() stays real.
    color = saturate(AgxDefaultContrastApprox(color));

    color = ApplyAgxLook(color);

    color = mul(c_AgxOutsetMat, color);

    // The sigmoid produces a 2.2-gamma display encoding; linearize for the sRGB render target.
    return pow(saturate(color), 2.2f);
}

float3 ToneMap(float3 color)
{
    float3 outColor = color * GetExposure();

    if (g_Tonemapping.toneMappingOperator == 0 /* Linear */)
        outColor = ToneMapLinear(outColor);
    else if (g_Tonemapping.toneMappingOperator == 1 /* Reinhard */)
        outColor = ToneMapReinhard(outColor);
    else if (g_Tonemapping.toneMappingOperator == 2 /* ACES */)
        outColor = ToneMapAces(outColor);
    else if (g_Tonemapping.toneMappingOperator == 3 /* AgX */)
        outColor = ToneMapAgx(outColor);

    return outColor;
}

void main_ps(in float4 pos: SV_Position, in float2 uv: UV, out float4 output: SV_Target)
{
    float4 color = pathTracingOutputBuffer[pos.xy];
    if (g_Tonemapping.enableAccumulation)
    {
        if (g_Tonemapping.recipAccumulatedFrames < 1.0f)
        {
            float4 accumulatedColor = accumulationBuffer[pos.xy];
            color = lerp(accumulatedColor, color, g_Tonemapping.recipAccumulatedFrames);
        }
        accumulationBuffer[pos.xy] = color;
    }

    color.xyz = ToneMap(color.xyz);
	
    if (g_Tonemapping.clamp)
        color = saturate(color);

    output = float4(color);
}