/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Exposure metering for the tonemapper.
// luminanceHistogram builds a log2 luminance histogram of the scene colour, one 16x16 group per tile, merged into the
// global histogram with atomics. exposureAdaptation reduces it to the geometric mean of the pixels between two
// percentiles, which rejects both the black pixels and the emitters/fireflies, and moves the adapted luminance toward
// that target. The adapted luminance is kept as a float bit pattern in a single R32_UINT element, which is also the
// layout Donut's DLSS integration reads as its exposure buffer.

#include "AutoExposureCb.h"

ConstantBuffer<AutoExposureConstants> g_AutoExposure        : register(b0);
Texture2D<float4>                     sourceColor           : register(t0);
Buffer<uint>                          histogram             : register(t1);
RWBuffer<uint>                        histogramOutput       : register(u0);
RWBuffer<uint>                        adaptedLuminanceOutput: register(u1);

// Pixels are split between the two nearest bins with this many fractional bits so the bin quantization does not bias
// the mean. The scale cancels out in the reduction.
#define FIXED_POINT_FRACTION_BITS 6
static const float c_FixedPointScale = float(1u << FIXED_POINT_FRACTION_BITS);

float Luminance(float3 color)
{
    // Rec. 709 primaries, matching the linear sRGB working space.
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

groupshared uint s_Histogram[AUTO_EXPOSURE_HISTOGRAM_BINS];

[numthreads(AUTO_EXPOSURE_GROUP_SIZE, AUTO_EXPOSURE_GROUP_SIZE, 1)]
void luminanceHistogram(uint groupIndex : SV_GroupIndex, uint2 pixel : SV_DispatchThreadID)
{
    if (groupIndex < AUTO_EXPOSURE_HISTOGRAM_BINS)
        s_Histogram[groupIndex] = 0;

    GroupMemoryBarrierWithGroupSync();

    if (all(pixel < g_AutoExposure.sourceSize))
    {
        // Denoiser and cache outputs can contain NaN or negative values; NaN is skipped, negatives count as black.
        const float luminance = max(Luminance(sourceColor[pixel].rgb), 0.0f);
        if (isfinite(luminance))
        {
            const float normalizedLog = saturate((log2(luminance) - g_AutoExposure.minLogLuminance) * g_AutoExposure.recipLogLuminanceRange);
            const float bin = normalizedLog * (AUTO_EXPOSURE_HISTOGRAM_BINS - 1);
            const uint lowerBin = uint(bin);
            const uint upperWeight = uint(frac(bin) * c_FixedPointScale);
            const uint lowerWeight = uint(c_FixedPointScale) - upperWeight;

            InterlockedAdd(s_Histogram[lowerBin], lowerWeight);
            if (upperWeight != 0) // Implies lowerBin + 1 is still a valid bin
                InterlockedAdd(s_Histogram[lowerBin + 1], upperWeight);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (groupIndex < AUTO_EXPOSURE_HISTOGRAM_BINS)
    {
        const uint count = s_Histogram[groupIndex];
        if (count != 0)
            InterlockedAdd(histogramOutput[groupIndex], count);
    }
}

[numthreads(1, 1, 1)]
void exposureAdaptation()
{
    float pixelCount = 0.0f;
    for (uint bin = 0; bin < AUTO_EXPOSURE_HISTOGRAM_BINS; ++bin)
        pixelCount += float(histogram[bin]);

    // Nothing finite was metered; keep the previous value.
    if (pixelCount <= 0.0f)
        return;

    // Geometric mean of the pixels inside the percentile window. Bins straddling a window edge contribute the
    // fraction of their pixels that falls inside it.
    const float lowCount = pixelCount * g_AutoExposure.lowPercentile;
    const float highCount = pixelCount * g_AutoExposure.highPercentile;
    const float binLogStep = g_AutoExposure.logLuminanceRange / (AUTO_EXPOSURE_HISTOGRAM_BINS - 1);

    float cumulativeCount = 0.0f;
    float weightedLogSum = 0.0f;
    float weightSum = 0.0f;
    for (uint bin = 0; bin < AUTO_EXPOSURE_HISTOGRAM_BINS; ++bin)
    {
        const float binCount = float(histogram[bin]);
        const float weight = max(min(cumulativeCount + binCount, highCount) - max(cumulativeCount, lowCount), 0.0f);
        weightedLogSum += weight * (g_AutoExposure.minLogLuminance + bin * binLogStep);
        weightSum += weight;
        cumulativeCount += binCount;
    }

    if (weightSum <= 0.0f)
        return;

    const float targetLogLuminance = weightedLogSum / weightSum;
    float logLuminance = targetLogLuminance;

    const float previousLuminance = asfloat(adaptedLuminanceOutput[0]);
    const bool hasHistory = !g_AutoExposure.resetAdaptation && isfinite(previousLuminance) && previousLuminance > 0.0f;
    if (hasHistory)
    {
        // Exponential approach in log space so the adaptation rate is constant in EV rather than in luminance.
        const float previousLogLuminance = log2(previousLuminance);
        const float speed = targetLogLuminance > previousLogLuminance ? g_AutoExposure.adaptationSpeedUp : g_AutoExposure.adaptationSpeedDown;
        logLuminance = lerp(targetLogLuminance, previousLogLuminance, exp2(-g_AutoExposure.frameTime * speed));
    }

    adaptedLuminanceOutput[0] = asuint(exp2(logLuminance));
}
