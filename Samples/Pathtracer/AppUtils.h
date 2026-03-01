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

#include <donut/app/ApplicationBase.h>
#include <../nvrhi/src/vulkan/vulkan-backend.h>
#include <donut/core/math/math.h>

#include <random>

static uint32_t DivideRoundUp(uint32_t x, uint32_t divisor)
{
    return (x + divisor - 1) / divisor;
}

inline donut::math::float2 GetPixelOffset(int frameIndex)
{
    donut::math::float2 pixelOffset;

    std::mt19937 rng(frameIndex);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    pixelOffset.x = dist(rng);
    pixelOffset.y = dist(rng);

    return pixelOffset;
}

inline std::filesystem::path GetLocalPath(const std::string subfolder)
{
    static std::filesystem::path oneChoice;

    std::filesystem::path candidateA = donut::app::GetDirectoryWithExecutable() / subfolder;
    std::filesystem::path candidateB = donut::app::GetDirectoryWithExecutable().parent_path() / subfolder;
    if (std::filesystem::exists(candidateA))
        oneChoice = candidateA;
    else
        oneChoice = candidateB;

    return oneChoice;
}

inline void InjectFeatures(VkDeviceCreateInfo& info)
{
    static vk::PhysicalDeviceFeatures2 deviceFeatures;
    vk::PhysicalDeviceVulkan12Features* features12 = (vk::PhysicalDeviceVulkan12Features*)info.pNext;

    assert((VkStructureType)features12->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    {
        features12->shaderBufferInt64Atomics = true;
        features12->shaderSharedInt64Atomics = true;
        features12->scalarBlockLayout = true;
    }

    deviceFeatures.features = *info.pEnabledFeatures;
    deviceFeatures.features.shaderInt64 = true;
    deviceFeatures.features.fragmentStoresAndAtomics = true;

    info.pEnabledFeatures = nullptr;
    deviceFeatures.pNext = (void*)info.pNext;
    info.pNext = &deviceFeatures;
}