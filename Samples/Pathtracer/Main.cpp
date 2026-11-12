/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/render/DLSS.h>

#include "Pathtracer.h"
#include "UI.h"

#if ENABLE_NRC
#include "Nrc/NrcUtils.h"
#ifdef NRC_WITH_VULKAN
#include <NrcVk.h>
#endif // NRC_WITH_VULKAN
#endif // ENABLE_NRC


using namespace donut;
using namespace donut::app;

const char* g_WindowTitle = "Pathtracer";

namespace
{
void InjectFeatures(VkDeviceCreateInfo& info)
{
    static VkPhysicalDeviceFeatures2 deviceFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    VkPhysicalDeviceVulkan12Features* features12 = (VkPhysicalDeviceVulkan12Features*)info.pNext;

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

#if RTXGI_HAS_SER_SHADERS_SPIRV
    // Shader Execution Reordering: the extension is requested as optional, so it is only in the list when the device
    // supports it, in which case the feature is guaranteed to be available.
    static VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV invocationReorderFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV };
    for (uint32_t i = 0; i < info.enabledExtensionCount; ++i)
    {
        if (!strcmp(info.ppEnabledExtensionNames[i], VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME))
        {
            invocationReorderFeatures.rayTracingInvocationReorder = VK_TRUE;
            invocationReorderFeatures.pNext = deviceFeatures.pNext;
            deviceFeatures.pNext = &invocationReorderFeatures;
            break;
        }
    }
#endif // RTXGI_HAS_SER_SHADERS_SPIRV
}
} // namespace

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = GetGraphicsAPIFromCommandLine(__argc, __argv);
    DeviceManager* deviceManager = DeviceManager::Create(api);

    bool disableNrc = false;
    bool disableDebugRuntime = false;
    DeviceCreationParameters deviceParams;
    deviceParams.enableRayTracingExtensions = true;
    deviceParams.startFullscreen = false;
    deviceParams.backBufferWidth = 1920;
    deviceParams.backBufferHeight = 1080;
    deviceParams.enablePerMonitorDPI = true;

    for (int n = 1; n < __argc; n++)
    {
        const char* arg = __argv[n];

        if (!strcmp(arg, "-fullscreen"))
            deviceParams.startFullscreen = true;
        else if (!strcmp(arg, "-disablenrc"))
            disableNrc = true;
        else if (!strcmp(arg, "-nodebug"))
            disableDebugRuntime = true;
        else if (!strcmp(arg, "-width"))
            deviceParams.backBufferWidth = atoi(__argv[n + 1]);
        else if (!strcmp(arg, "-height"))
            deviceParams.backBufferHeight = atoi(__argv[n + 1]);
    }

#ifdef _DEBUG
    // The debug runtime requires the "Graphics Tools" optional Windows feature; -nodebug allows
    // running a Debug build on machines where it is not installed.
    deviceParams.enableDebugRuntime = !disableDebugRuntime;
    deviceParams.enableNvrhiValidationLayer = !disableDebugRuntime;
#endif

    if (api == nvrhi::GraphicsAPI::VULKAN)
    {
#if ENABLE_NRC && NRC_WITH_VULKAN
        if (!disableNrc)
        {
            char const* const* nrcDeviceExtensions;
            uint32_t numNrcDeviceExtensions = nrc::vulkan::GetVulkanDeviceExtensions(nrcDeviceExtensions);
            for (uint32_t i = 0; i < numNrcDeviceExtensions; ++i)
                deviceParams.requiredVulkanDeviceExtensions.push_back(nrcDeviceExtensions[i]);

            char const* const* nrcInstanceExtensions;
            uint32_t numNrcInstanceExtensions = nrc::vulkan::GetVulkanInstanceExtensions(nrcInstanceExtensions);
            for (uint32_t i = 0; i < numNrcInstanceExtensions; ++i)

                deviceParams.requiredVulkanInstanceExtensions.push_back(nrcInstanceExtensions[i]);

            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME);
        }
#endif // ENABLE_NRC && NRC_WITH_VULKAN

        // Option used by SHARC
        {
            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);
            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
        }

#if RTXGI_HAS_SER_SHADERS_SPIRV
        // Shader Execution Reordering. Optional: the sample checks at runtime whether the device provides it.
        deviceParams.optionalVulkanDeviceExtensions.push_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
#endif // RTXGI_HAS_SER_SHADERS_SPIRV

#if DONUT_WITH_DLSS
        donut::render::DLSS::GetRequiredVulkanExtensions(deviceParams.optionalVulkanInstanceExtensions, deviceParams.optionalVulkanDeviceExtensions);
#endif // DONUT_WITH_DLSS
    }

    deviceParams.deviceCreateInfoCallback = &InjectFeatures;

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::RayTracingPipeline))
    {
        log::fatal("The graphics device does not support Ray Tracing Pipelines");
        return 1;
    }

    {
        UIData uiData;
        Pathtracer demo(deviceManager, uiData, api);
        if (demo.Initialize(__argc, __argv))
        {
            UI gui(deviceManager, demo, uiData);
            gui.Init(demo.GetShaderFactory());

            deviceManager->AddRenderPassToBack(&demo);
            deviceManager->AddRenderPassToBack(&gui);

            deviceManager->RunMessageLoop();

            deviceManager->RemoveRenderPass(&gui);
            deviceManager->RemoveRenderPass(&demo);
        }
    }

    deviceManager->Shutdown();
    delete deviceManager;

    return 0;
}