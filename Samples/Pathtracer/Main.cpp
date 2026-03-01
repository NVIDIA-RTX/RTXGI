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

#include "AppUtils.h"
#include "Pathtracer.h"
#include "PathtracerUi.h"

#if ENABLE_NRC
#include "NrcUtils.h"
#ifdef NRC_WITH_VULKAN
#include <NrcVk.h>
#endif // NRC_WITH_VULKAN
#endif // ENABLE_NRC


using namespace donut;
using namespace donut::app;

const char* g_WindowTitle = "Pathtracer";

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = GetGraphicsAPIFromCommandLine(__argc, __argv);
    DeviceManager* deviceManager = DeviceManager::Create(api);

    bool disableNrc = false;
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
        else if (!strcmp(arg, "-width"))
            deviceParams.backBufferWidth = atoi(__argv[n + 1]);
        else if (!strcmp(arg, "-height"))
            deviceParams.backBufferHeight = atoi(__argv[n + 1]);
    }

#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableNvrhiValidationLayer = true;
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
        if (demo.Init(__argc, __argv))
        {
            PathtracerUI gui(deviceManager, demo, uiData);
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