/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <algorithm>

#include <donut/core/math/math.h>
#include <donut/app/ApplicationBase.h>
#include <donut/app/UserInterfaceUtils.h>
#include <donut/engine/Scene.h>
#include <donut/engine/TextureCache.h>

#include "Pathtracer.h"

#if ENABLE_NRC
#include "Nrc/NrcUtils.h"
#include "NrcStructures.h"
#endif

using namespace donut::app;
using namespace donut::engine;
using namespace donut::math;

// Shared with HLSL, so it needs the donut::math types in scope (MAX_LIGHTS)
#include "LightingCb.h"

// Conversion between sRGB to linear color space
// Required here because of a known bug with ImGui and sRGB framebuffer
float ConvertSrgbToLinear(float value)
{
    if (value <= 0.04045f)
        return value / 12.92f;
    else
        return powf((value + 0.055f) / 1.055f, 2.4f);
}

void ConvertSrgbToLinear(ImVec4& color)
{
    color.x = ConvertSrgbToLinear(color.x);
    color.y = ConvertSrgbToLinear(color.y);
    color.z = ConvertSrgbToLinear(color.z);
}

UI::UI(DeviceManager* deviceManager, Pathtracer& app, UIData& ui) : ImGui_Renderer(deviceManager), m_app(app), m_ui(ui)
{
    m_commandList = GetDevice()->createCommandList();

    std::shared_ptr<donut::vfs::NativeFileSystem> nativeFS = std::make_shared<donut::vfs::NativeFileSystem>();
    const std::filesystem::path mediaPath = donut::app::FindDirectory(*nativeFS, GetDirectoryWithExecutable(), "Assets/Media");
    m_defaultFont = this->CreateFontFromFile(*nativeFS, mediaPath / "Fonts/NVIDIASans/NVIDIASans_Rg.ttf", 24.0f);

    ImGui::GetIO().IniFilename = nullptr;
}

void UI::buildUI(void)
{
    if (!m_ui.showUI)
        return;

    if (m_app.IsSceneLoading())
    {
        BeginFullScreenWindow();

        ImGui::PushFont(m_defaultFont->GetScaledFont(), 16.0f);

        char messageBuffer[256];
        const auto& stats = Scene::GetLoadingStats();
        snprintf(
            messageBuffer,
            std::size(messageBuffer),
            "Loading scene %s, please wait...\nObjects: %d/%d, Textures: %d/%d",
            m_app.GetCurrentSceneName().c_str(),
            stats.ObjectsLoaded.load(),
            stats.ObjectsTotal.load(),
            m_app.GetTextureCache()->GetNumberOfLoadedTextures(),
            m_app.GetTextureCache()->GetNumberOfRequestedTextures());

        DrawScreenCenteredText(messageBuffer);

        ImGui::PopFont();

        EndFullScreenWindow();

        return;
    }

    ImGui::PushFont(m_defaultFont->GetScaledFont(), 16.0f);

    bool updateAccum = false;
    bool updateAccelerationStructure = false;

    ImGui::Begin("Settings", 0, ImGuiWindowFlags_None);
    ImGui::SetWindowPos(ImVec2(5.0f, 5.0f));
    ImGui::SetWindowSize(ImVec2(0.0f, 800.0f), ImGuiCond_FirstUseEver);
    ImGui::StyleColorsDark();

    ImGui::Text("%s, %s", GetDeviceManager()->GetRendererString(), m_app.GetResolutionInfo().c_str());
    double frameTime = GetDeviceManager()->GetAverageFrameTimeSeconds();
    if (frameTime > 0.0)
        ImGui::Text("%.3f ms/frame (%.1f FPS)", frameTime * 1e3, 1.0 / frameTime);

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Generic:", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(12.0f);
        {
#ifdef _DEBUG
            float3 cameraPosition = m_app.GetCamera()->GetPosition();
            ImGui::Text("Camera (%0.2f, %0.2f, %0.2f)", cameraPosition.x, cameraPosition.y, cameraPosition.z);
#endif
            const std::string currentScene = m_app.GetCurrentSceneName();
            if (ImGui::BeginCombo("Scene", currentScene.c_str()))
            {
                const std::vector<std::string>& scenes = m_app.GetAvailableScenes();
                for (const std::string& scene : scenes)
                {
                    bool is_selected = scene == currentScene;
                    if (ImGui::Selectable(scene.c_str(), is_selected))
                        m_app.SetCurrentSceneName(scene);

                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            ImGui::Checkbox("VSync", &m_ui.enableVSync);
            ImGui::SameLine();
            ImGui::Checkbox("Animations", &m_ui.enableAnimations);
            ImGui::SameLine();
            if (ImGui::Button("Reload Shaders"))
                m_ui.reloadShaders = true;
        }
        ImGui::Indent(-12.0f);
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Path Tracing:", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(12.0f);
        {
            updateAccum |= ImGui::Checkbox("Jitter", &m_ui.enableJitter);
            ImGui::SameLine();
            updateAccum |= ImGui::Checkbox("Russian Roulette", &m_ui.enableRussianRoulette);
            ImGui::SameLine();
            updateAccum |= ImGui::Checkbox("Transmission", &m_ui.enableTransmission);

            // Selects the SER pipelines; the image is unchanged, so no accumulation reset
            ImGui::BeginDisabled(!m_ui.serSupported);
            ImGui::Checkbox("Shader Execution Reordering", &m_ui.enableSer);
            ImGui::EndDisabled();
            ImGui::SetItemTooltip(
                "Reorders ray tracing threads by hit before shading. Requires a Shader Model 6.9 device and driver on D3D12, or VK_NV_ray_tracing_invocation_reorder on Vulkan.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("Mode:");
            ImGui::SameLine();
            if (ImGui::RadioButton("Reference", (m_ui.techSelection == TechSelection::None)))
            {
                m_ui.techSelection = TechSelection::None;
                updateAccum = true;
            }

#if ENABLE_NRC
            ImGui::SameLine();
            ImGui::BeginDisabled(!m_app.GetNrcInstance()->IsInitialized());
            if (ImGui::RadioButton("NRC", m_ui.techSelection == TechSelection::Nrc))
            {
                m_ui.techSelection = TechSelection::Nrc;
                updateAccum = true;
            }
            ImGui::EndDisabled();
#endif // ENABLE_NRC

#if ENABLE_SHARC
            ImGui::SameLine();
            if (ImGui::RadioButton("SHaRC", m_ui.techSelection == TechSelection::Sharc))
            {
                m_ui.techSelection = TechSelection::Sharc;
                updateAccum = true;
            }
#endif // ENABLE_SHARC

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (m_ui.dlssSupported)
            {
                // DLSS quality modes
                updateAccum |= ImGui::Combo("DLSS Mode", (int*)&m_ui.dlssMode, m_ui.dlssModeStrings);
            }

            ImGui::Text("Denoiser:");
            ImGui::SameLine();
            if (ImGui::RadioButton("None", m_ui.denoiserType == DenoiserType::None))
            {
                m_ui.denoiserType = DenoiserType::None;
                updateAccum = true;
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(m_ui.dlssMode != DlssMode::None);
            if (ImGui::RadioButton("Accumulation", m_ui.denoiserType == DenoiserType::Accumulation))
            {
                m_ui.denoiserType = DenoiserType::Accumulation;
                updateAccum = true;
            }
            ImGui::EndDisabled();
#if ENABLE_NRD
            ImGui::SameLine();
            if (ImGui::RadioButton("NRD", m_ui.denoiserType == DenoiserType::Nrd))
            {
                m_ui.denoiserType = DenoiserType::Nrd;
                updateAccum = true;
            }
#endif // ENABLE_NRD
            if (m_ui.dlssSupported && m_ui.dlssRayReconstructionSupported)
            {
                ImGui::BeginDisabled(m_ui.dlssMode == DlssMode::None);
                ImGui::SameLine();
                if (ImGui::RadioButton("DLSS RR", m_ui.denoiserType == DenoiserType::Dlssrr))
                {
                    m_ui.denoiserType = DenoiserType::Dlssrr;
                    updateAccum = true;
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                ImGui::EndDisabled();
            }

            updateAccum |= ImGui::SliderInt("Bounces", &m_ui.bouncesMax, 1, 24);
            updateAccum |= ImGui::SliderInt("Samples Per Pixel", &m_ui.samplesPerPixel, 1, 16);
            updateAccum |= ImGui::SliderFloat("Roughness Min", &m_ui.roughnessMin, 0.0f, 1.0f);
            updateAccum |= ImGui::SliderFloat("Roughness Max", &m_ui.roughnessMax, 0.0f, 1.0f);
            updateAccum |= ImGui::SliderFloat("Metalness Min", &m_ui.metalnessMin, 0.0f, 1.0f);
            updateAccum |= ImGui::SliderFloat("Metalness Max", &m_ui.metalnessMax, 0.0f, 1.0f);
            updateAccum |= ImGui::SliderFloat("Transmission IoR", &m_ui.transmissionIor, 1.0f, 3.0f);

            // Debug views
            updateAccum |= ImGui::Combo("Debug Output", (int*)&m_ui.ptDebugOutput, m_ui.ptDebugOutputTypeStrings);

            updateAccum |= updateAccelerationStructure;
        }
        ImGui::Indent(-12.0f);
    }

#if ENABLE_NRC
    ImGui::Separator();
    if (ImGui::CollapsingHeader((m_ui.techSelection == TechSelection::Nrc) ? "NRC:" : "Select NRC PT Mode to see NRC options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(12.0f);
        if (m_ui.techSelection == TechSelection::Nrc)
        {
            updateAccum |= ImGui::Checkbox("Train The Cache", &m_ui.nrcTrainCache);
            updateAccum |= ImGui::SliderInt("Training Bounces", &m_ui.nrcMaxTrainingBounces, 1, 8);
            updateAccum |= ImGui::Checkbox("Learn Irradiance", &m_ui.nrcLearnIrradiance);
            updateAccum |= ImGui::Checkbox("Include Direct Illumination", &m_ui.nrcIncludeDirectIllumination);
            updateAccum |= ImGui::Checkbox("Skip delta vertices", &m_ui.nrcSkipDeltaVertices);
            updateAccum |= ImGui::SliderFloat("Heuristic Threshold", &m_ui.nrcTerminationHeuristicThreshold, 0.0f, 0.25f, "%.3f");
            updateAccum |= ImGui::SliderFloat("Max Average Radiance Value", &m_ui.nrcMaxAverageRadiance, 0.001f, 1000.0f);
            updateAccum |= ImGui::Combo("Resolve Mode", (int*)&m_ui.nrcResolveMode, nrc::GetImGuiResolveModeComboString());
        }
        ImGui::Indent(-12.0f);
    }
#endif // ENABLE_NRC

#if ENABLE_SHARC
    ImGui::Separator();
    if (ImGui::CollapsingHeader((m_ui.techSelection == TechSelection::Sharc) ? "SHaRC:" : "Select SHaRC PT Mode to see SHaRC options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(12.0f);
        if (m_ui.techSelection == TechSelection::Sharc)
        {
            updateAccum |= ImGui::Checkbox("Enable Clear", &m_ui.sharcEnableClear);
            updateAccum |= ImGui::Checkbox("Enable Update", &m_ui.sharcEnableUpdate);
            updateAccum |= ImGui::Checkbox("Enable Resolve", &m_ui.sharcEnableResolve);
            updateAccum |= ImGui::Checkbox("Enable Anti Firefly", &m_ui.sharcEnableAntiFireflyFilter);
            updateAccum |= ImGui::Checkbox("Update View Camera", &m_ui.sharcUpdateViewCamera);
            updateAccum |= ImGui::Checkbox("Enable Material Demodulation", &m_ui.sharcEnableMaterialDemodulation);
            updateAccum |= ImGui::Checkbox("Enable Debug", &m_ui.sharcEnableDebug);
            updateAccum |= ImGui::SliderInt("Accumulation Frame Number", &m_ui.sharcAccumulationFrameNum, 1, 120);
            updateAccum |= ImGui::SliderInt("Stale Frame Number", &m_ui.sharcStaleFrameNum, 1, 120);
            updateAccum |= ImGui::SliderInt("Downscale Factor", &m_ui.sharcDownscaleFactor, 1, 10);
            updateAccum |= ImGui::SliderFloat("Scene Scale", &m_ui.sharcSceneScale, 5.0f, 100.0f);
            updateAccum |= ImGui::SliderFloat("Roughness Threshold", &m_ui.sharcRoughnessThreshold, 0.0f, 1.0f);
        }
        ImGui::Indent(-12.0f);
    }
#endif // ENABLE_SHARC

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Lighting:", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(12.0f);
        updateAccum |= ImGui::Checkbox("Enable Sky", &m_ui.enableSky);
        updateAccum |= ImGui::ColorEdit4("Sky Color", m_ui.skyColor, ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_Float);
        updateAccum |= ImGui::SliderFloat("Sky Intensity", &m_ui.skyIntensity, 0.f, 10.f);
        updateAccum |= ImGui::Checkbox("Enable Emissives", &m_ui.enableEmissives);
        updateAccum |= ImGui::Checkbox("Enable Direct Lighting", &m_ui.enableLighting);

        const auto& lights = m_app.GetScene()->GetSceneGraph()->GetLights();

        // The selection outlives scene changes, so make sure it still refers to a light in the current scene and
        // re-derive its index, which is what "Target this light?" writes into targetLight
        if (m_selectedLight)
        {
            const auto it = std::find(lights.begin(), lights.end(), m_selectedLight);
            if (it == lights.end())
            {
                m_selectedLight = nullptr;
                m_selectedLightIndex = 0;
            }
            else
            {
                m_selectedLightIndex = int(std::distance(lights.begin(), it));
            }
        }

        if (!lights.empty() && ImGui::CollapsingHeader("Lights"))
        {
            if (ImGui::BeginCombo("Select Light", m_selectedLight ? m_selectedLight->GetName().c_str() : "(None)"))
            {
                int lightIndex = 0;
                for (const auto& light : lights)
                {
                    bool selected = m_selectedLight == light;
                    ImGui::Selectable(light->GetName().c_str(), &selected);
                    if (selected)
                    {
                        m_selectedLight = light;
                        m_selectedLightIndex = lightIndex;
                        ImGui::SetItemDefaultFocus();
                    }
                    lightIndex++;
                }
                ImGui::EndCombo();
            }

            if (m_selectedLight)
            {
                // Only the first MAX_LIGHTS scene lights are uploaded to the GPU, so only those can be targeted
                if (m_selectedLightIndex < MAX_LIGHTS)
                {
                    bool target = (m_ui.targetLight == m_selectedLightIndex);
                    updateAccum |= ImGui::Checkbox("Target this light?", &target);
                    if (target)
                        m_ui.targetLight = m_selectedLightIndex;
                    else if (m_ui.targetLight == m_selectedLightIndex)
                        m_ui.targetLight = -1;
                }
                else
                {
                    ImGui::TextDisabled("Not uploaded to the GPU (only the first %d lights are), cannot be targeted", MAX_LIGHTS);
                }

                updateAccum |= donut::app::LightEditor(*m_selectedLight);
            }
        }
        ImGui::Indent(-12.0f);
    }

    // Post-process
    {
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Tone mapping:", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Indent(12.0f);

            // Tonemapping runs after accumulation on the linear radiance, so none of these settings invalidate it.
            ImGui::Combo("Operator", (int*)&m_ui.toneMappingOperator, m_ui.toneMappingOperatorStrings);
            if (m_ui.toneMappingOperator == ToneMappingOperator::Reinhard)
            {
                ImGui::SliderFloat("White Point", &m_ui.reinhardWhitePoint, 1.0f, 16.0f);
            }
            else if (m_ui.toneMappingOperator == ToneMappingOperator::Agx)
            {
                ImGui::Combo("Look", (int*)&m_ui.agxLook, m_ui.agxLookStrings);
            }
            ImGui::Checkbox("Clamp", &m_ui.toneMappingClamp);

            if (ImGui::Checkbox("Auto Exposure", &m_ui.autoExposure) && m_ui.autoExposure)
                m_app.ResetAutoExposure();
            ImGui::SameLine();
            ImGui::BeginDisabled(!m_ui.autoExposure);
            ImGui::Checkbox("Lock", &m_ui.autoExposureLock);
            ImGui::SameLine();
            if (ImGui::Button("Reset##AutoExposure"))
                m_app.ResetAutoExposure();
            ImGui::EndDisabled();

            if (m_ui.autoExposure)
            {
                ImGui::SliderFloat("Exposure Compensation (EV)", &m_ui.exposureCompensation, -8.0f, 8.0f);

                if (ImGui::TreeNode("Auto Exposure Settings"))
                {
                    AutoExposurePass::Settings& settings = m_ui.autoExposureSettings;
                    ImGui::SliderFloat("Adaptation Speed Up", &settings.adaptationSpeedUp, 0.0f, 10.0f);
                    ImGui::SliderFloat("Adaptation Speed Down", &settings.adaptationSpeedDown, 0.0f, 10.0f);
                    ImGui::SliderFloat("Low Percentile", &settings.lowPercentile, 0.0f, 1.0f);
                    ImGui::SliderFloat("High Percentile", &settings.highPercentile, 0.0f, 1.0f);
                    ImGui::SliderFloat("Min Luminance (EV)", &settings.minLogLuminance, -16.0f, 16.0f);
                    ImGui::SliderFloat("Max Luminance (EV)", &settings.maxLogLuminance, -16.0f, 16.0f);
                    ImGui::TreePop();
                }
            }
            else
            {
                ImGui::SliderFloat("Exposure (EV)", &m_ui.exposureAdjustment, -8.0f, 8.0f);
            }

            ImGui::Indent(-12.0f);
        }
    }
    ImGui::End();

    ImGui::PopFont();

    if (updateAccum)
        m_app.ResetAccumulation();

    if (updateAccelerationStructure)
        m_app.RebuildAccelerationStructure();
};