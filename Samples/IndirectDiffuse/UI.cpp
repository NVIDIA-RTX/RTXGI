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

UI::UI(DeviceManager* deviceManager, Pathtracer& app, UIData& ui) : ImGui_Renderer(deviceManager), m_app(app), m_ui(ui)
{
    m_commandList = GetDevice()->createCommandList();

    std::shared_ptr<donut::vfs::NativeFileSystem> nativeFS = std::make_shared<donut::vfs::NativeFileSystem>();
    const std::filesystem::path mediaPath = donut::app::FindDirectory(*nativeFS, GetDirectoryWithExecutable(), "Assets/Media");
    m_defaultFont = this->CreateFontFromFile(*nativeFS, mediaPath / "Fonts/NVIDIASans/NVIDIASans_Rg.ttf", 16.0f);

    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
}

void UI::buildUI(void)
{
    if (!m_ui.showUI)
        return;

    ImFont* font = m_defaultFont->GetScaledFont();
    ImGui::PushFont(font, font ? font->LegacySize : 0.0f);

    if (m_app.IsSceneLoading())
    {
        BeginFullScreenWindow();

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

        EndFullScreenWindow();
        ImGui::PopFont();

        return;
    }

    bool resetRenderHistory = false;
    bool updateAccelerationStructure = false;

    ImGuiIO const& io = ImGui::GetIO();
    float const uiUnit = ImGui::GetFontSize() / 16.0f;
    float const margin = 5.0f * uiUnit;
    float const availableHeight = max(200.0f, io.DisplaySize.y - 2.0f * margin);
    ImGui::SetNextWindowPos(ImVec2(margin, margin), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(0.0f, min(800.0f * uiUnit, availableHeight)), ImGuiCond_Always);

    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysVerticalScrollbar);
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
            resetRenderHistory |= ImGui::Checkbox("Animations", &m_ui.enableAnimations);
            ImGui::SameLine();
            if (ImGui::Button("Reload Shaders"))
            {
                m_ui.reloadShaders = true;
                resetRenderHistory = true;
            }
        }
        ImGui::Indent(-12.0f);
    }

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Path Tracing:", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(12.0f);
        {
            const bool jitterAvailable = m_ui.denoiserType == DenoiserType::Accumulation || m_ui.dlssMode != DlssMode::None;
            bool jitterEnabled = jitterAvailable && m_ui.enableJitter;
            ImGui::BeginDisabled(!jitterAvailable);
            if (ImGui::Checkbox("Jitter", &jitterEnabled))
            {
                m_ui.enableJitter = jitterEnabled;
                resetRenderHistory = true;
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            const bool secondaryPathTracingEnabled = m_ui.indirectDiffuseMode != IndirectDiffuseMode::Disabled;
            ImGui::BeginDisabled(!secondaryPathTracingEnabled);
            resetRenderHistory |= ImGui::Checkbox("Russian Roulette", &m_ui.enableRussianRoulette);
            ImGui::EndDisabled();
            ImGui::SameLine();
            resetRenderHistory |= ImGui::Checkbox("Transmission", &m_ui.enableTransmission);
            ImGui::BeginDisabled(!m_ui.enableTransmission);
            resetRenderHistory |= ImGui::SliderFloat("Transmission IoR", &m_ui.transmissionIor, 1.0f, 3.0f);
            ImGui::EndDisabled();

            // Selects the SER pipelines; the image is unchanged, so the history is kept
            ImGui::BeginDisabled(!m_ui.serSupported);
            ImGui::Checkbox("Shader Execution Reordering", &m_ui.enableSer);
            ImGui::EndDisabled();
            ImGui::SetItemTooltip(
                "Reorders ray tracing threads by hit before shading. Requires a Shader Model 6.9 device and driver on D3D12, or VK_NV_ray_tracing_invocation_reorder on Vulkan.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("Indirect diffuse:");
            ImGui::SameLine();
            if (ImGui::RadioButton("Deferred only (indirect diffuse disabled)", (m_ui.indirectDiffuseMode == IndirectDiffuseMode::Disabled)))
            {
                m_ui.indirectDiffuseMode = IndirectDiffuseMode::Disabled;
                resetRenderHistory = true;
            }
#if ENABLE_NRC
            ImGui::SameLine();
            ImGui::BeginDisabled(!m_app.GetNrcInstance()->IsInitialized());
            if (ImGui::RadioButton("NRC", m_ui.indirectDiffuseMode == IndirectDiffuseMode::Nrc))
            {
                m_ui.indirectDiffuseMode = IndirectDiffuseMode::Nrc;
                resetRenderHistory = true;
            }
            ImGui::EndDisabled();
#else
            const bool normalNrcOutput = false;

#endif // ENABLE_NRC

#if ENABLE_SHARC
            ImGui::SameLine();
            if (ImGui::RadioButton("SHaRC", m_ui.indirectDiffuseMode == IndirectDiffuseMode::Sharc))
            {
                m_ui.indirectDiffuseMode = IndirectDiffuseMode::Sharc;
                resetRenderHistory = true;
            }
#endif // ENABLE_SHARC

            bool hasResolvedIndirectDiffuse = false;

#if ENABLE_NRC
            hasResolvedIndirectDiffuse |= m_ui.indirectDiffuseMode == IndirectDiffuseMode::Nrc && m_ui.nrcResolveMode == NrcResolveMode::AddQueryResultToOutput;
#endif

#if ENABLE_SHARC
            hasResolvedIndirectDiffuse |= m_ui.indirectDiffuseMode == IndirectDiffuseMode::Sharc && !m_ui.sharcEnableDebug;
#endif
            if (!hasResolvedIndirectDiffuse && m_ui.outputViewMode == OutputViewMode::IndirectDiffuse)
            {
                m_ui.outputViewMode = OutputViewMode::Composite;
                resetRenderHistory = true;
            }

            ImGui::BeginDisabled(!hasResolvedIndirectDiffuse);
            if (ImGui::Combo("Output View", reinterpret_cast<int*>(&m_ui.outputViewMode), m_ui.outputViewModeStrings))
            {
                if (m_ui.outputViewMode == OutputViewMode::IndirectDiffuse && m_ui.denoiserType == DenoiserType::Dlssrr)
                {
                    // DLSS-RR operates on the composed scene, not the isolated signal.
                    m_ui.denoiserType = DenoiserType::None;
                }

                resetRenderHistory = true;
            }
            ImGui::SetItemTooltip(
                "Shows the cached indirect diffuse signal traced without the primary diffuse albedo (Fresnel attenuation and occlusion remain), so the lighting is visible independently of surface color.");
            ImGui::EndDisabled();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (m_ui.dlssSupported)
            {
                // DLSS quality modes
                resetRenderHistory |= ImGui::Combo("DLSS Mode", (int*)&m_ui.dlssMode, m_ui.dlssModeStrings);
            }

            ImGui::Text("Denoiser:");
            ImGui::SameLine();
            if (ImGui::RadioButton("None", m_ui.denoiserType == DenoiserType::None))
            {
                m_ui.denoiserType = DenoiserType::None;
                resetRenderHistory = true;
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(m_ui.dlssMode != DlssMode::None);
            if (ImGui::RadioButton("Accumulation", m_ui.denoiserType == DenoiserType::Accumulation))
            {
                m_ui.denoiserType = DenoiserType::Accumulation;
                resetRenderHistory = true;
            }
            ImGui::EndDisabled();
#if ENABLE_NRD
            ImGui::SameLine();
            const bool nrdAvailable = hasResolvedIndirectDiffuse;
            ImGui::BeginDisabled(!nrdAvailable);
            if (ImGui::RadioButton("NRD", m_ui.denoiserType == DenoiserType::Nrd))
            {
                m_ui.denoiserType = DenoiserType::Nrd;
                resetRenderHistory = true;
            }
            ImGui::EndDisabled();
#endif // ENABLE_NRD
            if (m_ui.dlssSupported && m_ui.dlssRayReconstructionSupported)
            {
                const bool dlssRayReconstructionAvailable = hasResolvedIndirectDiffuse && m_ui.outputViewMode == OutputViewMode::Composite && m_ui.dlssMode != DlssMode::None;

                ImGui::BeginDisabled(!dlssRayReconstructionAvailable);
                ImGui::SameLine();
                if (ImGui::RadioButton("DLSS RR", m_ui.denoiserType == DenoiserType::Dlssrr))
                {
                    m_ui.denoiserType = DenoiserType::Dlssrr;
                    resetRenderHistory = true;
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                ImGui::EndDisabled();
            }

            resetRenderHistory |= ImGui::SliderInt("Bounces", &m_ui.bouncesMax, 1, 24);
            resetRenderHistory |= ImGui::SliderInt("Samples Per Pixel", &m_ui.samplesPerPixel, 1, 16);
            resetRenderHistory |= ImGui::SliderFloat("Roughness Min", &m_ui.roughnessMin, 0.0f, 1.0f);
            resetRenderHistory |= ImGui::SliderFloat("Roughness Max", &m_ui.roughnessMax, 0.0f, 1.0f);
            resetRenderHistory |= ImGui::SliderFloat("Metalness Min", &m_ui.metalnessMin, 0.0f, 1.0f);
            resetRenderHistory |= ImGui::SliderFloat("Metalness Max", &m_ui.metalnessMax, 0.0f, 1.0f);
            resetRenderHistory |= updateAccelerationStructure;
        }
        ImGui::Indent(-12.0f);
    }

#if ENABLE_NRC
    ImGui::Separator();
    if (ImGui::CollapsingHeader((m_ui.indirectDiffuseMode == IndirectDiffuseMode::Nrc) ? "NRC:" : "Select NRC PT Mode to see NRC options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(12.0f);
        if (m_ui.indirectDiffuseMode == IndirectDiffuseMode::Nrc)
        {
            resetRenderHistory |= ImGui::Checkbox("Train The Cache", &m_ui.nrcTrainCache);
            resetRenderHistory |= ImGui::SliderInt("Training Bounces", &m_ui.nrcMaxTrainingBounces, 1, 8);
            resetRenderHistory |= ImGui::Checkbox("Learn Irradiance", &m_ui.nrcLearnIrradiance);
            resetRenderHistory |= ImGui::Checkbox("Include Direct Illumination", &m_ui.nrcIncludeDirectIllumination);
            resetRenderHistory |= ImGui::Checkbox("Skip delta vertices", &m_ui.nrcSkipDeltaVertices);
            resetRenderHistory |= ImGui::SliderFloat("Heuristic Threshold", &m_ui.nrcTerminationHeuristicThreshold, 0.0f, 0.25f, "%.3f");
            resetRenderHistory |= ImGui::SliderFloat("Max Average Radiance Value", &m_ui.nrcMaxAverageRadiance, 0.001f, 1000.0f);
            resetRenderHistory |= ImGui::Combo("Resolve Mode", (int*)&m_ui.nrcResolveMode, nrc::GetImGuiResolveModeComboString());
        }
        ImGui::Indent(-12.0f);
    }
#endif // ENABLE_NRC

#if ENABLE_SHARC
    ImGui::Separator();
    if (ImGui::CollapsingHeader((m_ui.indirectDiffuseMode == IndirectDiffuseMode::Sharc) ? "SHaRC:" : "Select SHaRC PT Mode to see SHaRC options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(12.0f);
        if (m_ui.indirectDiffuseMode == IndirectDiffuseMode::Sharc)
        {
            resetRenderHistory |= ImGui::Checkbox("Enable Clear", &m_ui.sharcEnableClear);
            resetRenderHistory |= ImGui::Checkbox("Enable Update", &m_ui.sharcEnableUpdate);
            resetRenderHistory |= ImGui::Checkbox("Enable Resolve", &m_ui.sharcEnableResolve);
            resetRenderHistory |= ImGui::Checkbox("Enable Anti Firefly", &m_ui.sharcEnableAntiFireflyFilter);
            resetRenderHistory |= ImGui::Checkbox("Track SHARC Grid Camera", &m_ui.sharcUpdateViewCamera);
            resetRenderHistory |= ImGui::Checkbox("Enable Material Demodulation", &m_ui.sharcEnableMaterialDemodulation);
            resetRenderHistory |= ImGui::Checkbox("Enable Debug", &m_ui.sharcEnableDebug);
            resetRenderHistory |= ImGui::SliderInt("Accumulation Frame Number", &m_ui.sharcAccumulationFrameNum, 1, 120);
            resetRenderHistory |= ImGui::SliderInt("Stale Frame Number", &m_ui.sharcStaleFrameNum, 1, 120);
            resetRenderHistory |= ImGui::SliderInt("Downscale Factor", &m_ui.sharcDownscaleFactor, 1, 10);
            resetRenderHistory |= ImGui::SliderFloat("Scene Scale", &m_ui.sharcSceneScale, 5.0f, 100.0f);
            resetRenderHistory |= ImGui::SliderFloat("Roughness Threshold", &m_ui.sharcRoughnessThreshold, 0.0f, 1.0f);
        }
        ImGui::Indent(-12.0f);
    }
#endif // ENABLE_SHARC

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Lighting:", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Indent(12.0f);
        resetRenderHistory |= ImGui::Checkbox("Enable Sky", &m_ui.enableSky);
        resetRenderHistory |= ImGui::ColorEdit4("Sky Color", m_ui.skyColor, ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_Float);
        resetRenderHistory |= ImGui::SliderFloat("Sky Intensity", &m_ui.skyIntensity, 0.f, 10.f);
        resetRenderHistory |= ImGui::Checkbox("Enable Emissives", &m_ui.enableEmissives);
        resetRenderHistory |= ImGui::Checkbox("Enable Direct Lighting", &m_ui.enableLighting);
        ImGui::BeginDisabled(!m_ui.enableLighting);
        resetRenderHistory |= ImGui::Checkbox("Enable Primary Shadows", &m_ui.enablePrimaryShadows);
        ImGui::EndDisabled();

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
                    resetRenderHistory |= ImGui::Checkbox("Target this light?", &target);
                    if (target)
                        m_ui.targetLight = m_selectedLightIndex;
                    else if (m_ui.targetLight == m_selectedLightIndex)
                        m_ui.targetLight = -1;
                }
                else
                {
                    ImGui::TextDisabled("Not uploaded to the GPU (only the first %d lights are), cannot be targeted", MAX_LIGHTS);
                }

                resetRenderHistory |= donut::app::LightEditor(*m_selectedLight);
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

            // Tonemapping runs after accumulation and upscaling on the linear radiance, so none of these settings
            // invalidate the history.
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

    if (resetRenderHistory)
    {
        m_app.ResetAccumulation();
        m_app.ResetTemporalHistory();
    }

    if (updateAccelerationStructure)
        m_app.RebuildAccelerationStructure();
};
