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

#include <donut/engine/SceneTypes.h>
#include <donut/engine/SceneGraph.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/imgui_console.h>

#define ENABLE_NRC   1
#define ENABLE_SHARC 1
#define ENABLE_NRD   1

#if ENABLE_NRC
#include "NrcIntegration.h"
#endif // ENABLE_NRC

enum class PTDebugOutputType : uint32_t
{
    None = 0,
    DiffuseReflectance = 1,
    SpecularReflectance = 2,
    WorldSpaceNormals = 3,
    WorldSpacePosition = 4,
    Barycentrics = 5,
    HitT = 6,
    InstanceID = 7,
    Emissives = 8,
    BounceHeatmap = 9,
};

enum class DlssMode : uint32_t
{
    None = 0,
    UltraPerformance = 1,
    Performance = 2,
    Balanced = 3,
    Quality = 4,
    DLAA = 5,
};

constexpr float GetDlssModeScale(DlssMode dlssMode)
{
    switch (dlssMode)
    {
    case DlssMode::UltraPerformance:
        return 1 / 3.0f;
    case DlssMode::Performance:
        return 0.5f;
    case DlssMode::Balanced:
        return 1 / 1.68f;
    case DlssMode::Quality:
        return 1 / 1.5f;
    case DlssMode::DLAA:
        return 1.0f;
    }

    return 1.0f;
}

enum class TechSelection : uint32_t
{
    None = 0,
    Nrc = 1,
    Sharc = 2,
};

enum class DenoiserType : uint32_t
{
    None = 0,
    Accumulation = 1,
    Nrd = 2,
    Dlssrr = 3,
};

enum class ToneMappingOperator : uint32_t
{
    Linear = 0,
    Reinhard = 1,
};

struct UIData
{
    bool showUI = true;
    bool enableVSync = false;
    bool enableAnimations = false;
    bool enableJitter = true;
    bool enableTransmission = false;
    bool enableBackFaceCull = true;
    int bouncesMax = 8;
    int accumulatedFrames = 1;
    int accumulatedFramesMax = 128;
    float exposureAdjustment = 0.0f;
    float roughnessMin = 0.0f;
    float roughnessMax = 1.0f;
    float metalnessMin = 0.0f;
    float metalnessMax = 1.0f;
    bool enableSky = true;
    bool enableEmissives = true;
    bool enableLighting = true;
    bool enableAbsorbtion = true;
    bool enableTransparentShadows = true;
    bool enableSoftShadows = true;
    float throughputThreshold = 0.01f;
    bool enableRussianRoulette = true;
    dm::float3 skyColor = dm::float3(0.5f, 0.75f, 1.0f);
    float skyIntensity = 8.0f;
    int samplesPerPixel = 1;
    int targetLight = 0;
    bool enableTonemapping = true;
    bool reloadShaders = false;

    TechSelection techSelection = TechSelection::None;
    DenoiserType denoiserType = DenoiserType::Accumulation;
    bool enableDenoiser = false;

    ToneMappingOperator toneMappingOperator = ToneMappingOperator::Reinhard;
    bool toneMappingClamp = true;
    const char* toneMappingOperatorStrings = "Linear\0Reinhard\0";

#if ENABLE_NRC
    bool nrcLearnIrradiance = true;
    bool nrcIncludeDirectIllumination = true;
    bool nrcTrainCache = true;
    int nrcMaxTrainingBounces = 8;
    bool nrcCalculateTrainingLoss = false;
    float nrcMaxAverageRadiance = 1.0f;
    NrcResolveMode nrcResolveMode = NrcResolveMode::AddQueryResultToOutput;

    uint32_t nrcTrainingWidth = 0;
    uint32_t nrcTrainingHeight = 0;

    // Settings used by the path tracer
    bool nrcSkipDeltaVertices = false;
    float nrcTerminationHeuristicThreshold = 0.01f;
    int nrcNumTrainingIterations = 1;
#endif // ENABLE_NRC

#if ENABLE_SHARC
    bool sharcEnableClear = false;
    bool sharcEnableUpdate = true;
    bool sharcEnableResolve = true;
    bool sharcEnableAntiFireflyFilter = true;
    bool sharcUpdateViewCamera = true;
    bool sharcEnableMaterialDemodulation = true;
    bool sharcEnableDebug = false;
    int sharcDownscaleFactor = 5;
    float sharcSceneScale = 50.0f;
    int sharcAccumulationFrameNum = 20;
    int sharcStaleFrameFrameNum = 60;
    float sharcRoughnessThreshold = 0.4f;
#endif // ENABLE_SHARC

    bool dlssSupported = false;
    bool dlssRayReconstructionSupported = false;

    std::shared_ptr<donut::engine::Material> selectedMaterial = nullptr;
    std::shared_ptr<donut::engine::SceneCamera> activeSceneCamera = nullptr;

    PTDebugOutputType ptDebugOutput = PTDebugOutputType::None;
    const char* ptDebugOutputTypeStrings =
        "None\0Diffuse Reflectance\0Specular Reflectance\0Worldspace Normals\0Worldspace Position\0Barycentrics\0HitT\0InstanceID\0Emissives\0Bounce Heatmap\0";

    DlssMode dlssMode = DlssMode::None;
    const char* dlssModeStrings = "None\0Ultra Performance\0Performance\0Balanced\0Quality\0DLAA\0";
};

class PathtracerUI : public donut::app::ImGui_Renderer
{
public:
    PathtracerUI(donut::app::DeviceManager* deviceManager, class Pathtracer& app, UIData& ui);
    virtual ~PathtracerUI();

protected:
    virtual void buildUI(void) override;

private:
    class Pathtracer& m_app;
    UIData& m_ui;

    std::shared_ptr<donut::engine::Light> m_selectedLight;
    int m_selectedLightIndex = 0;

    nvrhi::CommandListHandle m_commandList;
};