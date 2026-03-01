/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/app/ApplicationBase.h>
#include <donut/engine/SceneGraph.h>

#include <algorithm>

#include "AppUtils.h"
#include "PathtracerScene.h"

using namespace donut::app;
using namespace donut::engine;
using namespace donut::math;

bool PathtracerScene::Load(
    nvrhi::IDevice* device,
    donut::engine::ShaderFactory& shaderFactory,
    std::shared_ptr<donut::vfs::IFileSystem> fs,
    std::shared_ptr<donut::engine::TextureCache> textureCache,
    std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable,
    const std::filesystem::path& sceneFileName)
{
    m_scene = std::make_unique<Scene>(device, shaderFactory, fs, textureCache, descriptorTable, nullptr);

    if (m_scene->Load(sceneFileName))
        return true;

    m_scene = nullptr;

    return false;
}

void PathtracerScene::OnLoaded(uint32_t frameIndex, UIData& ui, FirstPersonCamera& camera, bool& resetAccumulation, bool& rebuildAS, uint32_t& accumulatedFrameCount)
{
    m_scene->FinishedLoading(frameIndex);

    m_sceneReloaded = true;
    resetAccumulation = true;
    accumulatedFrameCount = 1;
    rebuildAS = true;

    // Look for an existing sunlight
    for (auto light : m_scene->GetSceneGraph()->GetLights())
    {
        if (light->GetLightType() == LightType_Directional)
        {
            m_sunLight = std::static_pointer_cast<DirectionalLight>(light);
            break;
        }
    }

    auto cameras = m_scene->GetSceneGraph()->GetCameras();
    if (!cameras.empty())
    {
        if (m_cameraIndex != -1 && m_cameraIndex < (int)cameras.size())
        {
            ui.activeSceneCamera = cameras[m_cameraIndex];
            m_cameraIndex = -1;
        }
        else
        {
            std::string cameraName = "DefaultCamera";
            auto it = std::find_if(cameras.begin(), cameras.end(), [&cameraName](std::shared_ptr<SceneCamera> const& cam) { return cam->GetName() == cameraName; });
            ui.activeSceneCamera = (it != cameras.end()) ? *it : cameras[0];
        }

        CopyActiveCameraToFirstPerson(ui, camera);
    }
    else
    {
        ui.activeSceneCamera.reset();
        camera.LookAt(float3(0.f, 1.8f, 0.f), float3(1.f, 1.8f, 0.f));
    }

    // Create a sunlight if there isn't one in the scene
    if (!m_sunLight)
    {
        m_sunLight = std::make_shared<DirectionalLight>();
        auto node = std::make_shared<SceneGraphNode>();
        node->SetLeaf(m_sunLight);
        m_sunLight->SetName("Sun");
        m_scene->GetSceneGraph()->Attach(m_scene->GetSceneGraph()->GetRootNode(), node);
    }

    m_sunLight->angularSize = 0.8f;
    m_sunLight->irradiance = 20.f;
    m_sunLight->SetDirection(dm::double3(-0.049f, -0.87f, 0.48f));
}

void PathtracerScene::OnUnloading(UIData& ui)
{
    m_sunLight = nullptr;
    m_headLight = nullptr;
    ui.selectedMaterial = nullptr;
    ui.activeSceneCamera = nullptr;
    ui.targetLight = -1;
}

void PathtracerScene::DiscoverScenes(const std::filesystem::path& mediaPath)
{
    const std::string mediaExt = ".scene.json";
    for (const auto& file : std::filesystem::directory_iterator(mediaPath))
    {
        if (!file.is_regular_file())
            continue;

        std::string fileName = file.path().filename().string();
        std::string longExt = (fileName.size() <= mediaExt.length()) ? "" : fileName.substr(fileName.length() - mediaExt.length());
        if (longExt == mediaExt)
            m_sceneFilesAvailable.push_back(file.path().string());
    }
}

const std::vector<std::string>& PathtracerScene::GetAvailableScenes() const
{
    return m_sceneFilesAvailable;
}

const std::string& PathtracerScene::GetCurrentSceneName() const
{
    return m_currentSceneName;
}

void PathtracerScene::SetCurrentSceneName(const std::string& sceneName)
{
    m_currentSceneName = sceneName;
}

void PathtracerScene::SetPreferredSceneName(const std::string& sceneName)
{
    m_currentSceneName = FindPreferredScene(m_sceneFilesAvailable, sceneName);
}

void PathtracerScene::CopyActiveCameraToFirstPerson(UIData& ui, FirstPersonCamera& camera)
{
    if (ui.activeSceneCamera)
    {
        dm::affine3 viewToWorld = ui.activeSceneCamera->GetViewToWorldMatrix();
        dm::float3 cameraPos = viewToWorld.m_translation;
        camera.LookAt(cameraPos, cameraPos + viewToWorld.m_linear.row2, viewToWorld.m_linear.row1);
    }
}

std::shared_ptr<donut::engine::Scene> PathtracerScene::GetScene() const
{
    return m_scene;
}

std::shared_ptr<donut::engine::DirectionalLight> PathtracerScene::GetSunLight() const
{
    return m_sunLight;
}

bool PathtracerScene::WasReloaded() const
{
    return m_sceneReloaded;
}

void PathtracerScene::ClearReloaded()
{
    m_sceneReloaded = false;
}

void PathtracerScene::SetCameraIndex(int index)
{
    m_cameraIndex = index;
}