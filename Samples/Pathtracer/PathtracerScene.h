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

#include <donut/app/Camera.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/DescriptorTableManager.h>
#include <donut/engine/Scene.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>

#include <filesystem>
#include <string>
#include <vector>

#include "PathtracerUi.h"

class PathtracerScene
{
public:
    // Called from Pathtracer::LoadScene
    bool Load(
        nvrhi::IDevice* device,
        donut::engine::ShaderFactory& shaderFactory,
        std::shared_ptr<donut::vfs::IFileSystem> fs,
        std::shared_ptr<donut::engine::TextureCache> textureCache,
        std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable,
        const std::filesystem::path& sceneFileName);

    // Called from Pathtracer::SceneLoaded
    void OnLoaded(uint32_t frameIndex, UIData& ui, donut::app::FirstPersonCamera& camera, bool& resetAccumulation, bool& rebuildAS, uint32_t& accumulatedFrameCount);

    // Called from Pathtracer::SceneUnloading
    void OnUnloading(UIData& ui);

    // Scene file discovery and selection
    void DiscoverScenes(const std::filesystem::path& mediaPath);

    const std::vector<std::string>& GetAvailableScenes() const;

    const std::string& GetCurrentSceneName() const;

    void SetCurrentSceneName(const std::string& sceneName);

    void SetPreferredSceneName(const std::string& sceneName);

    void CopyActiveCameraToFirstPerson(UIData& ui, donut::app::FirstPersonCamera& camera);

    std::shared_ptr<donut::engine::Scene> GetScene() const;

    std::shared_ptr<donut::engine::DirectionalLight> GetSunLight() const;

    bool WasReloaded() const;

    void ClearReloaded();

    void SetCameraIndex(int index);

private:
    std::shared_ptr<donut::engine::Scene> m_scene;
    std::vector<std::string> m_sceneFilesAvailable;
    std::string m_currentSceneName;
    std::shared_ptr<donut::engine::DirectionalLight> m_sunLight;
    std::shared_ptr<donut::engine::PointLight> m_headLight;
    bool m_sceneReloaded = false;
    int m_cameraIndex = -1;
};