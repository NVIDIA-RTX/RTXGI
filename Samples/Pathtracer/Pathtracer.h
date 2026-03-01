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
#include <donut/app/Camera.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/Scene.h>
#include <donut/engine/View.h>
#include <donut/render/DLSS.h>

#include "PathtracerUi.h"
#include "PathtracerPipelines.h"
#include "AccelStructManager.h"
#include "PathtracerScene.h"
#include "RenderTargets.h"

#if ENABLE_NRD
#include "NrdIntegration.h"
#endif // ENABLE_NRD


class Pathtracer : public donut::app::ApplicationBase
{
public:
    using ApplicationBase::ApplicationBase;

    Pathtracer(donut::app::DeviceManager* deviceManager, UIData& ui, nvrhi::GraphicsAPI api);
    Pathtracer(const Pathtracer&) = delete;
    Pathtracer& operator=(const Pathtracer&) = delete;
    Pathtracer(Pathtracer&&) = delete;
    Pathtracer& operator=(Pathtracer&&) = delete;
    virtual ~Pathtracer();

    bool Init(int argc, const char* const* argv);

    virtual bool LoadScene(std::shared_ptr<donut::vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName) override;
    virtual void SceneUnloading() override;
    virtual void SceneLoaded() override;
    std::vector<std::string> const& GetAvailableScenes() const;
    std::shared_ptr<donut::engine::Scene> GetScene() const;

    std::string GetCurrentSceneName() const;
    void SetPreferredSceneName(const std::string& sceneName);
    void SetCurrentSceneName(const std::string& sceneName);

    void CopyActiveCameraToFirstPerson();

    void Animate(float fElapsedTimeSeconds) override;

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override;

    bool MousePosUpdate(double xpos, double ypos) override;
    bool MouseButtonUpdate(int button, int action, int mods) override;
    bool MouseScrollUpdate(double xoffset, double yoffset) override;

    void CreateCommonPipelines();
    bool CreateRayTracingPipelines();

#if ENABLE_NRC
    NrcIntegration* GetNrcInstance() const;
#endif

    void BackBufferResizing() override;

    void Render(nvrhi::IFramebuffer* framebuffer) override;

    std::shared_ptr<donut::engine::ShaderFactory> GetShaderFactory();
    std::shared_ptr<donut::vfs::IFileSystem> GetRootFS() const;
    std::shared_ptr<donut::engine::TextureCache> GetTextureCache();

    void RebuildAccelerationStructure();
    void ResetAccumulation();

    donut::app::FirstPersonCamera* GetCamera();

    std::string GetResolutionInfo();

private:
    PathtracerPipelines::BindingLayouts BuildBindingLayouts() const;

    std::shared_ptr<donut::vfs::RootFileSystem> m_rootFileSystem;
    std::shared_ptr<donut::vfs::NativeFileSystem> m_nativeFileSystem;

    nvrhi::CommandListHandle m_commandList;
    nvrhi::BindingLayoutHandle m_globalBindingLayout;
    nvrhi::BindingSetHandle m_globalBindingSet;
    nvrhi::BindingLayoutHandle m_bindlessLayout;

    nvrhi::GraphicsPipelineHandle m_tonemappingPSO;
    nvrhi::BindingLayoutHandle m_tonemappingBindingLayout;
    nvrhi::BindingSetHandle m_tonemappingBindingSet;

    nvrhi::BufferHandle m_constantBuffer;
    nvrhi::BufferHandle m_debugBuffer;

    std::shared_ptr<donut::engine::ShaderFactory> m_shaderFactory;
    std::shared_ptr<donut::engine::DescriptorTableManager> m_descriptorTable;

    nvrhi::TextureHandle m_accumulationBuffer;

    donut::app::FirstPersonCamera m_camera;
    donut::engine::PlanarView m_view;
    donut::engine::PlanarView m_viewPrevious;

    std::unique_ptr<donut::engine::BindingCache> m_bindingCache;

    float m_wallclockTime = 0.0f;
    int m_frameIndex = 0;

    UIData& m_ui;

    bool m_resetAccumulation;
    uint32_t m_accumulatedFrameCount;

    nvrhi::GraphicsAPI m_api;

    float m_renderScale = 1.0f;
    std::unique_ptr<RenderTargets> m_renderTargets;

    std::unique_ptr<AccelStructManager> m_accelStructManager;
    std::unique_ptr<PathtracerScene> m_pathtracerScene;
    std::unique_ptr<PathtracerPipelines> m_pathtracerPipelines;

#if ENABLE_NRC
    std::unique_ptr<NrcIntegration> m_nrc;
    nrc::ContextSettings m_nrcContextSettings;
    int m_nrcUsedTrainingWidth = 0;
    int m_nrcUsedTrainingHeight = 0;
    nrc::BuffersAllocationInfo m_nrcBuffersAllocation;
    nvrhi::BindingLayoutHandle m_nrcBindingLayout;
    nvrhi::BindingSetHandle m_nrcBindingSet;
#endif // ENABLE_NRC

#if ENABLE_SHARC
    static const uint32_t m_sharcInvalidEntry = 0;
    uint32_t m_sharcEntriesNum = 0;
    nvrhi::BufferHandle m_sharcHashEntriesBuffer;
    nvrhi::BufferHandle m_sharcLockBuffer;
    nvrhi::BufferHandle m_sharcAccumulationBuffer;
    nvrhi::BufferHandle m_sharcResolvedBuffer;

    nvrhi::BindingLayoutHandle m_sharcBindingLayout;
    nvrhi::BindingSetHandle m_sharcBindingSet;
#endif // ENABLE_SHARC

    nvrhi::BindingLayoutHandle m_denoiserBindingLayout;
    nvrhi::BindingSetHandle m_denoiserBindingSet;
    nvrhi::BindingSetHandle m_denoiserOutBindingSet;

#if ENABLE_NRD
    std::unique_ptr<NrdIntegration> m_nrd;
#endif // ENABLE_NRD

#if DONUT_WITH_DLSS
    std::unique_ptr<donut::render::DLSS> m_dlss;
#endif

    // Unified Binding
    nvrhi::BindingLayoutHandle m_dummyLayouts[DescriptorSetIDs::COUNT];
    nvrhi::BindingSetHandle m_dummyBindingSets[DescriptorSetIDs::COUNT];
};