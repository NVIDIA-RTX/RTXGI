/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/render/GBufferFillPass.h>
#include <donut/render/ForwardShadingPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/SceneGraph.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <nvrhi/utils.h>
#include <donut/app/imgui_console.h>
#include <donut/app/imgui_renderer.h>
#include <donut/engine/TextureCache.h>

#include <random>

#include "AppUtils.h"
#include "Pathtracer.h"
#include "ScopedMarker.h"

using namespace donut;
using namespace donut::app;
using namespace donut::engine;
using namespace donut::math;

#if ENABLE_NRC
#include "NrcUtils.h"
#endif // ENABLE_NRC

#include "LightingCb.h"
#include "GlobalCb.h"

#if ENABLE_NRD
#include "NrdConfig.h"
#endif // ENABLE_NRD

// Required for Agility SDK on Windows 10. Setup 1.c. 2.a.
// https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/
extern "C"
{
    __declspec(dllexport) extern const UINT D3D12SDKVersion = 610;
}
extern "C"
{
    __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}

// TODO: Remove when this is addressed in Donut
extern "C"
{
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
extern "C"
{
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

extern const char* g_WindowTitle; //  Defined in Main.cpp

Pathtracer::Pathtracer(DeviceManager* deviceManager, UIData& ui, nvrhi::GraphicsAPI api) : ApplicationBase(deviceManager), m_ui(ui), m_api(api)
{
}

Pathtracer::~Pathtracer()
{
#if ENABLE_NRC
    m_nrc->Shutdown();
#endif // ENABLE_NRC
}

bool Pathtracer::Init(int argc, const char* const* argv)
{
    char* sceneName = nullptr;
    for (int n = 1; n < argc; n++)
    {
        const char* arg = argv[n];

        if (!strcmp(arg, "-noaccumulation"))
            m_ui.denoiserType = DenoiserType::None;
        else if (!strcmp(arg, "-scene"))
            sceneName = (char*)argv[n + 1];
        else if (!strcmp(arg, "-camera"))
            m_pathtracerScene->SetCameraIndex(atoi(argv[n + 1]));
#if ENABLE_NRC
        else if (!strcmp(arg, "-nrc"))
            m_ui.techSelection = TechSelection::Nrc;
#endif // ENABLE_NRC
#if ENABLE_SHARC
        else if (!strcmp(arg, "-sharc"))
            m_ui.techSelection = TechSelection::Sharc;
#endif // ENABLE_SHARC
    }

    m_resetAccumulation = true;
    m_accumulatedFrameCount = 0;

    m_nativeFileSystem = std::make_shared<vfs::NativeFileSystem>();
    std::filesystem::path sceneFileName = app::GetDirectoryWithExecutable().parent_path() / "Assets/Media/bistro.scene.json";
    std::filesystem::path mediaPath = app::GetDirectoryWithExecutable().parent_path() / "Assets/Media";
    std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
    std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/pathtracer" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

    m_rootFileSystem = std::make_shared<vfs::RootFileSystem>();
    m_rootFileSystem->mount("Assets/Media", mediaPath);
    m_rootFileSystem->mount("/shaders/donut", frameworkShaderPath);
    m_rootFileSystem->mount("/shaders/app", appShaderPath);
    m_rootFileSystem->mount("/native", m_nativeFileSystem);

    // Override default scene
    const std::string mediaExt = ".scene.json";
    if (sceneName)
    {
        sceneFileName = app::GetDirectoryWithExecutable().parent_path() / "Assets/Media/";
        sceneFileName += sceneName;

        if (!strstr(sceneName, mediaExt.c_str()))
            sceneFileName += ".scene.json";
    }

#if ENABLE_NRD
    std::filesystem::path nrdShaderPath = app::GetDirectoryWithExecutable() / "shaders/nrd" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
    m_rootFileSystem->mount("/shaders/nrd", nrdShaderPath);
#endif // ENABLE_NRD

    m_shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), m_rootFileSystem, "/shaders");
    m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_shaderFactory);
    m_bindingCache = std::make_unique<engine::BindingCache>(GetDevice());

    m_accelStructManager = std::make_unique<AccelStructManager>();
    m_pathtracerScene = std::make_unique<PathtracerScene>();
    m_pathtracerPipelines = std::make_unique<PathtracerPipelines>();

    m_pathtracerScene->DiscoverScenes(GetLocalPath("Assets/Media"));

#if ENABLE_NRC
    m_nrc = CreateNrcIntegration(m_api);
#endif // ENABLE_NRC

    nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
    bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
    bindlessLayoutDesc.firstSlot = 0;
    bindlessLayoutDesc.maxCapacity = 1024;
    bindlessLayoutDesc.registerSpaces = { nvrhi::BindingLayoutItem::RawBuffer_SRV(1), nvrhi::BindingLayoutItem::Texture_SRV(2) };

    m_bindlessLayout = GetDevice()->createBindlessLayout(bindlessLayoutDesc);
    m_descriptorTable = std::make_shared<engine::DescriptorTableManager>(GetDevice(), m_bindlessLayout);
    m_TextureCache = std::make_shared<engine::TextureCache>(GetDevice(), m_nativeFileSystem, m_descriptorTable);

    SetAsynchronousLoadingEnabled(false);
    SetCurrentSceneName(sceneFileName.string());
    m_pathtracerScene->GetScene()->FinishedLoading(GetFrameIndex());

    m_camera.SetMoveSpeed(3.0f);

    m_constantBuffer =
        GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(LightingConstants), "LightingConstants", engine::c_MaxRenderPassConstantBufferVersions));

    m_debugBuffer =
        GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(GlobalConstants), "GlobalConstants", engine::c_MaxRenderPassConstantBufferVersions));

    // Unified binding
    nvrhi::BindingLayoutDesc bindingLayoutDesc;
    bindingLayoutDesc.visibility = nvrhi::ShaderType::All;
    bindingLayoutDesc.registerSpaceIsDescriptorSet = (m_api == nvrhi::GraphicsAPI::VULKAN);
    for (int i = 0; i < DescriptorSetIDs::COUNT; ++i)
    {
        bindingLayoutDesc.registerSpace = i;
        m_dummyLayouts[i] = GetDevice()->createBindingLayout(bindingLayoutDesc);

        nvrhi::BindingSetDesc dummyBindingDesc;
        m_dummyBindingSets[i] = GetDevice()->createBindingSet(dummyBindingDesc, m_dummyLayouts[i]);
    }

    // Global binding layout
    bindingLayoutDesc.registerSpace = DescriptorSetIDs::Globals;
    bindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(1),
        nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1), // instance
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2), // geometry
        nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3), // materials
        nvrhi::BindingLayoutItem::Sampler(0),
        nvrhi::BindingLayoutItem::Texture_UAV(0),          // path tracer output
    };
    m_globalBindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);

    bindingLayoutDesc.registerSpace = DescriptorSetIDs::Denoiser;
    bindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::Texture_UAV(0), // DiffuseHitDistance
        nvrhi::BindingLayoutItem::Texture_UAV(1), // SpecularHitDistance
        nvrhi::BindingLayoutItem::Texture_UAV(2), // ViewSpaceZ
        nvrhi::BindingLayoutItem::Texture_UAV(3), // NormalRoughness
        nvrhi::BindingLayoutItem::Texture_UAV(4), // MotionVectors
        nvrhi::BindingLayoutItem::Texture_UAV(5), // Emissive
        nvrhi::BindingLayoutItem::Texture_UAV(6), // DiffuseAlbedo
        nvrhi::BindingLayoutItem::Texture_UAV(7), // SpecularAlbedo
    };
    m_denoiserBindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);

#if ENABLE_NRC
    bindingLayoutDesc.registerSpace = DescriptorSetIDs::Nrc;
    bindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0), // QueryPathInfo
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1), // TrainingPathInfo
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2), // PathVertices
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3), // QueryRadianceParams
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(4), // CountersData
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(5), // DebugTrainingPathInfo,
    };
    m_nrcBindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);
#endif                                                     // ENABLE_NRC

#if ENABLE_SHARC
    bindingLayoutDesc.registerSpace = DescriptorSetIDs::Sharc;
    bindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3),
    };
    m_sharcBindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);
#endif // ENABLE_SHARC

    CreateCommonPipelines();

    if (!CreateRayTracingPipelines())
        return false;

        // Prepare resources for the SHARC copy and resolve compute passes
#if ENABLE_SHARC
    {
        m_sharcEntriesNum = 4 * 1024 * 1024;

        // Buffers
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.isConstantBuffer = false;
        bufferDesc.isVolatile = false;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.cpuAccess = nvrhi::CpuAccessMode::None;
        bufferDesc.keepInitialState = true;
        bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;

        bufferDesc.structStride = sizeof(uint64_t);
        bufferDesc.byteSize = m_sharcEntriesNum * bufferDesc.structStride;
        bufferDesc.debugName = "m_sharcHashEntriesBuffer";
        m_sharcHashEntriesBuffer = GetDevice()->createBuffer(bufferDesc);

        bufferDesc.structStride = sizeof(uint32_t);
        bufferDesc.byteSize = m_sharcEntriesNum * bufferDesc.structStride;
        bufferDesc.debugName = "m_sharcLockBuffer";
        m_sharcLockBuffer = GetDevice()->createBuffer(bufferDesc);

        bufferDesc.structStride = 4 * sizeof(uint32_t);
        bufferDesc.byteSize = m_sharcEntriesNum * bufferDesc.structStride;
        bufferDesc.canHaveRawViews = true;

        bufferDesc.debugName = "m_sharcAccumulationBuffer";
        m_sharcAccumulationBuffer = GetDevice()->createBuffer(bufferDesc);

        bufferDesc.debugName = "m_sharcResolvedBuffer";
        m_sharcResolvedBuffer = GetDevice()->createBuffer(bufferDesc);

        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_sharcHashEntriesBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_sharcLockBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_sharcAccumulationBuffer),
            nvrhi::BindingSetItem::StructuredBuffer_UAV(3, m_sharcResolvedBuffer),
        };
        m_sharcBindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_sharcBindingLayout);
    }
#endif // ENABLE_SHARC

#if DONUT_WITH_DLSS
    {
        m_dlss = donut::render::DLSS::Create(GetDevice(), *m_shaderFactory, app::GetDirectoryWithExecutable().generic_string());
        if (m_dlss)
        {
            m_ui.dlssSupported = m_dlss->IsDlssSupported();
            m_ui.dlssRayReconstructionSupported = m_dlss->IsRayReconstructionSupported();
        }
    }
#endif

    // Create the tonemapping pass
    {
        nvrhi::BindingLayoutDesc bindingLayoutDesc;
        bindingLayoutDesc.visibility = nvrhi::ShaderType::Pixel;
        bindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::Texture_UAV(0),
            nvrhi::BindingLayoutItem::Texture_UAV(1),
        };

        m_tonemappingBindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);
    }

    m_commandList = GetDevice()->createCommandList();

    return true;
}

void Pathtracer::CreateCommonPipelines()
{
    m_pathtracerPipelines->CreateCommon(GetDevice(), *m_shaderFactory, m_api, BuildBindingLayouts());
}

bool Pathtracer::CreateRayTracingPipelines()
{
    return m_pathtracerPipelines->CreateRayTracing(GetDevice(), *m_shaderFactory, BuildBindingLayouts(), m_ui.denoiserType == DenoiserType::Nrd);
}

#if ENABLE_NRC
NrcIntegration* Pathtracer::GetNrcInstance() const
{
    return m_nrc.get();
}
#endif

bool Pathtracer::LoadScene(std::shared_ptr<vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName)
{
    return m_pathtracerScene->Load(GetDevice(), *m_shaderFactory, fs, m_TextureCache, m_descriptorTable, sceneFileName);
}

void Pathtracer::SceneLoaded()
{
    ApplicationBase::SceneLoaded();

    bool rebuildAS = false;
    m_pathtracerScene->OnLoaded(GetFrameIndex(), m_ui, m_camera, m_resetAccumulation, rebuildAS, m_accumulatedFrameCount);
    if (rebuildAS)
        m_accelStructManager->RequestRebuild();
}

void Pathtracer::SceneUnloading()
{
    GetDevice()->waitForIdle();

    m_shaderFactory->ClearCache();
    m_bindingCache->Clear();
    m_pathtracerScene->OnUnloading(m_ui);
    m_accelStructManager->RequestRebuild();

    BackBufferResizing();
}

std::vector<std::string> const& Pathtracer::GetAvailableScenes() const
{
    return m_pathtracerScene->GetAvailableScenes();
}

std::string Pathtracer::GetCurrentSceneName() const
{
    return m_pathtracerScene->GetCurrentSceneName();
}

void Pathtracer::SetPreferredSceneName(const std::string& sceneName)
{
    SetCurrentSceneName(app::FindPreferredScene(m_pathtracerScene->GetAvailableScenes(), sceneName));
}

void Pathtracer::SetCurrentSceneName(const std::string& sceneName)
{
    if (m_pathtracerScene->GetCurrentSceneName() == sceneName)
        return;

    m_pathtracerScene->SetCurrentSceneName(sceneName);
    BeginLoadingScene(m_nativeFileSystem, sceneName);

#if ENABLE_NRC
    if (!m_nrc->IsInitialized())
        m_nrc->Initialize(GetDevice());
#endif // ENABLE_NRC
}

void Pathtracer::CopyActiveCameraToFirstPerson()
{
    m_pathtracerScene->CopyActiveCameraToFirstPerson(m_ui, m_camera);
}

bool Pathtracer::KeyboardUpdate(int key, int scancode, int action, int mods)
{
    m_camera.KeyboardUpdate(key, scancode, action, mods);

    if (key == GLFW_KEY_F2 && action == GLFW_PRESS)
        m_ui.showUI = !m_ui.showUI;

    return true;
}

bool Pathtracer::MousePosUpdate(double xpos, double ypos)
{
    m_camera.MousePosUpdate(xpos, ypos);
    return true;
}

bool Pathtracer::MouseButtonUpdate(int button, int action, int mods)
{
    m_camera.MouseButtonUpdate(button, action, mods);
    return true;
}

bool Pathtracer::MouseScrollUpdate(double xoffset, double yoffset)
{
    m_camera.MouseScrollUpdate(xoffset, yoffset);
    return true;
}

void Pathtracer::Animate(float fElapsedTimeSeconds)
{
    m_camera.Animate(fElapsedTimeSeconds);

    if (IsSceneLoaded() && m_ui.enableAnimations)
    {
        m_wallclockTime += fElapsedTimeSeconds;
        float offset = 0;

        for (const auto& anim : m_pathtracerScene->GetScene()->GetSceneGraph()->GetAnimations())
        {
            float duration = anim->GetDuration();
            float integral;
            float animationTime = std::modf((m_wallclockTime + offset) / duration, &integral) * duration;
            (void)anim->Apply(animationTime);
            offset += 1.0f;
        }
    }

    GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
}

void Pathtracer::BackBufferResizing()
{
    m_renderTargets = nullptr;
    m_accumulationBuffer = nullptr;
    m_bindingCache->Clear();
    m_resetAccumulation = true;

    m_denoiserBindingSet = nullptr;
    m_denoiserOutBindingSet = nullptr;

#if ENABLE_NRD
    m_nrd = nullptr;
#endif // ENABLE_NRD
}

void Pathtracer::Render(nvrhi::IFramebuffer* framebuffer)
{
    nvrhi::IDevice* device = GetDevice();
    const auto& fbInfo = framebuffer->getFramebufferInfo();
    uint32_t renderWidth = fbInfo.width;
    uint32_t renderHeight = fbInfo.height;
    uint32_t outputWidth = fbInfo.width;
    uint32_t outputHeight = fbInfo.height;
    const uint32_t frameIndex = GetFrameIndex();
    const bool enableJitter = m_ui.enableJitter && (m_ui.denoiserType != DenoiserType::Nrd || m_ui.dlssMode != DlssMode::None);
    const float2 pixelOffset = enableJitter ? GetPixelOffset(frameIndex) : float2(0.0f, 0.0f);
    float renderScale = 1.0f;
    const bool runReferencePathTracer = m_ui.techSelection == TechSelection::None;
    const bool useDlss = m_ui.dlssSupported && m_ui.dlssMode != DlssMode::None;
    const bool useRayReconstruction = useDlss && m_ui.dlssRayReconstructionSupported && m_ui.denoiserType == DenoiserType::Dlssrr;
    const bool enableDlss = useDlss || useRayReconstruction;

    m_pathtracerScene->GetScene()->RefreshSceneGraph(frameIndex);

#if DONUT_WITH_DLSS
    if (enableDlss)
    {
        renderScale = GetDlssModeScale(m_ui.dlssMode);
        renderWidth = uint(renderWidth * renderScale);
        renderHeight = uint(renderHeight * renderScale);

        donut::render::DLSS::InitParameters params;
        params.inputWidth = renderWidth;
        params.inputHeight = renderHeight;
        params.outputWidth = outputWidth;
        params.outputHeight = outputHeight;
        params.useLinearDepth = true;
        params.useRayReconstruction = useRayReconstruction;

        m_dlss->Init(params);
    }
#endif // DONUT_WITH_DLSS

    if (m_renderScale != renderScale)
    {
        BackBufferResizing();
        m_renderScale = renderScale;
    }

    if (!m_renderTargets)
        m_renderTargets = std::make_unique<RenderTargets>(device, renderWidth, renderHeight);

    if (!m_accumulationBuffer)
    {
        nvrhi::TextureDesc desc;
        desc.width = outputWidth;
        desc.height = outputHeight;
        desc.isUAV = true;
        desc.keepInitialState = true;
        desc.format = nvrhi::Format::RGBA32_FLOAT;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.debugName = "AccumulationBuffer";
        m_accumulationBuffer = device->createTexture(desc);
    }

    m_commandList->open();

    if (m_accelStructManager->NeedsRebuild())
    {
        device->waitForIdle();
        m_accelStructManager->Build(device, m_commandList, *m_pathtracerScene->GetScene(), m_ui.enableTransmission);
    }

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
        nvrhi::BindingSetItem::ConstantBuffer(1, m_debugBuffer),
        nvrhi::BindingSetItem::RayTracingAccelStruct(0, m_accelStructManager->GetTLAS()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_pathtracerScene->GetScene()->GetInstanceBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_pathtracerScene->GetScene()->GetGeometryBuffer()),
        nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_pathtracerScene->GetScene()->GetMaterialBuffer()),
        nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_AnisotropicWrapSampler),
        nvrhi::BindingSetItem::Texture_UAV(0, m_renderTargets->outPathTracer),
    };

    m_globalBindingSet = device->createBindingSet(bindingSetDesc, m_globalBindingLayout);

    // Transition pathTracerOutput
    m_commandList->setTextureState(m_renderTargets->outPathTracer.Get(), nvrhi::TextureSubresourceSet(0, 1, 0, 1), nvrhi::ResourceStates::UnorderedAccess);
    m_commandList->commitBarriers();

    nvrhi::Viewport windowViewport((float)renderWidth, (float)renderHeight);
    m_viewPrevious = m_view;
    m_view.SetViewport(windowViewport);
    m_view.SetMatrices(m_camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(dm::PI_f * 0.25f, windowViewport.width() / windowViewport.height(), 0.1f));
    m_view.SetPixelOffset(pixelOffset);

    if (frameIndex == 0)
        m_viewPrevious = m_view;

    m_view.UpdateCache();
    m_viewPrevious.UpdateCache();

    if (m_viewPrevious.GetViewMatrix() != m_view.GetViewMatrix() || m_ui.enableAnimations)
        m_resetAccumulation = true;

    m_accumulatedFrameCount = m_resetAccumulation ? 1 : m_accumulatedFrameCount + 1;

    m_pathtracerScene->GetScene()->Refresh(m_commandList, frameIndex);
    m_accelStructManager->UpdateTLAS(m_commandList, *m_pathtracerScene->GetScene(), m_ui.enableTransmission);

    LightingConstants constants = {};
    constants.skyColor = float4(m_ui.skyColor * (m_ui.enableSky ? m_ui.skyIntensity : 0.0f), 1.0f);

    // View constants
    m_view.FillPlanarViewConstants(constants.view);
    m_viewPrevious.FillPlanarViewConstants(constants.viewPrev);

    m_pathtracerScene->GetSunLight()->FillLightConstants(constants.sunLight);

    // Add all lights
    constants.lightCount = 0;
    for (auto light : m_pathtracerScene->GetScene()->GetSceneGraph()->GetLights())
    {
        if (constants.lightCount < MAX_LIGHTS)
            light->FillLightConstants(constants.lights[constants.lightCount++]);
    }

#if ENABLE_NRC
    // Update NRC
    if (m_ui.techSelection == TechSelection::Nrc)
    {
        // Check settings that would require NRC to be re-configured
        nrc::ContextSettings nrcContextSettings;
        nrcContextSettings.learnIrradiance = m_ui.nrcLearnIrradiance;
        nrcContextSettings.includeDirectLighting = m_ui.nrcIncludeDirectIllumination;

        nrcContextSettings.frameDimensions = { renderWidth, renderHeight };

        nrcContextSettings.trainingDimensions = nrc::ComputeIdealTrainingDimensions(nrcContextSettings.frameDimensions, 0);
        nrcContextSettings.maxPathVertices = m_ui.nrcMaxTrainingBounces;
        nrcContextSettings.samplesPerPixel = m_ui.samplesPerPixel;

        dm::box3 aabb = m_pathtracerScene->GetScene()->GetSceneGraph()->GetRootNode()->GetGlobalBoundingBox();
        nrcContextSettings.sceneBoundsMin = { aabb.m_mins[0], aabb.m_mins[1], aabb.m_mins[2] };
        nrcContextSettings.sceneBoundsMax = { aabb.m_maxs[0], aabb.m_maxs[1], aabb.m_maxs[2] };
        nrcContextSettings.smallestResolvableFeatureSize = 0.01f;
        // nrcContextSettings.collectAccurateLoss = false;

        if (nrcContextSettings != m_nrcContextSettings)
        {
            // The context settings have changed, so we need to re-configure NRC
            m_nrc->Configure(nrcContextSettings);
            m_nrcContextSettings = nrcContextSettings;

            // Create NVRHI binding set
            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_nrc->m_bufferHandles[nrc::BufferIdx::QueryPathInfo]),
                nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_nrc->m_bufferHandles[nrc::BufferIdx::TrainingPathInfo]),
                nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_nrc->m_bufferHandles[nrc::BufferIdx::TrainingPathVertices]),
                nvrhi::BindingSetItem::StructuredBuffer_UAV(3, m_nrc->m_bufferHandles[nrc::BufferIdx::QueryRadianceParams]),
                nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_nrc->m_bufferHandles[nrc::BufferIdx::Counter]),
                nvrhi::BindingSetItem::StructuredBuffer_UAV(5, m_nrc->m_bufferHandles[nrc::BufferIdx::DebugTrainingPathInfo]),
            };
            m_nrcBindingSet = device->createBindingSet(bindingSetDesc, m_nrcBindingLayout);
        }

        // Settings expected to change frequently that do not require instance reset
        nrc::FrameSettings nrcPerFrameSettings;
        nrcPerFrameSettings.maxExpectedAverageRadianceValue = m_ui.nrcMaxAverageRadiance;
        nrcPerFrameSettings.terminationHeuristicThreshold = m_ui.nrcTerminationHeuristicThreshold;
        nrcPerFrameSettings.trainingTerminationHeuristicThreshold = m_ui.nrcTerminationHeuristicThreshold;
        nrcPerFrameSettings.numTrainingIterations = m_ui.nrcNumTrainingIterations;
        nrcPerFrameSettings.resolveMode = m_ui.nrcResolveMode;
        nrcPerFrameSettings.trainTheCache = m_ui.nrcTrainCache;
        nrcPerFrameSettings.usedTrainingDimensions = nrc::ComputeIdealTrainingDimensions(nrcContextSettings.frameDimensions, nrcPerFrameSettings.numTrainingIterations);
        m_nrcUsedTrainingWidth = nrcPerFrameSettings.usedTrainingDimensions.x;
        m_nrcUsedTrainingHeight = nrcPerFrameSettings.usedTrainingDimensions.y;

        m_nrc->BeginFrame(m_commandList, nrcPerFrameSettings);
        m_nrc->PopulateShaderConstants(constants.nrcConstants);
    }
#endif

    constants.sharcRoughnessThreshold = 0.0f;

    static PlanarViewConstants updatePassView = constants.view;

#if ENABLE_SHARC
    if (m_ui.techSelection == TechSelection::Sharc)
    {
        static float3 sharcCameraPosition = m_view.GetViewOrigin();
        static float3 sharcCameraPositionPrev = m_view.GetViewOrigin();
        if (m_ui.sharcEnableUpdate && m_ui.sharcUpdateViewCamera)
        {
            sharcCameraPositionPrev = sharcCameraPosition;
            sharcCameraPosition = m_view.GetViewOrigin();
        }

        constants.sharcEntriesNum = m_sharcEntriesNum;
        constants.sharcDownscaleFactor = m_ui.sharcDownscaleFactor;
        constants.sharcSceneScale = m_ui.sharcSceneScale;
        constants.sharcRoughnessThreshold = m_ui.sharcRoughnessThreshold;
        constants.sharcCameraPositionPrev.xyz() = sharcCameraPositionPrev;
        constants.sharcCameraPosition.xyz() = sharcCameraPosition;
        constants.sharcAccumulationFrameNum = m_ui.sharcAccumulationFrameNum;
        constants.sharcStaleFrameNum = m_ui.sharcStaleFrameFrameNum;
        constants.sharcEnableAntifirefly = m_ui.sharcEnableAntiFireflyFilter;
    }

    if (m_ui.sharcUpdateViewCamera || m_ui.techSelection != TechSelection::Sharc)
#endif // ENABLE_SHARC
        updatePassView = constants.view;

    constants.updatePassView = updatePassView;

    static bool enableNrd = false;
    bool skipDenoiser = m_ui.ptDebugOutput != PTDebugOutputType::None;

    if (!m_denoiserBindingSet)
    {
        nvrhi::BindingSetDesc bindingSetDesc;

        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::Texture_UAV(0, m_renderTargets->inDiffRadianceHitDist),
            nvrhi::BindingSetItem::Texture_UAV(1, m_renderTargets->inSpecRadianceHitDist),
            nvrhi::BindingSetItem::Texture_UAV(2, m_renderTargets->depth),
            nvrhi::BindingSetItem::Texture_UAV(3, m_renderTargets->normalRoughness),
            nvrhi::BindingSetItem::Texture_UAV(4, m_renderTargets->motionVectors),
            nvrhi::BindingSetItem::Texture_UAV(5, m_renderTargets->emissive),
            nvrhi::BindingSetItem::Texture_UAV(6, m_renderTargets->diffuseAlbedo),
            nvrhi::BindingSetItem::Texture_UAV(7, m_renderTargets->specularAlbedo),
        };

        m_denoiserBindingSet = device->createBindingSet(bindingSetDesc, m_denoiserBindingLayout);

        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::Texture_UAV(0, m_renderTargets->outDiffRadianceHitDist),
            nvrhi::BindingSetItem::Texture_UAV(1, m_renderTargets->outSpecRadianceHitDist),
            nvrhi::BindingSetItem::Texture_UAV(2, m_renderTargets->depth),
            nvrhi::BindingSetItem::Texture_UAV(3, m_renderTargets->normalRoughness),
            nvrhi::BindingSetItem::Texture_UAV(4, m_renderTargets->motionVectors),
            nvrhi::BindingSetItem::Texture_UAV(5, m_renderTargets->emissive),
            nvrhi::BindingSetItem::Texture_UAV(6, m_renderTargets->diffuseAlbedo),
            nvrhi::BindingSetItem::Texture_UAV(7, m_renderTargets->specularAlbedo),
        };

        m_denoiserOutBindingSet = device->createBindingSet(bindingSetDesc, m_denoiserBindingLayout);
    }

    bool resetDenoiser = enableNrd != (m_ui.denoiserType == DenoiserType::Nrd);
    if ((resetDenoiser && !skipDenoiser) || m_ui.reloadShaders)
    {
        if (m_ui.reloadShaders)
        {
            m_shaderFactory->ClearCache();
            CreateCommonPipelines();
            m_ui.reloadShaders = false;
        }
        CreateRayTracingPipelines();
    }

#if ENABLE_NRD
    if (!m_nrd)
    {
        m_nrd = std::make_unique<NrdIntegration>(device, nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR);
        m_nrd->Initialize(renderWidth, renderHeight, *m_shaderFactory);
    }
    enableNrd = (m_ui.denoiserType == DenoiserType::Nrd) && !skipDenoiser;
#endif // ENABLE_NRD

    m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));

    GlobalConstants globalConstants = {};
    globalConstants.enableJitter = m_ui.enableJitter;
    globalConstants.enableBackFaceCull = m_ui.enableBackFaceCull;
    globalConstants.bouncesMax = m_ui.bouncesMax;
    globalConstants.frameIndex = m_frameIndex++;
    globalConstants.enableAccumulation = m_ui.denoiserType == DenoiserType::Accumulation;
    globalConstants.recipAccumulatedFrames = globalConstants.enableAccumulation ? (1.0f / (float)m_accumulatedFrameCount) : 1.0f;
    globalConstants.intensityScale = 1.0f;
    globalConstants.enableEmissives = m_ui.enableEmissives;
    globalConstants.enableLighting = m_ui.enableLighting;
    globalConstants.enableTransmission = m_ui.enableTransmission;
    globalConstants.enableAbsorbtion = m_ui.enableAbsorbtion;
    globalConstants.enableTransparentShadows = m_ui.enableTransparentShadows;
    globalConstants.enableSoftShadows = m_ui.enableSoftShadows;
    globalConstants.throughputThreshold = m_ui.throughputThreshold;
    globalConstants.enableRussianRoulette = m_ui.enableRussianRoulette;
    globalConstants.samplesPerPixel = m_ui.samplesPerPixel;
    globalConstants.exposureScale = donut::math::exp2f(m_ui.exposureAdjustment);
    globalConstants.roughnessMin = m_ui.roughnessMin;
    globalConstants.roughnessMax = std::max(m_ui.roughnessMin, m_ui.roughnessMax);
    globalConstants.metalnessMin = m_ui.metalnessMin;
    globalConstants.metalnessMax = std::max(m_ui.metalnessMin, m_ui.metalnessMax);

    globalConstants.clamp = (uint)m_ui.toneMappingClamp;
    globalConstants.toneMappingOperator = (uint)m_ui.toneMappingOperator;

    globalConstants.targetLight = m_ui.targetLight;
    globalConstants.debugOutputMode = (uint)m_ui.ptDebugOutput;

#if ENABLE_NRC
    globalConstants.nrcSkipDeltaVertices = m_ui.nrcSkipDeltaVertices;
    globalConstants.nrcTerminationHeuristicThreshold = m_ui.nrcTerminationHeuristicThreshold;
#endif // ENABLE_NRC

#if ENABLE_SHARC
    globalConstants.sharcMaterialDemodulation = m_ui.sharcEnableMaterialDemodulation;
    globalConstants.sharcDebug = m_ui.sharcEnableDebug;
#endif // ENABLE_SHARC

#if ENABLE_NRD
    if (enableNrd)
    {
        nrd::HitDistanceParameters hitDistanceParameters;
        globalConstants.nrdHitDistanceParams = (float4&)hitDistanceParameters;
    }
#endif // ENABLE_NRD

    m_commandList->writeBuffer(m_debugBuffer, &globalConstants, sizeof(globalConstants));
    m_commandList->clearState();

    using PT = PathtracerPipelines::Type;

    nvrhi::rt::State state;
    for (int i = 0; i < DescriptorSetIDs::COUNT; ++i)
        state.bindings.push_back(m_dummyBindingSets[i]); // Unified Binding

    state.bindings[DescriptorSetIDs::Globals] = m_globalBindingSet;
    state.bindings[DescriptorSetIDs::Bindless] = m_descriptorTable->GetDescriptorTable();
    state.bindings[DescriptorSetIDs::Denoiser] = m_denoiserBindingSet;

    if (enableDlss || enableNrd)
        m_commandList->clearTextureFloat(m_renderTargets->depth, nvrhi::AllSubresources, nvrhi::Color(0.0f));

    if (enableDlss)
        m_commandList->clearTextureFloat(m_renderTargets->motionVectors, nvrhi::AllSubresources, nvrhi::Color(0.0f));

#if ENABLE_NRC
    if (m_ui.techSelection == TechSelection::Nrc)
    {
        ScopedMarker scopedMarker(m_commandList, "Nrc");

        if (enableNrd)
            m_commandList->clearTextureFloat(m_renderTargets->outPathTracer, nvrhi::AllSubresources, nvrhi::Color(0.0f));

        assert(m_nrcBindingSet);
        state.bindings[DescriptorSetIDs::Nrc] = m_nrcBindingSet;
        {
            ScopedMarker scopedMarker(m_commandList, "NrcQueryPathtracingPass");

            state.shaderTable = m_pathtracerPipelines->GetPermutation(PT::NRC_Query).shaderTable;
            m_commandList->setRayTracingState(state);
            nvrhi::rt::DispatchRaysArguments args;
            args.width = renderWidth;
            args.height = renderHeight;
            m_commandList->dispatchRays(args);
            // There is no dependency between the two
            // raygens, so a barrier is not required here.
            // NVRHI sees that we're using the same UAVs and inserts UAV barriers.
            // Fortunately, we can tell it that we know better.
            m_commandList->setEnableAutomaticBarriers(false);

            // NRC update pathtracing pass
            if (m_ui.nrcTrainCache)
            {
                ScopedMarker scopedMarker(m_commandList, "NrcUpdatePathtracingPass");

                state.shaderTable = m_pathtracerPipelines->GetPermutation(PT::NRC_Update).shaderTable;
                m_commandList->setRayTracingState(state);
                args.width = m_nrcUsedTrainingWidth;
                args.height = m_nrcUsedTrainingHeight;
                m_commandList->dispatchRays(args);
                m_commandList->setEnableAutomaticBarriers(true);
            }
        }

        {
            ScopedMarker scopedMarker(m_commandList, "NrcQueryPropagateTrain");
            m_nrc->QueryAndTrain(m_commandList, m_ui.nrcCalculateTrainingLoss);
        }

        if (m_ui.ptDebugOutput == PTDebugOutputType::None)
        {
            ScopedMarker scopedMarker(m_commandList, "NrcResolve");
            m_nrc->Resolve(m_commandList, m_renderTargets->outPathTracer);
        }
    }

    // Reset heap
    m_commandList->clearState();
#endif // ENABLE_NRC

#if ENABLE_SHARC
    if (m_ui.techSelection == TechSelection::Sharc)
    {
        ScopedMarker scopedMarker(m_commandList, "Sharc");

        assert(m_sharcBindingSet);
        state.bindings[DescriptorSetIDs::Sharc] = m_sharcBindingSet;

        static bool enableMaterialDemodulation = m_ui.sharcEnableMaterialDemodulation;
        if (m_ui.sharcEnableUpdate)
        {
            if (m_ui.sharcEnableClear || m_pathtracerScene->WasReloaded() || (enableMaterialDemodulation != m_ui.sharcEnableMaterialDemodulation))
            {
                enableMaterialDemodulation = m_ui.sharcEnableMaterialDemodulation;

                m_commandList->clearBufferUInt(m_sharcHashEntriesBuffer, m_sharcInvalidEntry);
                m_commandList->clearBufferUInt(m_sharcAccumulationBuffer, 0);
                m_commandList->clearBufferUInt(m_sharcResolvedBuffer, 0);
            }

            // SHARC update
            {
                state.shaderTable = m_pathtracerPipelines->GetPermutation(PT::Sharc_Update).shaderTable;
                m_commandList->setRayTracingState(state);

                nvrhi::rt::DispatchRaysArguments args;
                args.width = renderWidth / m_ui.sharcDownscaleFactor;
                args.height = renderHeight / m_ui.sharcDownscaleFactor;

                ScopedMarker scopedMarker(m_commandList, "SharcUpdate");
                m_commandList->dispatchRays(args);
            }

            if (m_ui.sharcEnableResolve)
            {
                nvrhi::ComputeState computeState;
                // Unified Binding
                if (m_api == nvrhi::GraphicsAPI::D3D12)
                    computeState.bindings = { m_globalBindingSet, m_sharcBindingSet };
                else
                    computeState.bindings = { m_globalBindingSet, m_dummyBindingSets[1], m_dummyBindingSets[2], m_sharcBindingSet };

                // SHARC resolve
                {
                    computeState.pipeline = m_pathtracerPipelines->GetSharcResolvePSO();
                    m_commandList->setComputeState(computeState);

                    const uint groupSize = 256;
                    const dm::uint2 dispatchSize = { DivideRoundUp(m_sharcEntriesNum, groupSize), 1 };

                    ScopedMarker scopedMarker(m_commandList, "SharcResolve");
                    m_commandList->dispatch(dispatchSize.x, dispatchSize.y);
                }
            }
        }

        // SHARC query
        {
            state.shaderTable = m_pathtracerPipelines->GetPermutation(PT::Sharc_Query).shaderTable;
            m_commandList->setRayTracingState(state);

            nvrhi::rt::DispatchRaysArguments args;
            args.width = renderWidth;
            args.height = renderHeight;
            ScopedMarker scopedMarker(m_commandList, "SharcQuery");
            m_commandList->dispatchRays(args);
        }
    }
#endif // ENABLE_SHARC

    if (runReferencePathTracer)
    {
        state.shaderTable = m_pathtracerPipelines->GetPermutation(PT::DefaultPathTracing).shaderTable;
        m_commandList->setRayTracingState(state);

        nvrhi::rt::DispatchRaysArguments args;
        args.width = renderWidth;
        args.height = renderHeight;
        ScopedMarker scopedMarker(m_commandList, "ReferencePathTracer");
        m_commandList->dispatchRays(args);
    }

#if ENABLE_NRD
    if (enableNrd)
    {
        // Denoiser data packing
        {
            nvrhi::ComputeState computeState;
            computeState.bindings = { m_globalBindingSet, m_denoiserBindingSet };
            computeState.pipeline = m_pathtracerPipelines->GetDenoiserReblurPackPSO();
#if ENABLE_NRC
            computeState.pipeline =
                (m_ui.techSelection == TechSelection::Nrc) ? m_pathtracerPipelines->GetDenoiserReblurPack_NRC_PSO() : m_pathtracerPipelines->GetDenoiserReblurPackPSO();
#endif // ENABLE_NRC
            m_commandList->setComputeState(computeState);

            const uint groupSize = 16;
            const dm::uint2 dispatchSize = { DivideRoundUp(renderWidth, groupSize), DivideRoundUp(renderHeight, groupSize) };
            m_commandList->dispatch(dispatchSize.x, dispatchSize.y);
        }

        nrd::ReblurSettings reblurSettings = NrdConfig::GetDefaultREBLURSettings();
        m_nrd->RunDenoiserPasses(m_commandList, *m_renderTargets, 0, m_view, m_viewPrevious, GetFrameIndex(), 0.01f, 0.05f, false, false, &reblurSettings, resetDenoiser);

        // Denoiser resolve
        {
            nvrhi::ComputeState computeState;
            computeState.bindings = { m_globalBindingSet, m_denoiserOutBindingSet };
            computeState.pipeline = m_pathtracerPipelines->GetDenoiserResolvePSO();
            m_commandList->setComputeState(computeState);

            const uint groupSize = 16;
            const dm::uint2 dispatchSize = { DivideRoundUp(renderWidth, groupSize), DivideRoundUp(renderHeight, groupSize) };
            m_commandList->dispatch(dispatchSize.x, dispatchSize.y);
        }
    }
#endif // ENABLE_NRD

#if DONUT_WITH_DLSS
    if (enableDlss)
    {
        donut::render::DLSS::EvaluateParameters params;
        params.depthTexture = m_renderTargets->depth;
        params.motionVectorsTexture = m_renderTargets->motionVectors;
        params.inputColorTexture = m_renderTargets->outPathTracer;
        params.outputColorTexture = m_accumulationBuffer;

        if (useRayReconstruction)
        {
            params.normalRoughness = m_renderTargets->normalRoughness;
            params.diffuseAlbedo = m_renderTargets->diffuseAlbedo;
            params.specularAlbedo = m_renderTargets->specularAlbedo;
        }

        m_dlss->Evaluate(m_commandList, params, m_view);
    }
#endif // DONUT_WITH_DLSS

    // Accumulation and tonemapping
    {
        if (!m_tonemappingPSO)
        {
            nvrhi::GraphicsPipelineDesc pipelineDesc;
            pipelineDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
            pipelineDesc.VS = m_CommonPasses->m_FullscreenVS;
            pipelineDesc.PS = m_pathtracerPipelines->GetTonemappingPS();
            pipelineDesc.bindingLayouts = { m_tonemappingBindingLayout };

            pipelineDesc.renderState.rasterState.setCullNone();
            pipelineDesc.renderState.depthStencilState.depthTestEnable = false;
            pipelineDesc.renderState.depthStencilState.stencilEnable = false;

            m_tonemappingPSO = device->createGraphicsPipeline(pipelineDesc, framebuffer);
        }

        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_debugBuffer),
                                    nvrhi::BindingSetItem::Texture_UAV(0, enableDlss ? m_accumulationBuffer : m_renderTargets->outPathTracer),
                                    nvrhi::BindingSetItem::Texture_UAV(1, m_accumulationBuffer) };

        m_tonemappingBindingSet = device->createBindingSet(bindingSetDesc, m_tonemappingBindingLayout);

        nvrhi::GraphicsState state;
        state.pipeline = m_tonemappingPSO;
        state.framebuffer = framebuffer;
        state.bindings = { m_tonemappingBindingSet };

        nvrhi::Viewport windowViewport((float)outputWidth, (float)outputHeight);
        donut::engine::PlanarView view;
        view.SetViewport(windowViewport);
        state.viewport = view.GetViewportState();

        m_commandList->setGraphicsState(state);

        nvrhi::DrawArguments args;
        args.instanceCount = 1;
        args.vertexCount = 4;
        m_commandList->draw(args);
    }

    m_commandList->close();
    device->executeCommandList(m_commandList);

    m_resetAccumulation = false;
    m_pathtracerScene->ClearReloaded();

#if ENABLE_NRC
    if (m_ui.techSelection == TechSelection::Nrc)
    {
        if (m_api == nvrhi::GraphicsAPI::D3D12)
            m_nrc->EndFrame(device->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Graphics));
        else if (m_api == nvrhi::GraphicsAPI::VULKAN)
            m_nrc->EndFrame(device->getNativeQueue(nvrhi::ObjectTypes::VK_Queue, nvrhi::CommandQueue::Graphics));
    }
#endif // ENABLE_NRC

    GetDeviceManager()->SetVsyncEnabled(m_ui.enableVSync);
}

std::shared_ptr<donut::engine::ShaderFactory> Pathtracer::GetShaderFactory()
{
    return m_shaderFactory;
}

std::shared_ptr<vfs::IFileSystem> Pathtracer::GetRootFS() const
{
    return m_rootFileSystem;
}

std::shared_ptr<TextureCache> Pathtracer::GetTextureCache()
{
    return m_TextureCache;
}

std::shared_ptr<donut::engine::Scene> Pathtracer::GetScene() const
{
    return m_pathtracerScene->GetScene();
}

void Pathtracer::ResetAccumulation()
{
    m_resetAccumulation = true;
}

void Pathtracer::RebuildAccelerationStructure()
{
    m_accelStructManager->RequestRebuild();
}

FirstPersonCamera* Pathtracer::GetCamera()
{
    return &m_camera;
}

std::string Pathtracer::GetResolutionInfo()
{
    if (m_renderTargets->outPathTracer)
    {
        uint2 renderResolution = uint2(m_renderTargets->outPathTracer.Get()->getDesc().width, m_renderTargets->outPathTracer.Get()->getDesc().height);
        uint2 outputResolution = uint2(m_accumulationBuffer->getDesc().width, m_accumulationBuffer->getDesc().height);

        std::string renderResolutionStr = "(" + std::to_string(renderResolution.x) + "x" + std::to_string(renderResolution.y) + ")";
        if (renderResolution.x != outputResolution.x)
            renderResolutionStr += " -> (" + std::to_string(outputResolution.x) + "x" + std::to_string(outputResolution.y) + ")";

        return renderResolutionStr;
    }

    return "uninitialized";
}

PathtracerPipelines::BindingLayouts Pathtracer::BuildBindingLayouts() const
{
    PathtracerPipelines::BindingLayouts layouts;
    layouts.global = m_globalBindingLayout;
    layouts.denoiser = m_denoiserBindingLayout;
    layouts.bindless = m_bindlessLayout;
#if ENABLE_NRC
    layouts.nrc = m_nrcBindingLayout;
#endif
#if ENABLE_SHARC
    layouts.sharc = m_sharcBindingLayout;
#endif
    for (int i = 0; i < DescriptorSetIDs::COUNT; ++i)
        layouts.dummy[i] = m_dummyLayouts[i];
    return layouts;
}