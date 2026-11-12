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

#include "Pathtracer.h"
#include "Application/ScopedMarker.h"

using namespace donut;
using namespace donut::app;
using namespace donut::engine;
using namespace donut::math;

#if ENABLE_NRC
#include "Nrc/NrcUtils.h"
#endif // ENABLE_NRC

#if RTXGI_HAS_SER_SHADERS_DXIL && DONUT_WITH_DX12
#include <nvrhi/d3d12.h>
#endif // RTXGI_HAS_SER_SHADERS_DXIL && DONUT_WITH_DX12

#include "LightingCb.h"
#include "GlobalCb.h"
#include "Shaders/TonemappingCb.h"
#include "Shaders/AutoExposureCb.h"

#if ENABLE_NRD
#include "Nrd/NrdConfig.h"
#endif // ENABLE_NRD


// Required for Agility SDK on Windows 10. Setup 1.c. 2.a.
// https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/
// The version matches the package the root CMakeLists.txt downloads and deploys next to the executable.
#ifndef RTXGI_D3D12_SDK_VERSION
#error "RTXGI_D3D12_SDK_VERSION must be defined by the root CMakeLists.txt"
#endif
extern "C"
{
    __declspec(dllexport) extern const UINT D3D12SDKVersion = RTXGI_D3D12_SDK_VERSION;
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


namespace
{
uint32_t DivideRoundUp(uint32_t x, uint32_t divisor)
{
    return (x + divisor - 1) / divisor;
}

donut::math::float2 GetPixelOffset(int frameIndex)
{
    donut::math::float2 pixelOffset;

    std::mt19937 rng(frameIndex);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    pixelOffset.x = dist(rng);
    pixelOffset.y = dist(rng);

    return pixelOffset;
}

#if RTXGI_HAS_SER_SHADERS_DXIL && DONUT_WITH_DX12
// On D3D12, Shader Execution Reordering uses the Shader Model 6.9 HitObject API, so the device and driver must report
// SM 6.9. This differs from nvrhi's Feature::ShaderExecutionReordering, which reports the NVAPI reordering capability.
bool SupportsD3D12ShaderModel69(nvrhi::IDevice* device)
{
    ID3D12Device* d3dDevice = static_cast<ID3D12Device*>(device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device));
    if (!d3dDevice)
    {
        return false;
    }

    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { .HighestShaderModel = D3D_SHADER_MODEL_6_9 };
    return SUCCEEDED(d3dDevice->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))) && shaderModel.HighestShaderModel >= D3D_SHADER_MODEL_6_9;
}
#endif // RTXGI_HAS_SER_SHADERS_DXIL && DONUT_WITH_DX12
} // namespace

Pathtracer::Pathtracer(DeviceManager* deviceManager, UIData& ui, nvrhi::GraphicsAPI api) : ApplicationBase(deviceManager), m_ui(ui), m_api(api)
{
    m_updatePassView = std::make_unique<PlanarViewConstants>();
}

Pathtracer::~Pathtracer()
{
#if ENABLE_NRC
    m_nrc->Shutdown();
#endif // ENABLE_NRC
}

bool Pathtracer::Initialize(int argc, const char* const* argv)
{
    char* sceneName = nullptr;
    int cameraIndex = -1;
    for (int n = 1; n < argc; n++)
    {
        const char* arg = argv[n];

        if (!strcmp(arg, "-noaccumulation"))
            m_ui.denoiserType = DenoiserType::None;
        else if (!strcmp(arg, "-noautoexposure"))
            m_ui.autoExposure = false;
        else if (!strcmp(arg, "-scene"))
            sceneName = (char*)argv[n + 1];
        else if (!strcmp(arg, "-camera"))
            cameraIndex = atoi(argv[n + 1]);
#if ENABLE_NRC
        else if (!strcmp(arg, "-nrc"))
            m_ui.techSelection = TechSelection::Nrc;
#endif // ENABLE_NRC
#if ENABLE_SHARC
        else if (!strcmp(arg, "-sharc"))
            m_ui.techSelection = TechSelection::Sharc;
#endif // ENABLE_SHARC
    }

#if SHADOW_RAY_USE_RAY_QUERY
    // Ray tracing pipeline support does not imply inline ray query support, and the shaders are compiled for it.
    if (!GetDevice()->queryFeatureSupport(nvrhi::Feature::RayQuery))
    {
        log::error(
            "The graphics device does not support inline ray queries, which the shadow rays require. Rebuild with SHADOW_RAY_USE_RAY_QUERY set to 0 in GlobalCb.h to use the shadow hit group instead.");
        return false;
    }
#endif // SHADOW_RAY_USE_RAY_QUERY

    m_resetAccumulation = true;
    m_accumulatedFrameCount = 0;

    m_nativeFileSystem = std::make_shared<vfs::NativeFileSystem>();
    const std::filesystem::path mediaPath = app::FindDirectory(*m_nativeFileSystem, app::GetDirectoryWithExecutable(), "Assets/Media");
    std::filesystem::path sceneFileName = mediaPath / "bistro.scene.json";
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
        sceneFileName = mediaPath / sceneName;

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

    m_accelStructBuilder = std::make_unique<AccelStructBuilder>();
    m_pathtracerScene = std::make_unique<SampleScene>();
    m_pathtracerScene->SetCameraIndex(cameraIndex); // Applied by SampleScene::OnLoaded once the scene cameras exist
    m_pathtracerPipelines = std::make_unique<Pipelines>();

    // Obtain the scene
    m_pathtracerScene->DiscoverScenes(mediaPath);

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
    if (!IsSceneLoaded())
    {
        log::error("Failed to load scene '%s'", sceneFileName.string().c_str());
        return false;
    }
    m_pathtracerScene->GetScene()->FinishedLoading(GetFrameIndex());

    m_camera.SetMoveSpeed(3.0f);

    m_constantBuffer =
        GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(LightingConstants), "LightingConstants", engine::c_MaxRenderPassConstantBufferVersions));

    m_debugBuffer =
        GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(GlobalConstants), "GlobalConstants", engine::c_MaxRenderPassConstantBufferVersions));

    m_tonemappingConstantBuffer = GetDevice()->createBuffer(
        nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(TonemappingConstants), "TonemappingConstants", engine::c_MaxRenderPassConstantBufferVersions));

    // Unified binding
    nvrhi::BindingLayoutDesc bindingLayoutDesc;
    bindingLayoutDesc.visibility = nvrhi::ShaderType::All;
    bindingLayoutDesc.registerSpaceIsDescriptorSet = (m_api == nvrhi::GraphicsAPI::VULKAN);
    for (int i = 0; i < DescriptorSetIDs::Count; ++i)
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

    m_autoExposurePass = std::make_unique<AutoExposurePass>(GetDevice());

#if RTXGI_HAS_SER_SHADERS_DXIL && DONUT_WITH_DX12
    if (m_api == nvrhi::GraphicsAPI::D3D12)
    {
        m_ui.serSupported = SupportsD3D12ShaderModel69(GetDevice());
    }
#endif // RTXGI_HAS_SER_SHADERS_DXIL && DONUT_WITH_DX12

#if RTXGI_HAS_SER_SHADERS_SPIRV
    if (m_api == nvrhi::GraphicsAPI::VULKAN)
    {
        // On Vulkan, the SER permutations use VK_NV_ray_tracing_invocation_reorder, which Main.cpp requests. nvrhi
        // reports the feature when the device enabled the extension and the driver actually reorders.
        m_ui.serSupported = GetDevice()->queryFeatureSupport(nvrhi::Feature::ShaderExecutionReordering);
    }
#endif // RTXGI_HAS_SER_SHADERS_SPIRV

    if (m_ui.serSupported)
    {
        log::info("Shader Execution Reordering is available and %s.", m_ui.enableSer ? "enabled" : "disabled");
    }
    else
    {
        // The checkbox is grayed out, so it must not read as enabled
        m_ui.enableSer = false;
        log::info(
            "Shader Execution Reordering is not available: it requires the SER shader permutations and %s.",
            m_api == nvrhi::GraphicsAPI::VULKAN ? "a device and driver that reorder with VK_NV_ray_tracing_invocation_reorder" : "a Shader Model 6.9 device and driver");
    }
    m_enableSer = m_ui.enableSer;

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

#if HASH_GRID_COMPACT
        bufferDesc.structStride = sizeof(uint32_t);
#else  // !HASH_GRID_COMPACT
        bufferDesc.structStride = sizeof(uint64_t);
#endif // HASH_GRID_COMPACT
        bufferDesc.byteSize = m_sharcEntriesNum * bufferDesc.structStride;
        bufferDesc.debugName = "m_sharcHashEntriesBuffer";
        m_sharcHashEntriesBuffer = GetDevice()->createBuffer(bufferDesc);

        bufferDesc.structStride = sizeof(uint32_t);
        bufferDesc.byteSize = m_sharcEntriesNum * bufferDesc.structStride;
        bufferDesc.debugName = "m_sharcLockBuffer";
        m_sharcLockBuffer = GetDevice()->createBuffer(bufferDesc);

#if SHARC_ENABLE_SH_ENCODING
        constexpr uint32_t sharcAccumulationBufferStride = 8 * sizeof(uint32_t);
        constexpr uint32_t sharcResolvedBufferStride = 6 * sizeof(uint32_t);
#else  // !SHARC_ENABLE_SH_ENCODING
        constexpr uint32_t sharcAccumulationBufferStride = 4 * sizeof(uint32_t);
        constexpr uint32_t sharcResolvedBufferStride = 4 * sizeof(uint32_t);
#endif // SHARC_ENABLE_SH_ENCODING

        bufferDesc.structStride = sharcAccumulationBufferStride;
        bufferDesc.byteSize = m_sharcEntriesNum * bufferDesc.structStride;
        bufferDesc.canHaveRawViews = true;

        bufferDesc.debugName = "m_sharcAccumulationBuffer";
        m_sharcAccumulationBuffer = GetDevice()->createBuffer(bufferDesc);

        bufferDesc.structStride = sharcResolvedBufferStride;
        bufferDesc.byteSize = m_sharcEntriesNum * bufferDesc.structStride;
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
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(0),
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
    m_autoExposurePass->CreatePipelines(*m_shaderFactory);
}

bool Pathtracer::CreateRayTracingPipelines()
{
    return m_pathtracerPipelines->CreateRayTracing(GetDevice(), *m_shaderFactory, BuildBindingLayouts(), m_ui.denoiserType == DenoiserType::Nrd, m_enableSer);
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
    // Donut calls SceneLoaded() after a synchronous load even when LoadScene() failed, in which case there is no scene
    // to set up. Returning before the base call keeps IsSceneLoaded() false so the caller can report the failure.
    if (!m_pathtracerScene->GetScene())
    {
        return;
    }

    ApplicationBase::SceneLoaded();

    bool rebuildAS = false;
    m_pathtracerScene->OnLoaded(GetFrameIndex(), m_ui, m_camera, m_resetAccumulation, rebuildAS, m_accumulatedFrameCount);

#if ENABLE_SHARC
    // The cache holds lighting from the previous scene. Keep the request pending until the SHaRC pass runs, which may
    // be several frames later if another indirect diffuse mode is active at load time.
    m_sharcCacheNeedsClear = true;
#endif // ENABLE_SHARC

    ResetAutoExposure();

    if (rebuildAS)
        m_accelStructBuilder->RequestRebuild();
}

void Pathtracer::SceneUnloading()
{
    GetDevice()->waitForIdle();

    m_shaderFactory->ClearCache();
    m_bindingCache->Clear();
    m_pathtracerScene->OnUnloading(m_ui);
    m_accelStructBuilder->RequestRebuild();

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
    m_frameTime = fElapsedTimeSeconds;
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
    m_autoExposurePass->ResetBindingCache();
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

    if (m_accelStructBuilder->NeedsRebuild())
    {
        device->waitForIdle();
        m_accelStructBuilder->Build(device, m_commandList, *m_pathtracerScene->GetScene(), m_ui.enableTransmission);
    }

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer),
        nvrhi::BindingSetItem::ConstantBuffer(1, m_debugBuffer),
        nvrhi::BindingSetItem::RayTracingAccelStruct(0, m_accelStructBuilder->GetTLAS()),
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
    m_viewPrev = m_view;
    m_view.SetViewport(windowViewport);
    m_view.SetMatrices(m_camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(dm::PI_f * 0.25f, windowViewport.width() / windowViewport.height(), 0.1f));
    m_view.SetPixelOffset(pixelOffset);

    if (frameIndex == 0)
        m_viewPrev = m_view;

    m_view.UpdateCache();
    m_viewPrev.UpdateCache();

    if (m_viewPrev.GetViewMatrix() != m_view.GetViewMatrix() || m_ui.enableAnimations)
        m_resetAccumulation = true;

    m_accumulatedFrameCount = m_resetAccumulation ? 1 : m_accumulatedFrameCount + 1;

    // The scene graph was already refreshed at the top of the frame; refreshing it again here would copy the current
    // transforms over the previous ones and erase the rigid-object motion that the motion vectors depend on
    m_pathtracerScene->GetScene()->RefreshBuffers(m_commandList, frameIndex);
    m_accelStructBuilder->UpdateTLAS(m_commandList, *m_pathtracerScene->GetScene(), m_ui.enableTransmission);

    LightingConstants constants = {};
    constants.skyColor = float4(m_ui.skyColor * (m_ui.enableSky ? m_ui.skyIntensity : 0.0f), 1.0f);

    // View constants
    m_view.FillPlanarViewConstants(constants.view);
    m_viewPrev.FillPlanarViewConstants(constants.viewPrev);

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
        nrcPerFrameSettings.numTrainingIterations = 1;
        nrcPerFrameSettings.resolveMode = m_ui.nrcResolveMode;
        nrcPerFrameSettings.trainTheCache = m_ui.nrcTrainCache;
        nrcPerFrameSettings.proportionPrimarySegmentsToTrainOn = 0.02f;
        // Only consulted by NRC when the termination heuristic is off (DirectCacheView); every other resolve mode skips delta vertices regardless
        nrcPerFrameSettings.skipDeltaVertices = m_ui.nrcSkipDeltaVertices;
        nrcPerFrameSettings.usedTrainingDimensions = nrc::ComputeIdealTrainingDimensions(nrcContextSettings.frameDimensions, nrcPerFrameSettings.numTrainingIterations);
        m_nrcUsedTrainingWidth = nrcPerFrameSettings.usedTrainingDimensions.x;
        m_nrcUsedTrainingHeight = nrcPerFrameSettings.usedTrainingDimensions.y;

        m_nrc->BeginFrame(m_commandList, nrcPerFrameSettings);
        m_nrc->PopulateShaderConstants(constants.nrcConstants);
    }
#endif

    constants.sharcRoughnessThreshold = 0.0f;

#if ENABLE_SHARC
    if (m_ui.techSelection == TechSelection::Sharc)
    {
        if (m_ui.sharcEnableUpdate && m_ui.sharcUpdateViewCamera)
        {
            m_sharcCameraPositionPrev = m_sharcCameraPosition;
            m_sharcCameraPosition = m_view.GetViewOrigin();
        }

        constants.sharcEntriesNum = m_sharcEntriesNum;
        constants.sharcDownscaleFactor = m_ui.sharcDownscaleFactor;
        constants.sharcSceneScale = m_ui.sharcSceneScale;
        constants.sharcRoughnessThreshold = m_ui.sharcRoughnessThreshold;
        constants.sharcCameraPositionPrev.xyz() = m_sharcCameraPositionPrev;
        constants.sharcCameraPosition.xyz() = m_sharcCameraPosition;
        constants.sharcAccumulationFrameNum = m_ui.sharcAccumulationFrameNum;
        constants.sharcStaleFrameNum = m_ui.sharcStaleFrameNum;
        constants.sharcEnableAntiFireflyFilter = m_ui.sharcEnableAntiFireflyFilter;
    }

    if (m_ui.sharcUpdateViewCamera || m_ui.techSelection != TechSelection::Sharc)
#endif // ENABLE_SHARC
        *m_updatePassView = constants.view;

    constants.updatePassView = *m_updatePassView;
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

#if ENABLE_NRD
    if (m_ui.denoiserType == DenoiserType::Nrd && !m_nrd)
    {
        m_nrd = std::make_unique<NrdIntegration>(device, nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR);

        if (!m_nrd->Initialize(renderWidth, renderHeight, *m_shaderFactory))
        {
            // Fall back to the undenoised output. Selecting NRD again in the UI retries the initialization.
            log::error("NRD initialization failed, the output is not denoised.");
            m_nrd = nullptr;
            m_ui.denoiserType = DenoiserType::None;
        }
    }
#endif // ENABLE_NRD

    // Evaluated after the NRD creation so that a failed initialization selects the pipelines without NRD output.
    bool resetDenoiser = m_enableNrd != (m_ui.denoiserType == DenoiserType::Nrd);
    // The SER toggle selects a different set of shader permutations; the image is unchanged, so nothing is reset.
    const bool serChanged = m_enableSer != m_ui.enableSer;
    if ((resetDenoiser && !skipDenoiser) || serChanged || m_ui.reloadShaders)
    {
        if (m_ui.reloadShaders)
        {
            m_shaderFactory->ClearCache();
            CreateCommonPipelines();
            m_ui.reloadShaders = false;
        }
        m_enableSer = m_ui.enableSer;
        CreateRayTracingPipelines();
    }

#if ENABLE_NRD
    m_enableNrd = (m_ui.denoiserType == DenoiserType::Nrd) && !skipDenoiser;
#endif // ENABLE_NRD

    m_commandList->writeBuffer(m_constantBuffer, &constants, sizeof(constants));

    GlobalConstants globalConstants = {};
    globalConstants.enableJitter = m_ui.enableJitter;
    globalConstants.enableBackFaceCull = m_ui.enableBackFaceCull;
    globalConstants.bouncesMax = m_ui.bouncesMax;
    globalConstants.frameIndex = m_frameIndex++;
    globalConstants.intensityScale = 1.0f;
    globalConstants.enableEmissives = m_ui.enableEmissives;
    globalConstants.enableLighting = m_ui.enableLighting;
    globalConstants.enableTransmission = m_ui.enableTransmission;
    globalConstants.enableAbsorption = m_ui.enableAbsorption;
    globalConstants.enableTransparentShadows = m_ui.enableTransparentShadows;
    globalConstants.enableSoftShadows = m_ui.enableSoftShadows;
    globalConstants.throughputThreshold = m_ui.throughputThreshold;
    globalConstants.enableRussianRoulette = m_ui.enableRussianRoulette;
    globalConstants.samplesPerPixel = m_ui.samplesPerPixel;
    globalConstants.roughnessMin = m_ui.roughnessMin;
    globalConstants.roughnessMax = std::max(m_ui.roughnessMin, m_ui.roughnessMax);
    globalConstants.metalnessMin = m_ui.metalnessMin;
    globalConstants.metalnessMax = std::max(m_ui.metalnessMin, m_ui.metalnessMax);
    globalConstants.transmissionIor = std::max(1.0f, m_ui.transmissionIor);

    // Only lights that were uploaded this frame can be targeted; anything else falls back to sampling all lights
    globalConstants.targetLight = (m_ui.targetLight >= 0 && m_ui.targetLight < constants.lightCount) ? m_ui.targetLight : -1;
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
    if (m_enableNrd)
    {
        nrd::ReblurHitDistanceParameters hitDistanceParameters;
        globalConstants.nrdHitDistanceParams = float4(hitDistanceParameters.A, hitDistanceParameters.B, hitDistanceParameters.C, 0.0f);
    }
#endif // ENABLE_NRD

    m_commandList->writeBuffer(m_debugBuffer, &globalConstants, sizeof(globalConstants));

    TonemappingConstants tonemappingConstants = {};
    // The manual EV reaches the tonemapper through the adapted luminance buffer, see the exposure metering below.
    tonemappingConstants.exposureScale = m_ui.autoExposure ? donut::math::exp2f(m_ui.exposureCompensation) : 1.0f;
    tonemappingConstants.toneMappingOperator = (uint)m_ui.toneMappingOperator;
    tonemappingConstants.clamp = (uint)m_ui.toneMappingClamp;
    tonemappingConstants.enableAccumulation = m_ui.denoiserType == DenoiserType::Accumulation;
    tonemappingConstants.recipAccumulatedFrames = tonemappingConstants.enableAccumulation ? (1.0f / (float)m_accumulatedFrameCount) : 1.0f;
    // ImGui lets Ctrl+click type values outside the slider range; below 1.0 the curve expands and 0 would divide by zero.
    const float reinhardWhitePoint = std::max(1.0f, m_ui.reinhardWhitePoint);
    tonemappingConstants.reinhardRecipWhitePointSquared = 1.0f / (reinhardWhitePoint * reinhardWhitePoint);
    tonemappingConstants.agxLook = (uint)m_ui.agxLook;
    m_commandList->writeBuffer(m_tonemappingConstantBuffer, &tonemappingConstants, sizeof(tonemappingConstants));

    m_commandList->clearState();

    using PT = Pipelines::Type;

    nvrhi::rt::State state;
    for (int i = 0; i < DescriptorSetIDs::Count; ++i)
        state.bindings.push_back(m_dummyBindingSets[i]); // Unified Binding

    state.bindings[DescriptorSetIDs::Globals] = m_globalBindingSet;
    state.bindings[DescriptorSetIDs::Bindless] = m_descriptorTable->GetDescriptorTable();
    state.bindings[DescriptorSetIDs::Denoiser] = m_denoiserBindingSet;

    if (enableDlss || m_enableNrd)
        m_commandList->clearTextureFloat(m_renderTargets->depth, nvrhi::AllSubresources, nvrhi::Color(0.0f));

    if (enableDlss)
        m_commandList->clearTextureFloat(m_renderTargets->motionVectors, nvrhi::AllSubresources, nvrhi::Color(0.0f));

#if ENABLE_NRC
    if (m_ui.techSelection == TechSelection::Nrc)
    {
        ScopedMarker scopedMarker(m_commandList, "Nrc");

        if (m_enableNrd)
            m_commandList->clearTextureFloat(m_renderTargets->outPathTracer, nvrhi::AllSubresources, nvrhi::Color(0.0f));

        assert(m_nrcBindingSet);
        state.bindings[DescriptorSetIDs::Nrc] = m_nrcBindingSet;
        {
            ScopedMarker scopedMarker(m_commandList, "NrcQueryPathtracingPass");

            state.shaderTable = m_pathtracerPipelines->GetPermutation(PT::NrcQuery).shaderTable;
            m_commandList->setRayTracingState(state);
            nvrhi::rt::DispatchRaysArguments args;
            args.width = renderWidth;
            args.height = renderHeight;
            m_commandList->dispatchRays(args);

            // NRC update pathtracing pass
            if (m_ui.nrcTrainCache)
            {
                // There is no dependency between the two raygens, so a barrier is not required here.
                // NVRHI sees that we're using the same UAVs and inserts UAV barriers.
                // Fortunately, we can tell it that we know better.
                m_commandList->setEnableAutomaticBarriers(false);

                ScopedMarker scopedMarker(m_commandList, "NrcUpdatePathtracingPass");

                state.shaderTable = m_pathtracerPipelines->GetPermutation(PT::NrcUpdate).shaderTable;
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

        if (m_ui.sharcEnableUpdate)
        {
            if (m_sharcCacheNeedsClear || m_ui.sharcEnableClear || (m_sharcMaterialDemodulationApplied != m_ui.sharcEnableMaterialDemodulation))
            {
                m_sharcMaterialDemodulationApplied = m_ui.sharcEnableMaterialDemodulation;

                m_commandList->clearBufferUInt(m_sharcHashEntriesBuffer, m_sharcInvalidEntry);
                m_commandList->clearBufferUInt(m_sharcAccumulationBuffer, 0);
                m_commandList->clearBufferUInt(m_sharcResolvedBuffer, 0);

                m_sharcCacheNeedsClear = false;
            }

            // SHARC update
            {
                state.shaderTable = m_pathtracerPipelines->GetPermutation(PT::SharcUpdate).shaderTable;
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
            state.shaderTable = m_pathtracerPipelines->GetPermutation(PT::SharcQuery).shaderTable;
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
    if (m_enableNrd)
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
        m_nrd->RunDenoiserPasses(m_commandList, *m_renderTargets, 0, m_view, m_viewPrev, GetFrameIndex(), 0.01f, 0.05f, false, false, &reblurSettings, resetDenoiser);

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

    // DLSS Ray Reconstruction does not consume an exposure, and it displays a denoised image, so its output is metered
    // after the evaluation below. Metering its noisy single-sample input instead would count every zero-radiance sample
    // as a dark pixel and over-expose the denoised result.
    bool meterRayReconstructionOutput = false;
#if DONUT_WITH_DLSS
    meterRayReconstructionOutput = useRayReconstruction && m_dlss->IsRayReconstructionInitialized();
#endif // DONUT_WITH_DLSS
    const bool updateAutoExposure = m_ui.autoExposure && (!m_ui.autoExposureLock || m_autoExposureResetPending);

    // Exposure metering runs before upscaling so that the tonemapper and DLSS share one exposure. With auto exposure
    // off, the adapted luminance is the one the manual EV maps to middle grey.
    if (m_ui.autoExposure)
    {
        if (updateAutoExposure && !meterRayReconstructionOutput)
        {
            // The log mean and the percentile window are nonlinear, so metering the noisy single-sample frame while
            // the display shows the accumulated radiance would converge to a biased exposure. The tonemapping pass
            // updates the accumulation buffer after this point, so it still holds the previous frame's accumulated
            // result here; meter that whenever it is valid, and fall back to the fresh frame otherwise.
            const bool meterAccumulation = tonemappingConstants.enableAccumulation && m_accumulatedFrameCount > 1;
            nvrhi::ITexture* meteringSource = meterAccumulation ? m_accumulationBuffer.Get() : m_renderTargets->outPathTracer.Get();
            const uint2 meteringSize = meterAccumulation ? uint2(outputWidth, outputHeight) : uint2(renderWidth, renderHeight);
            m_autoExposurePass->Render(m_commandList, meteringSource, meteringSize, m_ui.autoExposureSettings, m_frameTime, m_autoExposureResetPending);
        }
    }
    else
    {
        m_autoExposurePass->SetAdaptedLuminance(m_commandList, AUTO_EXPOSURE_KEY / donut::math::exp2f(m_ui.exposureAdjustment));
    }

#if DONUT_WITH_DLSS
    if (enableDlss)
    {
        donut::render::DLSS::EvaluateParameters params;
        params.depthTexture = m_renderTargets->depth;
        params.motionVectorsTexture = m_renderTargets->motionVectors;
        params.inputColorTexture = m_renderTargets->outPathTracer;
        params.outputColorTexture = m_accumulationBuffer;
        params.exposureBuffer = m_autoExposurePass->GetExposureBuffer();
        params.exposureScale = AutoExposurePass::GetDlssExposureScale(tonemappingConstants.exposureScale);

        if (useRayReconstruction)
        {
            params.normalRoughness = m_renderTargets->normalRoughness;
            params.diffuseAlbedo = m_renderTargets->diffuseAlbedo;
            params.specularAlbedo = m_renderTargets->specularAlbedo;
        }

        m_dlss->Evaluate(m_commandList, params, m_view);
    }
#endif // DONUT_WITH_DLSS

    if (updateAutoExposure && meterRayReconstructionOutput)
    {
        m_autoExposurePass->Render(
            m_commandList, m_accumulationBuffer.Get(), uint2(outputWidth, outputHeight), m_ui.autoExposureSettings, m_frameTime, m_autoExposureResetPending);
    }
    m_autoExposureResetPending = false;

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
        bindingSetDesc.bindings = { nvrhi::BindingSetItem::ConstantBuffer(0, m_tonemappingConstantBuffer),
                                    nvrhi::BindingSetItem::TypedBuffer_SRV(0, m_autoExposurePass->GetExposureBuffer()),
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

void Pathtracer::ResetAutoExposure()
{
    m_autoExposureResetPending = true;
}

void Pathtracer::RebuildAccelerationStructure()
{
    m_accelStructBuilder->RequestRebuild();
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

Pipelines::BindingLayouts Pathtracer::BuildBindingLayouts() const
{
    Pipelines::BindingLayouts layouts;
    layouts.global = m_globalBindingLayout;
    layouts.denoiser = m_denoiserBindingLayout;
    layouts.bindless = m_bindlessLayout;
#if ENABLE_NRC
    layouts.nrc = m_nrcBindingLayout;
#endif
#if ENABLE_SHARC
    layouts.sharc = m_sharcBindingLayout;
#endif
    for (int i = 0; i < DescriptorSetIDs::Count; ++i)
        layouts.dummy[i] = m_dummyLayouts[i];
    return layouts;
}