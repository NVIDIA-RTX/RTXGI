# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Shader Execution Reordering (SER). The SER shader permutations are built per shader format, depending on what the
# compiler for that format supports. The samples check device support at runtime.
option(RTXGI_ENABLE_SER "Build the Shader Execution Reordering shader permutations that the shader compilers support" ON)

# D3D12 (Shader Model 6.9 HitObject). The SM 6.9 shader permutations are only built when the DXC that ShaderMake uses
# for DXIL (SHADERMAKE_DXC_PATH, provided by Donut) is a production 1.9+ compiler that offers lib_6_9; DXC 1.8 only had
# a preview target that needs preview runtimes.
set(RTXGI_SER_SHADERS_DXIL OFF)
set(RTXGI_SER_SHADERMAKE_DEFINE_DXIL "")
if(NOT RTXGI_ENABLE_SER)
    message(STATUS "SER shader permutations disabled (RTXGI_ENABLE_SER=OFF)")
elseif(DONUT_WITH_DX12 AND SHADERMAKE_DXC_PATH)
    execute_process(
        COMMAND "${SHADERMAKE_DXC_PATH}" -help
        OUTPUT_VARIABLE RTXGI_DXC_HELP
        ERROR_VARIABLE RTXGI_DXC_HELP_ERROR
    )
    string(APPEND RTXGI_DXC_HELP "${RTXGI_DXC_HELP_ERROR}")

    set(RTXGI_DXC_VERSION "unknown")
    set(RTXGI_DXC_HAS_PRODUCTION_SM69 OFF)
    if(RTXGI_DXC_HELP MATCHES "dxcompiler.dll: ([0-9]+)\\.([0-9]+)")
        set(RTXGI_DXC_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
        if(CMAKE_MATCH_1 GREATER 1 OR (CMAKE_MATCH_1 EQUAL 1 AND CMAKE_MATCH_2 GREATER_EQUAL 9))
            set(RTXGI_DXC_HAS_PRODUCTION_SM69 ON)
        endif()
    endif()

    if(RTXGI_DXC_HAS_PRODUCTION_SM69 AND RTXGI_DXC_HELP MATCHES "lib_6_9")
        set(RTXGI_SER_SHADERS_DXIL ON)
        set(RTXGI_SER_SHADERMAKE_DEFINE_DXIL "-D RTXGI_COMPILE_SER")
        message(STATUS "SER shader permutations for D3D12 enabled (DXC ${RTXGI_DXC_VERSION})")
    else()
        message(STATUS "SER shader permutations for D3D12 disabled: DXC ${RTXGI_DXC_VERSION} at ${SHADERMAKE_DXC_PATH} does not provide production Shader Model 6.9")
    endif()
else()
    message(STATUS "SER shader permutations for D3D12 disabled: requires the D3D12 backend and a DXC for DXIL")
endif()

# Vulkan (VK_NV_ray_tracing_invocation_reorder). DXC has no SPIR-V lowering for the HitObject API, so the SPIR-V
# permutations use the hit object instructions through inline SPIR-V (Shared/Shaders/HitObjectNV.hlsli) and stay at the
# samples' regular shader model. They rely on the SPIR-V shaders being compiled with the DXC that ShaderMake downloads
# (see the root CMakeLists.txt): the DXC of Vulkan SDKs before 1.4.341 cannot legalize modules that declare
# SPV_NV_shader_invocation_reorder and fails with "variables may not allocate a pointer type".
set(RTXGI_SER_SHADERS_SPIRV OFF)
set(RTXGI_SER_SHADERMAKE_DEFINE_SPIRV "")
if(RTXGI_ENABLE_SER AND DONUT_WITH_VULKAN AND EXISTS "${SHADERMAKE_DXC_VK_PATH}")
    set(RTXGI_SER_SHADERS_SPIRV ON)
    set(RTXGI_SER_SHADERMAKE_DEFINE_SPIRV "-D RTXGI_COMPILE_SER")
    message(STATUS "SER shader permutations for Vulkan enabled")
elseif(RTXGI_ENABLE_SER)
    message(STATUS "SER shader permutations for Vulkan disabled: requires the Vulkan backend and a DXC for SPIR-V")
endif()

# Code shared by the samples builds first so the sample executables can link it.
add_subdirectory(Shared)
add_subdirectory(Pathtracer)
add_subdirectory(IndirectDiffuse)
