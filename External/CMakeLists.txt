#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

##########################################################################
########################### RayTracingDenoiser ###########################
##########################################################################

# see https://github.com/NVIDIAGameWorks/RayTracingDenoiser#cmake-options
set(NRD_NORMAL_ENCODING "2" CACHE STRING "Normal encoding variant (0-4, matches nrd::NormalEncoding)")
set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "Roughness encoding variant (0-2, matches nrd::RoughnessEncoding)")
# On arm64, NRD's MathLib runs on top of sse2neon, which needs the conforming MSVC preprocessor and
# emits a couple of warnings that NRD's /WX would otherwise promote. MSVC also has no SVML intrinsics
# on arm64, so force MathLib's emulation path. All of this is scoped to this directory (NRD only).
if(MSVC AND RTXGI_TARGET_ARCH STREQUAL "arm64")
    string(APPEND CMAKE_CXX_FLAGS " /Zc:preprocessor /wd4310 /wd4127")
    add_compile_definitions(ML_SVML_AVAILABLE=0)
endif()

# The samples compile NRD's shaders through Donut's ShaderMake pipeline (below) and load them at runtime,
# so the library must not embed its own copies.
option(NRD_EMBEDS_SPIRV_SHADERS "NRD embeds SPIRV shaders" OFF)
option(NRD_EMBEDS_DXIL_SHADERS "NRD embeds DXIL shaders" OFF)
option(NRD_EMBEDS_DXBC_SHADERS "NRD embeds DXBC shaders" OFF)

add_subdirectory("RayTracingDenoiser")

# NRD fetches MathLib at configure time; its shaders include "ml.hlsli" from there
get_target_property(NRD_MATHLIB_DIR MathLib SOURCE_DIR)

file(GLOB NRD_SHADER_SOURCES "RayTracingDenoiser/Shaders/*.hlsl" "RayTracingDenoiser/Shaders/*.hlsli")

include(${CMAKE_CURRENT_SOURCE_DIR}/donut/compileshaders.cmake)

# Compiling NRD's own config keeps the permutations in sync with the "shaderIdentifier" strings
# the library reports for each pipeline.
donut_compile_shaders(
    TARGET nrd_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/RayTracingDenoiser/Shaders/Shaders.cfg
    SOURCES ${NRD_SHADER_SOURCES}
    FOLDER "NRD"
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/nrd/dxil
    SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/nrd/spirv
    INCLUDES ${NRD_MATHLIB_DIR}
    SHADERMAKE_OPTIONS "-D NRD_INTERNAL"
)

set(RTXDI_SKIP_SHADER_VALIDATION ON CACHE STRING "Skip RTXDI shader validation")
