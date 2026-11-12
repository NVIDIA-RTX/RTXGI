# Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required (VERSION 3.19)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../External/donut/compileshaders.cmake)

file(GLOB shaders "Shaders/*.hlsl" "Shaders/*.hlsli")
file(GLOB sources "*.cpp" "*.h")
file(GLOB nrd_sources "Nrd/*.cpp" "Nrd/*.h")

set(project IndirectDiffuse)
set(folder "Samples/IndirectDiffuse")


# Is the NRC package a dev package (internal), or production
if(EXISTS ${NRC_LIB_DIR}/NRC_D3D12_relwithdebinfo.lib OR EXISTS ${NRC_LIB_DIR}/NRC_D3D12_debug.lib) 
    set (NRC_DEV_BUILD true)
else()
    set (NRC_DEV_BUILD false)
endif()

if(NRC_DEV_BUILD)
    set(NRC_LIBRARY_NAME_D3D12_DEBUG NRC_D3D12_debug)
    set(NRC_LIBRARY_NAME_D3D12_RELEASE NRC_D3D12_relwithdebinfo)

    if(DONUT_WITH_VULKAN AND SETUP_NRC_WITH_VULKAN)
        set(NRC_LIBRARY_NAME_VULKAN_DEBUG NRC_Vulkan_debug)
        set(NRC_LIBRARY_NAME_VULKAN_RELEASE NRC_Vulkan_relwithdebinfo)
    endif()
else()
    set(NRC_LIBRARY_NAME_D3D12_DEBUG NRC_D3D12)
    set(NRC_LIBRARY_NAME_D3D12_RELEASE NRC_D3D12)

    if(DONUT_WITH_VULKAN AND SETUP_NRC_WITH_VULKAN)
        set(NRC_LIBRARY_NAME_VULKAN_DEBUG NRC_Vulkan)
        set(NRC_LIBRARY_NAME_VULKAN_RELEASE NRC_Vulkan)
    endif()
endif()

# Set SHARC includes
set(SHARC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries)

# Shaders shared between the samples live next to the shared C++; the sample directory is on the path too so a
# shared shader can include this sample's constant buffer headers.
set(SHARED_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Shared/Shaders)
set(SHADER_INCLUDE_ARGS "-I ${SHARED_SHADER_DIR} -I ${CMAKE_CURRENT_SOURCE_DIR} -I ${NRC_DIR}/include -I ${SHARC_INCLUDE_DIR}/Sharc/include -I ${NRD_PATH}/Shaders")

# Shader Model 6_6 requires Agility SDK on Windows 10.
# RTXGI_SER_SHADERMAKE_DEFINE_<format> enables the Shader Execution Reordering block in Shaders.cfg for the shader
# formats whose compiler can build it (see Samples/CMakeLists.txt).
if(DONUT_WITH_VULKAN)
    set (SHADERMAKE_GENERAL_ARGS_SPIRV "--shaderModel 6_6 ${SHADER_INCLUDE_ARGS} --vulkanMemoryLayout dx ${RTXGI_SER_SHADERMAKE_DEFINE_SPIRV}")
endif()
set (SHADERMAKE_GENERAL_ARGS_DXIL "--shaderModel 6_6 --platform DXIL --WX --PDB ${SHADER_INCLUDE_ARGS} ${RTXGI_SER_SHADERMAKE_DEFINE_DXIL}")

donut_compile_shaders(
        TARGET ${project}_shaders
        CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/Shaders.cfg
        SOURCES ${shaders}
        FOLDER ${folder}
        SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}/spirv
        DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}/dxil
        SHADERMAKE_OPTIONS_SPIRV ${SHADERMAKE_GENERAL_ARGS_SPIRV}
        SHADERMAKE_OPTIONS_DXIL ${SHADERMAKE_GENERAL_ARGS_DXIL}
)

add_executable(${project} WIN32 ${sources} ${nrd_sources})
# Tells the sample which SER shader permutations exist; whether the device can run them is decided at runtime.
target_compile_definitions(${project} PRIVATE
    RTXGI_HAS_SER_SHADERS_DXIL=$<BOOL:${RTXGI_SER_SHADERS_DXIL}>
    RTXGI_HAS_SER_SHADERS_SPIRV=$<BOOL:${RTXGI_SER_SHADERS_SPIRV}>
)
target_link_libraries(${project} donut_render donut_app donut_engine NRD SharedApplication SharedNrc SharedScene SharedPostProcess)
add_dependencies(${project} ${project}_shaders nrd_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

ADD_DEFINITIONS(-DUNICODE)
ADD_DEFINITIONS(-D_UNICODE)

# NRC libs
if(NRC_DEV_BUILD)
    target_link_libraries(${project} debug "${NRC_LIB_DIR}/${NRC_LIBRARY_NAME_D3D12_DEBUG}.lib")
    target_link_libraries(${project} optimized "${NRC_LIB_DIR}/${NRC_LIBRARY_NAME_D3D12_RELEASE}.lib")
else()
    target_link_libraries(${project} "${NRC_LIB_DIR}/${NRC_LIBRARY_NAME_D3D12_RELEASE}.lib")
endif()

if(DONUT_WITH_VULKAN AND SETUP_NRC_WITH_VULKAN)
    if(NRC_DEV_BUILD)
        target_link_libraries(${project} debug "${NRC_LIB_DIR}/${NRC_LIBRARY_NAME_VULKAN_DEBUG}.lib")
        target_link_libraries(${project} optimized "${NRC_LIB_DIR}/${NRC_LIBRARY_NAME_VULKAN_RELEASE}.lib")
    else()
        target_link_libraries(${project} "${NRC_LIB_DIR}/${NRC_LIBRARY_NAME_VULKAN_RELEASE}.lib")
    endif()
endif()

# NRC includes
include_directories("${NRC_DIR}/include" "${NRD_PATH}/Include")

# Copy Agility SDK binaries (if needed) one level below project executable to avoid known issues.
# Details in the section "Known Issues" https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/
set(AGILITY_SDK_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../External/AgilitySDK/build/native/bin/${RTXGI_TARGET_ARCH})
add_custom_command(TARGET ${project} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${AGILITY_SDK_BIN_DIR}/D3D12Core.dll
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/D3D12/D3D12Core.dll
)

add_custom_command(TARGET ${project} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${AGILITY_SDK_BIN_DIR}/d3d12SDKLayers.dll
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/D3D12/d3d12SDKLayers.dll
)

# Copy runtime libraries
file(GLOB NVRTC64_PATH "${NRC_BIN_DIR}/nvrtc64_*")
file(GLOB CUDART64_PATH "${NRC_BIN_DIR}/cudart64_*")
file(GLOB NVRTC_BUILTINS64 "${NRC_BIN_DIR}/nvrtc-builtins64_*")
add_custom_command(
    TARGET ${project} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${NVRTC64_PATH}"
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDART64_PATH}"
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E copy "${NVRTC_BUILTINS64}"
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

set( copy_nrc_d3d_command
    ${CMAKE_COMMAND} -E copy
    "${NRC_BIN_DIR}/$<$<CONFIG:Debug>:${NRC_LIBRARY_NAME_D3D12_DEBUG}>$<$<CONFIG:Release,RelWithDebInfo>:${NRC_LIBRARY_NAME_D3D12_RELEASE}>.dll"
    $<TARGET_FILE_DIR:${project}>
)
add_custom_command( TARGET ${project} POST_BUILD
    COMMAND "${copy_nrc_d3d_command}"
    COMMAND_EXPAND_LISTS
)

if(DONUT_WITH_VULKAN AND SETUP_NRC_WITH_VULKAN)
    set( copy_nrc_vulkan_command
        ${CMAKE_COMMAND} -E copy
        "${NRC_BIN_DIR}/$<$<CONFIG:Debug>:${NRC_LIBRARY_NAME_VULKAN_DEBUG}>$<$<CONFIG:Release,RelWithDebInfo>:${NRC_LIBRARY_NAME_VULKAN_RELEASE}>.dll"
        $<TARGET_FILE_DIR:${project}>
    )
    add_custom_command( TARGET ${project} POST_BUILD 
        COMMAND "${copy_nrc_vulkan_command}"
        COMMAND_EXPAND_LISTS
    )
endif()

if (DLSS_SHARED_LIBRARY_PATH)
    # Donut selects the development DLSS DLLs for Debug builds. Their profiling path clears an internal texture
    # (nv.ngx.dlss.DummyTextureForProfiling) while it is still in the copy-destination state, which the D3D12 debug layer
    # shipped with Agility SDK 1.619 reports as INVALID_SUBRESOURCE_STATE; with break-on-error enabled the sample stops as
    # soon as DLSS is turned on. The release DLLs do not contain that path and the samples do not use the development
    # overlay, so the release DLLs are deployed for every configuration. Observed with DLSS SDK 310.9.1.
    string(REPLACE "$<IF:$<CONFIG:Debug>,dev,rel>" "rel" dlss_shared_library_path "${DLSS_SHARED_LIBRARY_PATH}")
    string(REPLACE "$<IF:$<CONFIG:Debug>,dev,rel>" "rel" dlss_rr_shared_library_path "${DLSS_RR_SHARED_LIBRARY_PATH}")

    add_custom_command(TARGET ${project} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${dlss_shared_library_path}"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/")
    add_custom_command(TARGET ${project} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${dlss_rr_shared_library_path}"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/")
endif()
