# Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required (VERSION 3.19)

project(
    RTXGI2Samples
    DESCRIPTION "Samples showcasing NRC and SHARC features"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Set MSVC debug level
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=2")
    #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set (CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Warnings as errors
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# -----------------------------------------------------------------------------
# Other packages downloaded from zips
# -----------------------------------------------------------------------------

# Helper to download and unzip a package from a URL
# Uses a zero-length file to identify the version of the package
function(CheckAndDownloadPackage NAME VERSION LOCAL_PATH URL ARCHIVE_TYPE)
    # Do we already have the correct version?
    if(NOT EXISTS ${LOCAL_PATH}/${VERSION}.ver)
        # Was there a previous version that we need to delete?
        if(EXISTS ${LOCAL_PATH})
            message(STATUS "Deleting old " ${NAME})
            file(REMOVE_RECURSE ${LOCAL_PATH})
        endif()
        message(STATUS "Obtaining " ${NAME} " " ${VERSION})
        file(DOWNLOAD ${URL} ${LOCAL_PATH}.${ARCHIVE_TYPE})
        message(STATUS "Extracting " ${NAME})
        file(ARCHIVE_EXTRACT INPUT ${LOCAL_PATH}.${ARCHIVE_TYPE} DESTINATION ${LOCAL_PATH})
        file(REMOVE ${LOCAL_PATH}.${ARCHIVE_TYPE})
        # Create an empty file so we know which version we have
        file(WRITE ${LOCAL_PATH}/${VERSION}.ver)
    endif()
endfunction()

# Agility SDK: the package version selects the download, the SDK version is what the samples export as D3D12SDKVersion.
set(D3D12_AGILITY_SDK_PACKAGE_VERSION "1.619.6")
set(D3D12_AGILITY_SDK_VERSION "619")
CheckAndDownloadPackage("Agility SDK" "v${D3D12_AGILITY_SDK_PACKAGE_VERSION}" ${CMAKE_CURRENT_SOURCE_DIR}/External/AgilitySDK https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/${D3D12_AGILITY_SDK_PACKAGE_VERSION} "zip")
add_compile_definitions(RTXGI_D3D12_SDK_VERSION=${D3D12_AGILITY_SDK_VERSION})
CheckAndDownloadPackage("DXC" "v1.7.2308" ${CMAKE_CURRENT_SOURCE_DIR}/External/dxc https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.7.2308/dxc_2023_08_14.zip "zip")

# -----------------------------------------------------------------------------
# Configure options
# -----------------------------------------------------------------------------

# Setup shader compiler 
set(DXC_DXIL_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/External/dxc/bin/x64/dxc.exe" CACHE STRING "DXC shader compiler path")
set(DXC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External/dxc/bin/x64/dxc.exe")
set(DXC_SPIRV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External/dxc/bin/x64/dxc.exe")

# Setup denoiser support (NRD)
option(USE_RAY_TRACING_DENOISER "Include NRD as part of the sample." ON)
set(NRD_PATH "${CMAKE_CURRENT_SOURCE_DIR}/External/RayTracingDenoiser")


include("${CMAKE_CURRENT_SOURCE_DIR}/External/Donut/nvrhi/cmake/NvrhiTargetArch.cmake")
set(RTXGI_TARGET_ARCH "${NVRHI_TARGET_ARCH}")
message(STATUS "RTXGI target architecture: ${RTXGI_TARGET_ARCH}")

# Enable DLSS support (DLSS SDK 310.9+ ships both x64 and arm64 Windows binaries)
option(DONUT_WITH_DLSS "" ON)

# Setup asset importer support
option(DONUT_WITH_ASSIMP "" OFF)

# Per-architecture output directory so x64 and arm64 builds do not overwrite each other's binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Bin/${RTXGI_TARGET_ARCH}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(DONUT_SHADERS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/framework")

# Setup option for local NRC library development
set(NRC_CUSTOM_DISTRO_PATH "" CACHE PATH "Enter the path to NRC distro directory containing lib, include, and bin.")

# Options for NRC DLL signing verification
option(NRC_ENABLE_DLL_CHECK "Enable NRC DLL signing verification." ON)
if(NRC_ENABLE_DLL_CHECK)
  add_compile_definitions(NRC_ENABLE_DLL_CHECK)
endif()

# Check if the custom directory path is provided.
if(NOT NRC_CUSTOM_DISTRO_PATH)
    message(STATUS "Custom NRC distro path is not provided.")
    set(NRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Libraries/Nrc)
else()
    set(NRC_DIR ${NRC_CUSTOM_DISTRO_PATH})  
endif()

# Fall back to the flat bin/ and lib/ layout for older packages and local NRC SDK builds.
if(EXISTS ${NRC_DIR}/bin/${RTXGI_TARGET_ARCH} AND EXISTS ${NRC_DIR}/lib/${RTXGI_TARGET_ARCH})
    set(NRC_BIN_DIR ${NRC_DIR}/bin/${RTXGI_TARGET_ARCH})
    set(NRC_LIB_DIR ${NRC_DIR}/lib/${RTXGI_TARGET_ARCH})
else()
    set(NRC_BIN_DIR ${NRC_DIR}/bin)
    set(NRC_LIB_DIR ${NRC_DIR}/lib)
endif()
message(STATUS "NRC binaries: ${NRC_BIN_DIR}")

# Enable NRC Vulkan support if the package contains a Vulkan binary (production or dev naming)
if(EXISTS ${NRC_BIN_DIR}/NRC_Vulkan.dll OR EXISTS ${NRC_BIN_DIR}/NRC_Vulkan_debug.dll OR EXISTS ${NRC_BIN_DIR}/NRC_Vulkan_relwithdebinfo.dll)
    set(SETUP_NRC_WITH_VULKAN TRUE) #TODO: maybe not required...
    add_compile_definitions(NRC_WITH_VULKAN)
endif()

unset(DONUT_WITH_VULKAN CACHE)
unset(NVRHI_WITH_VULKAN CACHE)

# Force Vulkan support
option(DONUT_WITH_VULKAN "Enable the Vulkan version of Donut" ON)

# Compile all SPIR-V shaders (Donut, NRD and the samples) with the DXC that ShaderMake downloads, which is also the one
# used for DXIL, rather than with the DXC of whichever Vulkan SDK is installed. 
set(SHADERMAKE_FIND_DXC_VK OFF CACHE BOOL "Find DXC in Vulkan SDK and populate 'SHADERMAKE_DXC_VK_PATH' variable" FORCE)
unset(SHADERMAKE_DXC_VK_PATH CACHE)

# Add projects
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/External/Donut)

message(STATUS "NVRHI Vulkan support " "${NVRHI_WITH_VULKAN}")
message(STATUS "Donut Vulkan support " "${DONUT_WITH_VULKAN}")

if(NVRHI_WITH_VULKAN OR NVRHI_WITH_DX12)
    add_subdirectory(Samples)
    # Set Pathtracer sample project as a startup project
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Pathtracer)
endif()

# Add SHARC includes to the solution
set(SHARC_LIBRARY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/Libraries/Sharc")
file(GLOB_RECURSE INCLUDE_SHARC "${SHARC_LIBRARY_ROOT}/*.h")

# Add NRC includes to the solution
file(GLOB_RECURSE INCLUDE_NRC
    "${NRC_DIR}/include/*.hlsli"
    "${NRC_DIR}/include/*.h"
)

add_library(Libraries ${INCLUDE_SHARC} ${INCLUDE_NRC})
set_target_properties(Libraries PROPERTIES LINKER_LANGUAGE CXX)
set(VS_FOLDER_LIB_NAME "Libraries")
set_target_properties(Libraries PROPERTIES FOLDER ${VS_FOLDER_LIB_NAME})

source_group("Sharc" FILES ${INCLUDE_SHARC})
source_group("Nrc"   FILES ${INCLUDE_NRC})

# Include External
add_subdirectory(External)
