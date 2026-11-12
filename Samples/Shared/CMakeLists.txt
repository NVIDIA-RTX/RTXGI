# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

set(folder "Samples/Shared")
set(shared_root ${CMAKE_CURRENT_SOURCE_DIR})

# Every library exports the shared root, so sample code includes by theme folder: "Nrc/NrcIntegration.h".

add_library(SharedApplication STATIC
    Application/Defines.h
    Application/ScopedMarker.cpp
    Application/ScopedMarker.h
)
target_include_directories(SharedApplication PUBLIC ${shared_root})
target_link_libraries(SharedApplication PUBLIC donut_app donut_engine)

add_library(SharedNrc STATIC
    Nrc/NrcIntegration.cpp
    Nrc/NrcIntegration.h
    Nrc/NrcUtils.cpp
    Nrc/NrcUtils.h
)
target_include_directories(SharedNrc PUBLIC ${shared_root} "${NRC_DIR}/include")
target_link_libraries(SharedNrc PUBLIC donut_app donut_engine)
# NrcUtils reports failures through the wide-character MessageBox.
target_compile_definitions(SharedNrc PRIVATE UNICODE _UNICODE)

add_library(SharedScene STATIC
    Scene/AccelStructBuilder.cpp
    Scene/AccelStructBuilder.h
)
target_include_directories(SharedScene PUBLIC ${shared_root})
target_link_libraries(SharedScene PUBLIC donut_engine SharedApplication)

add_library(SharedPostProcess STATIC
    PostProcess/AutoExposurePass.cpp
    PostProcess/AutoExposurePass.h
)
target_include_directories(SharedPostProcess PUBLIC ${shared_root})
target_link_libraries(SharedPostProcess PUBLIC donut_engine)

# The shared shaders are compiled by each sample's own shader target (see the samples' Shaders.cfg);
# this target only lists them in the IDE next to the libraries that go with them.
file(GLOB shared_shaders "Shaders/*.hlsl" "Shaders/*.hlsli" "Shaders/*.h")
set_source_files_properties(${shared_shaders} PROPERTIES VS_TOOL_OVERRIDE "None")
add_custom_target(SharedShaders SOURCES ${shared_shaders})

foreach(target SharedApplication SharedNrc SharedScene SharedPostProcess SharedShaders)
    set_target_properties(${target} PROPERTIES FOLDER ${folder})
    get_target_property(target_sources ${target} SOURCES)
    source_group(TREE ${shared_root} FILES ${target_sources})
endforeach()
