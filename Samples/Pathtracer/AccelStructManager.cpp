/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <donut/engine/SceneGraph.h>
#include <nvrhi/utils.h>

#include "AccelStructManager.h"
#include "ScopedMarker.h"

using namespace donut::engine;

void AccelStructManager::Build(nvrhi::IDevice* device, nvrhi::ICommandList* commandList, donut::engine::Scene& scene, bool enableTransmission)
{
    for (const auto& mesh : scene.GetSceneGraph()->GetMeshes())
    {
        if (mesh->buffers->hasAttribute(VertexAttribute::JointWeights))
            continue;

        nvrhi::rt::AccelStructDesc blasDesc;
        GetMeshBLASDesc(*mesh, blasDesc, !enableTransmission);

        nvrhi::rt::AccelStructHandle accelStruct = device->createAccelStruct(blasDesc);

        if (!mesh->skinPrototype)
            nvrhi::utils::BuildBottomLevelAccelStruct(commandList, accelStruct, blasDesc);

        mesh->accelStruct = accelStruct;
    }

    nvrhi::rt::AccelStructDesc tlasDesc;
    tlasDesc.isTopLevel = true;
    tlasDesc.topLevelMaxInstances = scene.GetSceneGraph()->GetMeshInstances().size();
    m_topLevelAS = device->createAccelStruct(tlasDesc);
    m_rebuildAS = false;
}

void AccelStructManager::UpdateTLAS(nvrhi::ICommandList* commandList, donut::engine::Scene& scene, bool enableTransmission)
{
    ScopedMarker scopedMarkerSkinnedBLAS(commandList, "Skinned BLAS Updates");

    for (const auto& skinnedInstance : scene.GetSceneGraph()->GetSkinnedMeshInstances())
    {
        commandList->setAccelStructState(skinnedInstance->GetMesh()->accelStruct, nvrhi::ResourceStates::AccelStructWrite);
        commandList->setBufferState(skinnedInstance->GetMesh()->buffers->vertexBuffer, nvrhi::ResourceStates::AccelStructBuildInput);
    }
    commandList->commitBarriers();

    for (const auto& skinnedInstance : scene.GetSceneGraph()->GetSkinnedMeshInstances())
    {
        nvrhi::rt::AccelStructDesc blasDesc;
        GetMeshBLASDesc(*skinnedInstance->GetMesh(), blasDesc, !enableTransmission);
        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, skinnedInstance->GetMesh()->accelStruct, blasDesc);
    }

    std::vector<nvrhi::rt::InstanceDesc> instances;
    for (const auto& instance : scene.GetSceneGraph()->GetMeshInstances())
    {
        nvrhi::rt::InstanceDesc instanceDesc;
        instanceDesc.bottomLevelAS = instance->GetMesh()->accelStruct;
        instanceDesc.instanceMask = 1;
        instanceDesc.instanceID = instance->GetInstanceIndex();

        auto node = instance->GetNode();
        dm::affineToColumnMajor(node->GetLocalToWorldTransformFloat(), instanceDesc.transform);

        instances.push_back(instanceDesc);
    }

    commandList->compactBottomLevelAccelStructs();

    ScopedMarker scopedMarkerTLASUpdate(commandList, "TLAS Update");
    commandList->buildTopLevelAccelStruct(m_topLevelAS, instances.data(), instances.size());
}

void AccelStructManager::RequestRebuild()
{
    m_rebuildAS = true;
}

bool AccelStructManager::NeedsRebuild() const
{
    return m_rebuildAS;
}

nvrhi::rt::AccelStructHandle AccelStructManager::GetTLAS() const
{
    return m_topLevelAS;
}

void AccelStructManager::GetMeshBLASDesc(const donut::engine::MeshInfo& mesh, nvrhi::rt::AccelStructDesc& blasDesc, bool skipTransmissiveMaterials)
{
    blasDesc.isTopLevel = false;
    blasDesc.debugName = mesh.name;

    for (const auto& geometry : mesh.geometries)
    {
        nvrhi::rt::GeometryDesc geometryDesc;
        auto& triangles = geometryDesc.geometryData.triangles;
        triangles.indexBuffer = mesh.buffers->indexBuffer;
        triangles.indexOffset = (mesh.indexOffset + geometry->indexOffsetInMesh) * sizeof(uint32_t);
        triangles.indexFormat = nvrhi::Format::R32_UINT;
        triangles.indexCount = geometry->numIndices;
        triangles.vertexBuffer = mesh.buffers->vertexBuffer;
        triangles.vertexOffset =
            (mesh.vertexOffset + geometry->vertexOffsetInMesh) * sizeof(donut::math::float3) + mesh.buffers->getVertexBufferRange(VertexAttribute::Position).byteOffset;
        triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
        triangles.vertexStride = sizeof(donut::math::float3);
        triangles.vertexCount = geometry->numVertices;
        geometryDesc.geometryType = nvrhi::rt::GeometryType::Triangles;
        geometryDesc.flags = (geometry->material->domain != MaterialDomain::Opaque) ? nvrhi::rt::GeometryFlags::None : nvrhi::rt::GeometryFlags::Opaque;

        blasDesc.bottomLevelGeometries.push_back(geometryDesc);
    }

    if (mesh.skinPrototype)
        blasDesc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace;
    else
        blasDesc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace | nvrhi::rt::AccelStructBuildFlags::AllowCompaction;
}
