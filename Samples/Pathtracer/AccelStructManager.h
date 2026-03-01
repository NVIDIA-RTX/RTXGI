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

#include <donut/engine/Scene.h>
#include <nvrhi/nvrhi.h>

class AccelStructManager
{
public:
    void Build(nvrhi::IDevice* device, nvrhi::ICommandList* commandList, donut::engine::Scene& scene, bool enableTransmission);
    void UpdateTLAS(nvrhi::ICommandList* commandList, donut::engine::Scene& scene, bool enableTransmission);
    void RequestRebuild();

    bool NeedsRebuild() const;
    nvrhi::rt::AccelStructHandle GetTLAS() const;

private:
    static void GetMeshBLASDesc(const donut::engine::MeshInfo& mesh, nvrhi::rt::AccelStructDesc& blasDesc, bool skipTransmissiveMaterials);

    nvrhi::rt::AccelStructHandle m_topLevelAS;
    bool m_rebuildAS = true;
};