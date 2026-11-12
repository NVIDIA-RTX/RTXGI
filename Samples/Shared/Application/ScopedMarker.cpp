/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "ScopedMarker.h"

ScopedMarker::ScopedMarker(nvrhi::ICommandList* commandList, const char* name) : m_commandList(commandList)
{
    m_commandList->beginMarker(name);
}

ScopedMarker::~ScopedMarker()
{
    m_commandList->endMarker();
}