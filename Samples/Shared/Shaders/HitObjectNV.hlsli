/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef HIT_OBJECT_NV_HLSLI
#define HIT_OBJECT_NV_HLSLI

// Shader Execution Reordering for Vulkan (VK_NV_ray_tracing_invocation_reorder).
//
// DXC has no SPIR-V lowering for the Shader Model 6.9 dx::HitObject API that the D3D12 permutations use, so the hit
// object instructions of SPV_NV_shader_invocation_reorder are declared here with inline SPIR-V. Only the instructions
// needed to trace, reorder and shade are declared; the numeric values come from the SPIR-V headers (spirv.h).

#define SPV_CAPABILITY_SHADER_INVOCATION_REORDER_NV 5383
#define SPV_STORAGE_CLASS_RAY_PAYLOAD_KHR           5338
#define SPV_OP_TYPE_HIT_OBJECT_NV                   5281
#define SPV_OP_REORDER_THREAD_WITH_HIT_OBJECT_NV    5279
#define SPV_OP_HIT_OBJECT_TRACE_RAY_NV              5260
#define SPV_OP_HIT_OBJECT_EXECUTE_SHADER_NV         5264

// OpTypeHitObjectNV. DeclareHitObjectTypeNV() has to be called before a HitObjectNV variable is declared.
#define HIT_OBJECT_NV_TYPE_ID 0
[[vk::ext_type_def(HIT_OBJECT_NV_TYPE_ID, SPV_OP_TYPE_HIT_OBJECT_NV)]]
void DeclareHitObjectTypeNV();
#define HitObjectNV vk::ext_type<HIT_OBJECT_NV_TYPE_ID>

// Traces the ray and records the resulting hit or miss in the hit object without invoking the closest-hit/miss shader
template<typename PayloadType>
[[vk::ext_capability(SPV_CAPABILITY_SHADER_INVOCATION_REORDER_NV)]]
[[vk::ext_extension("SPV_NV_shader_invocation_reorder")]]
[[vk::ext_instruction(SPV_OP_HIT_OBJECT_TRACE_RAY_NV)]]
void HitObjectTraceRayNV(
    [[vk::ext_reference]] HitObjectNV hitObject,
    RaytracingAccelerationStructure accelerationStructure,
    uint rayFlags,
    uint cullMask,
    uint sbtRecordOffset,
    uint sbtRecordStride,
    uint missIndex,
    float3 origin,
    float tMin,
    float3 direction,
    float tMax,
    [[vk::ext_reference]] [[vk::ext_storage_class(SPV_STORAGE_CLASS_RAY_PAYLOAD_KHR)]] PayloadType payload);

// Hints the driver to reorder the thread so that threads with a similar hit (shader and geometry) run together
[[vk::ext_capability(SPV_CAPABILITY_SHADER_INVOCATION_REORDER_NV)]]
[[vk::ext_extension("SPV_NV_shader_invocation_reorder")]]
[[vk::ext_instruction(SPV_OP_REORDER_THREAD_WITH_HIT_OBJECT_NV)]]
void ReorderThreadWithHitObjectNV([[vk::ext_reference]] HitObjectNV hitObject);

// Invokes the closest-hit or miss shader recorded in the hit object
template<typename PayloadType>
[[vk::ext_capability(SPV_CAPABILITY_SHADER_INVOCATION_REORDER_NV)]]
[[vk::ext_extension("SPV_NV_shader_invocation_reorder")]]
[[vk::ext_instruction(SPV_OP_HIT_OBJECT_EXECUTE_SHADER_NV)]]
void HitObjectExecuteShaderNV(
    [[vk::ext_reference]] HitObjectNV hitObject,
    [[vk::ext_reference]] [[vk::ext_storage_class(SPV_STORAGE_CLASS_RAY_PAYLOAD_KHR)]] PayloadType payload);

// Vulkan counterpart of dx::HitObject::TraceRay + dx::MaybeReorderThread + dx::HitObject::Invoke; takes the arguments
// of TraceRay.
template<typename PayloadType>
void TraceRayReorderNV(
    RaytracingAccelerationStructure accelerationStructure,
    uint rayFlags,
    uint cullMask,
    uint sbtRecordOffset,
    uint sbtRecordStride,
    uint missIndex,
    RayDesc ray,
    inout PayloadType payload)
{
    // The instructions take the payload as a variable in the RayPayloadKHR storage class, like the one DXC creates
    // for TraceRay. It is declared here rather than globally so that it only joins the interface of the entry points
    // that trace: that storage class is not allowed in any-hit shaders.
    [[vk::ext_storage_class(SPV_STORAGE_CLASS_RAY_PAYLOAD_KHR)]] PayloadType hitObjectPayload;
    hitObjectPayload = payload;

    DeclareHitObjectTypeNV();
    HitObjectNV hitObject;
    HitObjectTraceRayNV(hitObject, accelerationStructure, rayFlags, cullMask, sbtRecordOffset, sbtRecordStride, missIndex, ray.Origin, ray.TMin, ray.Direction, ray.TMax, hitObjectPayload);
    ReorderThreadWithHitObjectNV(hitObject);
    HitObjectExecuteShaderNV(hitObject, hitObjectPayload);

    payload = hitObjectPayload;
}

#endif // HIT_OBJECT_NV_HLSLI
