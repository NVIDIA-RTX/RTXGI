# Quick Start Guide

This document lists the necessary steps to get up and running with the RTXGI SDK sample projects.

For programming/integration guides specific to each technique, see the individual [NRC][NrcGuide] and [SHaRC][SharcGuide] guides.

## Build steps
Assuming the tools specified in the [prerequisites][Prereq] section are available:

1. You may require permission to run PowerShell scripts. Open PowerShell and invoke `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

2. Clone the project recursively to a preferred location.

3. Configure and then generate the solution using CMake GUI (or the CLI). The `source` directory should be the root of the repository and `build` directory should be a new directory located in the repository root.

4. The newly generated solution `RTXGI2Samples.sln` should be located in the `build` directory. Open it and run the `pathtracer`(default) or `indirect diffuse` projects. Optionally, use the debug command line argument `-vk` to run the NVRHI Vulkan rendering backend version.
Other command line arguments: `-nrc` / `-sharc` (select the caching technique), `-scene <name>` (a `*.scene.json` in `Assets/Media`), `-camera <index>`, `-noaccumulation`, `-noautoexposure`, `-fullscreen`, `-width <w>` / `-height <h>`.

## The pathtracer sample
This showcases a trivial path tracer that relies on either NRC or SHaRC to terminate early into the respective radiance/irradiance cache for an improved signal.

![overview](Figures/quickstart_ui.png)

**Generic settings.** New scenes can be loaded from here via a JSON file that specifies GLTF assets and user-defined properties for common constructs such as lights and camera. Additionally, four more settings that affect the pathtracer and either of the radiance caching techniques can be toggled here.

**Path tracing settings.** This section addresses typical pathtracer settings such as the number of bounces, samples per pixel, a global dial for altering material roughness, and a list of debug views that ensure the scene data is correctly represented (normals, worldspace positions, etc.). From this panel, the radiance/irradiance caching modes as well as the denoising modes can be selected. `Shader Execution Reordering` switches to the shader permutations that reorder ray tracing threads by their hit before shading; it is available on D3D12 when the device and driver report Shader Model 6.9, on Vulkan when they support `VK_NV_ray_tracing_invocation_reorder`, and is grayed out otherwise. It does not change the image.

**NRC settings.** In this section, NRC can be toggled, fine-tuned, as well as debugged visually via the `Resolve Mode`. For further details see the [in-depth NRC guide][NrcGuide]. 

**SHaRC settings.** These provide a way to toggle the tech, manually invoke a clearing of the cache, fine-tune factors that contribute to the hash-grid data, as well as visually inspect the direct contents of the cache via the `Enable Debug` option. For further details see the [in-depth SHaRC guide][SharcGuide].

**Lighting.** This section allows for modifying the initial light data specified in the JSON scene file.

**Tone mapping.** Post processing section. The `Operator` selects how the linear radiance is mapped to the display: `Linear`, `Reinhard` (extended Reinhard, with a `White Point` that sets the smallest luminance mapped to white), `ACES` (RRT + ODT fit) or `AgX` (with the `Default`, `Golden` and `Punchy` looks); `Clamp` saturates the result. `Auto Exposure` is on by default and meters the image, with `Lock` to freeze the adapted value, `Reset` to re-adapt instantly, an `Exposure Compensation (EV)` bias and the metering parameters under `Auto Exposure Settings`; untick it (or launch with `-noautoexposure`) for a fixed manual `Exposure (EV)`, which is preferable when comparing a caching technique against the reference since adaptation hides brightness differences. None of these settings reset accumulation.

## The indirect diffuse sample
This showcases a common hybrid rendering scenario: the visible surfaces come from a rasterized G-buffer and are lit by a deferred pass, while NRC or SHaRC provide only the indirect diffuse lighting, which is denoised and composited into the deferred pass. The sample starts with both caches off; select one in the UI or launch with `-nrc` / `-sharc`.

Every frame rasterizes the G-buffer (opaque and alpha-tested geometry) and a cascaded shadow map for the sun light. The selected cache then produces the indirect diffuse radiance reflected by each visible surface, and the deferred lighting pass adds it to the direct lighting and emissives it computes from the G-buffer. Upscaling, exposure, and tone mapping follow as in the pathtracer sample.

**Notable settings** 
- `Indirect diffuse` selects the technique: `Deferred only` (rasterized direct lighting with a constant sky ambient term, no ray tracing), `NRC` or `SHaRC`. 
- `Output View` switches between the `Composite` image and the indirect diffuse signal on its own, traced without the albedo of the visible surface so that the lighting can be inspected independently of the surface color.

[NrcGuide]: NrcGuide.md
[SharcGuide]: SharcGuide.md
[Prereq]: ../Readme.md/#prerequisites
