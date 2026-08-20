# ComfyUI-TRELLIS.2-AMD

## 🔴 NEW (August 2026): AMD ROCm / HIP Support on Windows (including RDNA2)

**This fork adds a working build for AMD GPUs (including RX 6000 series) on Windows**.
The upstream sources are CUDA-only; here they are hipified to
`.hip` files (kept alongside the original `.cu`), and `setup.py` routes to the
correct variant per platform. NVIDIA users can use the upstream repo unchanged.

**Validated on:** AMD Radeon RX 6800 XT (`gfx1030`, RDNA2), Windows 11,
ROCm 7.14, PyTorch 2.13, Python 3.12. It should work on other RDNA2/RDNA3 cards
whose `gfx` target is supported by your ROCm SDK (the build script will auto-detect the
architecture), but only `gfx1030` is tested.

### Prerequisites (AMD)

*   An **AMD Radeon GPU (RDNA2 tested; RDNA1/3/4 should work - see table below)**
*   **Python 3.12** with a recent **pip** (`python -m pip install --upgrade pip`).
    * ⚠️ **Python 3.13 or greater WILL fail at runtime on Windows due to lacking module support**
*   A **ROCm 7.14 SDK + ROCm PyTorch** for your `gfx` target, installed into your
    pip environment. ROCm 7.14 or above is required specifically because `torch_hip.dll` ships
    the HIP `MasqueradingAsCUDA` symbols that earlier Windows builds lack.
*   [OPTIONAL]  If you self compile for Windows, you need **Visual Studio 2022** with the C++ workload and MSVC Version 14.44
    * ⚠️ **MSVC 14.53 or greater WILL fail to compile due to lacking ROCm support**

  
ROCm has been sanity tested and is considered release ready on the vast majority of AMD cards for both
Windows and Linux. You can check for your hardware at [TheRock/SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md).

The required wheels can be found at `https://rocm.nightlies.amd.com/whl-multi-arch/`.
For more details about ROCm wheels, see [TheRock/RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).

| Card(s) | Arch | Device extra | Win sanity-tested |
|---|---|---|---|
| RX 5700 / 5700 XT / 5600 XT | gfx1010 | `device-gfx1010` | ✅ |
| Radeon Pro V520 | gfx1011 | `device-gfx1011` | ✅ |
| RX 5500 / 5500 XT / Pro W5500 | gfx1012 | `device-gfx1012` | ✅ |
| RX 6800 / 6800 XT / 6900 XT / 6950 XT / Pro W6800 | gfx1030 | `device-gfx1030` | ✅ |
| RX 6700 / 6700 XT / 6750 XT | gfx1031 | `device-gfx1031` | ✅ |
| RX 6600 / 6600 XT / 6650 XT / Pro W6600 | gfx1032 | `device-gfx1032` | ✅ |
| Van Gogh iGPU (Steam Deck) | gfx1033 | `device-gfx1033` | ✅ |
| RX 6400 / 6500 XT | gfx1034 | `device-gfx1034` | ✅ |
| Rembrandt iGPU (680M) | gfx1035 | `device-gfx1035` | ✅ |
| Raphael iGPU | gfx1036 | `device-gfx1036` | ✅ |
| RX 7900 XTX / 7900 XT / 7900 GRE / Pro W7900 | gfx1100 | `device-gfx1100` | ✅ |
| RX 7800 XT / 7700 XT / Pro W7700 | gfx1101 | `device-gfx1101` | ✅ |
| RX 7600 / 7600 XT | gfx1102 | `device-gfx1102` | ✅ |
| Ryzen 7040/8040 (780M) | gfx1103 | `device-gfx1103` | ✅ |
| Ryzen AI 300 (890M, Strix Point) | gfx1150 | `device-gfx1150` | ✅ |
| Ryzen AI Max+ (Strix Halo) | gfx1151 | `device-gfx1151` | ✅ |
| Ryzen AI 7 350 | gfx1152 | `device-gfx1152` | ✅ |
| Radeon 820M iGPU (Krackan) | gfx1153 | `device-gfx1153` | ⚠️ not tested |
| RX 9060 / 9060 XT | gfx1200 | `device-gfx1200` | ⚠️ not tested |
| RX 9070 / 9070 XT / AI PRO R9700 | gfx1201 | `device-gfx1201` | ⚠️ not tested |
| *All supported* | — | `device-all` | — |

> ⚠️ = installs but **not** sanity-tested on Windows by TheRock (as of 2026-08-18); device enumeration or kernel launch may fail at runtime.
> ✅ = sanity-tested on Windows by TheRock. Note this is **TheRock's** ROCm sanity test, not validation of *this* pipeline.
> This project is developed and validated end-to-end only on **gfx1030 (RX 6800 XT)**; all other archs are best-effort — the wheels contain compiled code for them, but the full TRELLIS.2 pipeline is unverified there.

## Step 1: Set up a clean environment

Always install into a **fresh Python 3.12 virtual environment** - this avoids
conflicts with any existing PyTorch or CUDA packages, which can silently break
the ROCm install.

```powershell
# NOTE: Some ComfyUI extensions might expect venv to be in the ComfyUI directory
cd C:/path/to/your/ComfyUI # wherever you want to place the environment

# If you already have ComfyUI installed and working, backup your venv
if (Test-Path .\venv) { Rename-Item .\venv venv_backup }

# Create fresh venv and confirm
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python --version   # confirm 3.12.x
```

> **Python 3.12 is required.** 3.13 has ecosystem gaps (open3d and others lack
> 3.13 Windows wheels). 3.11 may work but is untested.

## Step 2: Install ROCm and torch dependencies

Make sure your fresh Python 3.12 venv is activated. Choose **one** of the two
methods below.

---

### Option A - Automatic install from the ROCm index (recommended)

Pulls the pinned wheels directly from AMD's index. Replace `device-gfx1030` with
the arch for your card (see the hardware table above), or use `device-all` for a
larger install that covers every AMD GPU.

```powershell
pip install --pre --extra-index-url https://rocm.nightlies.amd.com/whl-multi-arch/ `
  "torch[device-gfx1030]==2.12.0+rocm7.14.0a20260624" `
  "torchvision[device-gfx1030]==0.27.0+rocm7.14.0a20260624" `
  "torchaudio==2.11.0+rocm7.14.0a20260624" `
  "rocm[libraries,devel,device-gfx1030]==7.14.*"
```


### Option B - Manual download and install

Use this if the automatic install fails, the nightly has been pruned, or you
want an offline/archived copy. Download the wheels below into a folder (e.g.
`D:\rocm\gfx1030\`), then install from that folder.

**Recommended and tested wheel set** (gfx103x / RDNA2 / RX 6800 XT, Windows 11,
Python 3.12), from `https://rocm.nightlies.amd.com/whl-multi-arch/`:

```
# Generic wheels (work across all GPUs)
rocm-7.14.0a20260624.tar.gz
rocm_sdk_core-7.14.0a20260624-py3-none-win_amd64.whl
rocm_sdk_devel-7.14.0a20260624-py3-none-win_amd64.whl
rocm_sdk_libraries-7.14.0a20260624-py3-none-win_amd64.whl
torch-2.12.0+rocm7.14.0a20260624-cp312-cp312-win_amd64.whl
torchvision-0.27.0+rocm7.14.0a20260624-cp312-cp312-win_amd64.whl
torchaudio-2.11.0+rocm7.14.0a20260624-cp312-cp312-win_amd64.whl

# Device-specific wheels (gfx1030 kernels - swap gfx1030 for your arch)
rocm_sdk_device_gfx1030-7.14.0a20260624-py3-none-win_amd64.whl
amd_torch_device_gfx1030-2.12.0+rocm7.14.0a20260624-cp312-cp312-win_amd64.whl
amd_torchvision_device_gfx1030-0.27.0+rocm7.14.0a20260624-cp312-cp312-win_amd64.whl
```

Once you have all the wheels downloaded, run the following command (ensure to change the
paths and swap `gfx1030` for your device's arch throughout):

```powershell
pip install --find-links D:\rocm\gfx1030
  "torch==2.12.0+rocm7.14.0a20260624" `
  "torchvision==0.27.0+rocm7.14.0a20260624" `
  "torchaudio==2.11.0+rocm7.14.0a20260624" `
  "rocm==7.14.0a20260624" `
  "rocm-sdk-core==7.14.0a20260624" `
  "rocm-sdk-libraries==7.14.0a20260624" `
  "rocm-sdk-devel==7.14.0a20260624" `
  "rocm-sdk-device-gfx1030==7.14.0a20260624" `
  "amd-torch-device-gfx1030==2.12.0+rocm7.14.0a20260624" `
  "amd-torchvision-device-gfx1030==0.27.0+rocm7.14.0a20260624"
```

## Step 3: Verify your install

Confirm every package installed at the correct, matching version before moving on.

**Check the installed versions:**

```powershell
pip list | Select-String "torch|rocm"
```

Expected output (all sharing the `7.14.0a20260624` suffix):
```
amd-torch-device-gfx1030       2.12.0+rocm7.14.0a20260624
amd-torchvision-device-gfx1030 0.27.0+rocm7.14.0a20260624
rocm                           7.14.0a20260624
rocm-bootstrap                 0.1.0
rocm-sdk-core                  7.14.0a20260624
rocm-sdk-devel                 7.14.0a20260624
rocm-sdk-device-gfx1030        7.14.0a20260624
rocm-sdk-libraries             7.14.0a20260624
torch                          2.12.0+rocm7.14.0a20260624
torchaudio                     2.11.0+rocm7.14.0a20260624
torchvision                    0.27.0+rocm7.14.0a20260624
```

Comfirm torch works in Python:
```powershell
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected output (all sharing the `7.14.0a20260624` suffix):
```
2.12.0+rocm7.14.0a20260624 7.14.60850 True AMD Radeon RX 6800 XT
```

## Step 4: Clone this repo, and install requirements.txt into your venv
This is a modified version of the `requirements.txt` that ships with ComfyUI main.
It includes the Python packages required for ComfyUI, TRELLIS.2, and the
ComfyUI-Trellis2 extension.

> **Important:** this requirements file deliberately **excludes** `torch`,
> `torchvision`, and `torchaudio` because you need the custom ROCm builds you
> installed in Steps 2-3. 

```powershell
cd D:\
git clone https://github.com/dmonkman/ComfyUI-TRELLIS.2-AMD.git

cd ComfyUI-TRELLIS.2-AMD
pip install -r requirements.txt
```

## Step 5: Install the custom extension wheels

The four native extensions (CuMesh, FlexGEMM, o_voxel, nvdiffrast) are provided
as prebuilt wheels in the `wheels/` folder. They were **built by GitHub Actions
CI** from the tagged source of each repo, so you can audit the build logs and
verify them against the source rather than trusting an opaque binary.

Each wheel is a **fat multi-arch build** containing compiled device code for all
supported RDNA/CDNA architectures (see the hardware table), so the same wheel
works on any listed card.

```powershell
pip install `
  .\wheels\cumesh-1.0-cp312-cp312-win_amd64.whl `
  .\wheels\flex_gemm-1.0.0-cp312-cp312-win_amd64.whl `
  .\wheels\o_voxel-0.0.1-cp312-cp312-win_amd64.whl `
  .\wheels\nvdiffrast-0.4.0-cp312-cp312-win_amd64.whl
```

> Prefer to build them yourself? Each repo has a `build-wheel.yaml` GitHub Actions
> workflow, or you can build locally with its `setup_build_env.ps1`. See
> [Building from source](#building-from-source).

## Step 6: Install ComfyUI-Manager

ComfyUI-Manager provides the node-management UI and is needed to install and
update custom nodes. Clone it into your `custom_nodes` folder:

```powershell
cd D:\dev\comfyui\custom_nodes
git clone https://github.com/Comfy-Org/ComfyUI-Manager.git comfyui-manager
```

> **"Blocked by policy" on startup?** ComfyUI's security level can restrict
> Manager's network features. If you need them, open **ComfyUI settings →
> Security level** (or edit `user\default\ComfyUI-Manager\config.ini`) and set a
> less restrictive level. This only affects Manager; the TRELLIS.2 pipeline runs
> regardless.

## Step 7: Install the ComfyUI-Trellis2 custom nodes

Either install via the ComfuUI-Manager, or clone via git into `custom_nodes`:

```powershell
cd D:\dev\comfyui\custom_nodes
git clone https://github.com/VisualBruno/ComfyUI-Trellis2.git comfyui-trellis2
```

> This repo provides the ComfyUI *nodes*; the ROCm-compatible native extensions
> and pinned dependencies are what you installed in Steps 2-5. If a node fails to
> load with a missing-module error, it's almost always a Python dependency from
> `requirements.txt` (Step 4).

Restart ComfyUI. The **Trellis2** nodes should now appear in the node menu
(right-click → Add Node → search "Trellis2").

## Step 8: Run the bundled Simple workflow

ComfyUI-Trellis2 ships with a **Simple** workflow that exercises the full
image-to-3D pipeline. Running it verifies your entire install end-to-end and
triggers the first-run model download.

1. Start ComfyUI:
```powershell
   cd D:\dev\comfyui
   python main.py
```
2. Open the UI at `http://127.0.0.1:8188`.
3. Load the Simple workflow: **Workflow → Browse Templates → ComfyUI-Trellis2 →
   Simple** (or drag the `.json` from
   `custom_nodes\ComfyUI-Trellis2\workflows\` onto the canvas).
4. On the **TRELLIS.2 model loader** node, set the attention backends to `sdpa`
   (see the note below — this is required on RDNA2).
5. Queue the workflow with the provided sample image.

**First run downloads the models** (several GB, from Hugging Face) to your HF
cache — this is slow and hands-off. Subsequent runs reuse the cache.

> **Model VRAM:** on 16 GB cards (RX 6800 XT etc.), start with the **FP8**
> model variant (`visualbruno/TRELLIS.2-4B-FP8`) rather than the full 4B — the
> full model may OOM. Keep `low_vram` enabled.

If it produces a textured GLB, **your install is complete and working.**

### Dependencies installed by the workflow

The wrapper pulls a few extra Python packages on first use. They're already in
`requirements.txt` (Step 4), but for reference these are what the Simple
workflow touches at runtime:

- `pymeshlab`, `trimesh`, `open3d`, `pyfqmr` — mesh processing / decimation
- `rembg` + `onnxruntime` (CPU) — background removal on the input image
- `opencv-python` — image I/O in the post-process nodes
- `plyfile` — PLY mesh read/write

> If a node fails to load with `ModuleNotFoundError`, install the named package
> and restart — it's a runtime dependency the requirements step didn't cover on
> your system.


### What was changed for HIP (CumMesh)

Mostly mechanical (`hipify-perl` over `src/**` and `third_party/cubvh/**`),
plus manual cleanup that hipify does not cover:

*   **Source router** in `setup.py` compiles `.hip` on HIP, `.cu` on CUDA.
*   **`::cuda::std::` → `::std::`** (libcu++ constructs are not remapped).
*   **rocPRIM radix-sort decomposer** returns `::rocprim::tuple<...>` (not
    `std::tuple`) for `hipcub::DeviceRadixSort::SortPairs`.
*   **`thrust::cuda` → `thrust::hip`** (thrust execution-policy namespace).
*   Reverted hipify's over-rename of the project's own `cubvh` namespace.
*   **Eigen include path** points at the repo root (Eigen is vendored there).

---

## Acknowledgements

This package builds upon and integrates code from several excellent open-source libraries. We would like to express our gratitude to the authors of:

*   **[cubvh](https://github.com/ashawkey/cubvh)**: For the high-performance CUDA BVH acceleration toolkit.
*   **[xatlas](https://github.com/jpcy/xatlas)**: For the robust UV parameterization and atlas packing library.
*   **[Eigen](https://eigen.tuxfamily.org/)**: For the C++ template library for linear algebra, used by the cubvh backend.
*   **[pamo](https://github.com/SarahWeiii/pamo)**: For the reference implementation of the GPU parallel edge collapse algorithm used in our mesh simplification module.

## License

[MIT License](LICENSE)