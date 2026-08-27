# ComfyUI-TRELLIS.2-AMD-Lite

**A front-end-agnostic backend for [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) on AMD GPUs via ROCm.**

This repo gets the TRELLIS.2 image-to-3D backend running on AMD hardware: the pinned ROCm PyTorch stack plus the four native extension wheels (CuMesh, FlexGEMM, o_voxel, nvdiffrast) that TRELLIS.2 needs. Once it's installed you can drive it from whatever front-end you like. For the ComfyUI nodes and a full image-to-3D walkthrough, use the [ComfyUI-Trellis2-AMD](https://github.com/dmonkman/ComfyUI-Trellis2-AMD) extension instead. It's self-contained and repeats these backend steps.

The upstream sources are CUDA-only. Here they're hipified to `.hip` (kept alongside the original `.cu`), and each `setup.py` routes to the correct variant per platform.

The wheels are built by GitHub Actions CI from tagged source, with build-provenance attestations, so you can audit them against the source rather than trusting an opaque binary.

## Hardware support

Validated end-to-end on a **Radeon RX 6800 XT (`gfx1030`, RDNA2)**, Windows 11 and Ubuntu 24.04 (**22.04 will not work**), Python 3.12, ROCm 10.0.0, PyTorch 2.13. The wheels are fat multi-arch builds covering RDNA1-4 (and gfx1250), so they should load on any of the cards below, but only `gfx1030` is tested here.

| Family | Example cards | gfx targets |
| --- | --- | --- |
| RDNA1 | RX 5500-5700 XT | gfx1010-gfx1012 |
| RDNA2 | RX 6400-6950 XT | gfx1030-gfx1036 |
| RDNA3 | RX 7600-7900 XTX, Ryzen APUs | gfx1100-gfx1103, gfx1150-gfx1153 |
| RDNA4 | RX 9060-9070 XT | gfx1200, gfx1201 |

If a card here is listed as working, that means its `gfx` target is in the build and ROCm sanity-tests it. It does not mean this pipeline is validated on it. Check your card against [TheRock/SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md).

## Prerequisites

- An **AMD Radeon GPU** whose `gfx` target is supported by ROCm 10.0 (see above).
- **Python 3.12** with a recent pip.
  - Python 3.13+ currently fails at runtime on Windows (ecosystem gaps, e.g. `open3d`). Use 3.12.
- The **ROCm 10.0 PyTorch** stack, installed via pip in Step 2 below.

---

## Step 1: Fresh virtual environment

Install into a clean Python 3.12 venv to avoid clashing with any existing PyTorch/CUDA packages, which can silently break the ROCm install.

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python --version
```

On Linux:

```bash
python3.12 -m venv venv
source venv/bin/activate
python --version
```

## Step 2: Install the ROCm 10.0 PyTorch stack

One pip command pulls the pinned stable wheels from AMD's index. `device-all` covers every supported GPU. Swap it for your specific arch (e.g. `device-gfx1030`) for a smaller download.

Windows:
```powershell
pip install --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ `
  "torch[device-gfx1030]==2.13.0+rocm10.0.0" `
  "torchvision[device-gfx1030]==0.28.0+rocm10.0.0" `
  "torchaudio==2.11.0.2+rocm10.0.0" `
  "rocm[libraries,device-gfx1030]==10.0.0"
```

Linux:
```bash
pip install --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1030]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1030]==0.28.0+rocm10.0.0" \
  "torchaudio==2.11.0.2+rocm10.0.0" \
  "rocm[libraries,device-gfx1030]==10.0.0"
```

<details>
<summary><strong>Fallback: manual / offline wheel install</strong></summary>

If the index install fails (proxy, air-gapped machine, etc.), download the wheels for your arch from `https://stable.repo.amd.com/rocm/whl-next/` and install from a local folder. Replace `gfx1030` with your target throughout.

Wheel set for gfx1030 / Python 3.12 / Windows:

```
torch-2.13.0+rocm10.0.0-cp312-cp312-win_amd64.whl
torchvision-0.28.0+rocm10.0.0-cp312-cp312-win_amd64.whl
torchaudio-2.11.0.2+rocm10.0.0-cp312-cp312-win_amd64.whl
rocm_sdk_core-10.0.0-py3-none-win_amd64.whl
rocm_sdk_libraries-10.0.0-py3-none-win_amd64.whl
rocm_sdk_device_gfx1030-10.0.0-py3-none-win_amd64.whl
amd_torch_device_gfx1030-2.13.0+rocm10.0.0-cp312-cp312-win_amd64.whl
amd_torchvision_device_gfx1030-0.28.0+rocm10.0.0-cp312-cp312-win_amd64.whl
```

```powershell
pip install --find-links D:\rocm\gfx1030 `
  "torch==2.13.0+rocm10.0.0" `
  "torchvision==0.28.0+rocm10.0.0" `
  "torchaudio==2.11.0.2+rocm10.0.0" `
  "rocm==10.0.0" `
  "rocm-sdk-core==10.0.0" `
  "rocm-sdk-libraries==10.0.0" `
  "rocm-sdk-device-gfx1030==10.0.0" `
  "amd-torch-device-gfx1030==2.13.0+rocm10.0.0" `
  "amd-torchvision-device-gfx1030==0.28.0+rocm10.0.0"
```

</details>

### Verify Installations

Windows
```powershell
pip list | Select-String "torch|rocm"
```

Linux
```bash
pip list | grep "rocm"
```

Every package should share the `10.0.0` version:

```
amd-torch-device-gfx1030       2.13.0+rocm10.0.0
amd-torchvision-device-gfx1030 0.28.0+rocm10.0.0
rocm                           10.0.0
rocm-sdk-core                  10.0.0
rocm-sdk-devel                 10.0.0
rocm-sdk-device-gfx1030        10.0.0
rocm-sdk-libraries             10.0.0
torch                          2.13.0+rocm10.0.0
torchaudio                     2.11.0.2+rocm10.0.0
torchvision                    0.28.0+rocm10.0.0
```

Confirm torch sees the GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` should be `True` and the device name your card, e.g.:

```
2.13.0+rocm10.0.0 10.0.xxxxx True AMD Radeon RX 6800 XT
```

## Step 3: Install custom wheels

Clone or download this repo. The four native extensions ship as prebuilt fat multi-arch wheels under `wheels/`, one folder per OS and Python version. Install the set matching your platform:

Windows
```powershell
cd C:\path\to\your\ComfyUI-TRELLIS.2-AMD-Lite
pip install `
  ".\wheels\Windows\Python3.12\cumesh-1.0+rocm10.0-cp312-cp312-win_amd64.whl" `
  ".\wheels\Windows\Python3.12\flex_gemm-1.0.0+rocm10.0-cp312-cp312-win_amd64.whl" `
  ".\wheels\Windows\Python3.12\o_voxel-0.0.1+rocm.10.0-cp312-cp312-win_amd64.whl" `
  ".\wheels\Windows\Python3.12\nvdiffrast-0.4.0+rocm10.0-cp312-cp312-win_amd64.whl"
```

On Linux, use `wheels/Linux/Python3.12/` and the `linux_x86_64` wheels
```bash
cd ~/path/to/your/ComfyUI-TRELLIS.2-AMD-Lite
pip install \
  ./wheels/Linux/Python3.12/cumesh-1.0+rocm10.0-cp312-cp312-linux_x86_64.whl   \
  ./wheels/Linux/Python3.12/flex_gemm-1.0.0+rocm10.0-cp312-cp312-linux_x86_64.whl \
  ./wheels/Linux/Python3.12/o_voxel-0.0.1+rocm.10.0-cp312-cp312-linux_x86_64.whl  \
  ./wheels/Linux/Python3.12/nvdiffrast-0.4.0+rocm10.0-cp312-cp312-linux_x86_64.whl 
```

> Match these filenames to what's actually in your `wheels/` folder. FlexGEMM in particular carries a `+rocm10.0` local version tag, so its filename differs from the others.

<details>
<summary><strong>Fallback: build the wheels yourself</strong></summary>

Each native extension has its own repo with a build workflow. Use the `setup_build_env.ps1` script included with the CuMesh repo before compiling on Windows:

- [CuMesh-ROCm](https://github.com/dmonkman/CuMesh-ROCm)
- [FlexGEMM-ROCm](https://github.com/dmonkman/FlexGEMM-ROCm)
- [TRELLIS.2-ROCm](https://github.com/dmonkman/TRELLIS.2-ROCm) (contains o_voxel)
- [nvdiffrast-ROCm](https://github.com/dmonkman/nvdiffrast-ROCm)

The GitHub Actions workflow in each cross-compiles a multi-arch wheel on a GPU-less runner using TheRock's pip-installed ROCm toolchain.

</details>

## Step 4: Install requirements.txt

Windows and Linux
```bash
pip install -r requirements.txt
```

Ensure that all of the requirements are installed. At this point the TRELLIS.2 backend dependencies (o_voxel, CuMesh, FlexGEMM, and nvdiffrast) are installed and GPU-accelerated on AMD. You should be able to follow most CUDA written guidse from this point, but ensure you don't overwrite the custom dependencies (ex. torch). This way, you can drive it from your preferred front-end. For ComfyUI specifically, use the [ComfyUI-Trellis2-AMD](https://github.com/dmonkman/ComfyUI-Trellis2-AMD) extension.

## License

[MIT License](LICENSE). The bundled native extensions retain their own upstream licenses. Note that **nvdiffrast is under the NVIDIA Source Code License (non-commercial)** which applies to any pipeline that depends on it.

## Acknowledgements

- [visualbruno/ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) and [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2), the upstream wrapper and model this builds on.
- [cubvh](https://github.com/ashawkey/cubvh), [xatlas](https://github.com/jpcy/xatlas), [Eigen](https://eigen.tuxfamily.org/), and [pamo](https://github.com/SarahWeiii/pamo), native libraries used by the extensions.
- The "Blackwell Fix" from [ThatButters/trellis2-blackwell-fix](https://github.com/ThatButters/trellis2-blackwell-fix), and the ROCm/ComfyUI Discord community.
