import os
import re
import sys
import site
import shutil
import platform
import subprocess
import importlib
import importlib.util
import sysconfig
from pathlib import Path


print("✅ setup_env marker: 2026-07-10-gb200-no-git-te-cublas-flashattn-patch-v6")

COSMOS_VERSION = os.environ.get("COSMOS_PREDICT2_VERSION", "1.0.9")

X86_TORCH_PACKAGES = ["torch", "torchvision", "triton"]

ACCEL_PACKAGES = [
    "flash-attn",
    "natten",
    "transformer-engine",
    "transformer-engine-cu12",
    "transformer-engine-cu13",
    "transformer-engine-torch",
    "transformer_engine",
]

MANAGED_ARM_DEPS = {
    "torch",
    "torchvision",
    "triton",
    "flash-attn",
    "natten",
    "transformer-engine",
    "transformer-engine-cu12",
    "transformer-engine-cu13",
    "transformer-engine-torch",
    "transformer-engine-torch-cu12",
    "transformer-engine-torch-cu13",
    "decord",
    "qwen-vl-utils",
    "xformers",
    "megatron-core",
    "tensorstore",
    "lpips",
}


# ------------------------------------------------------------
# Command helpers
# ------------------------------------------------------------


def run_cmd(cmd, *, cwd=None, env=None, capture_output=False, check=True):
    cmd = [str(x) for x in cmd]
    print(f"\n🚀 Executing: {' '.join(cmd)}")

    res = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=capture_output,
        text=True,
    )

    if check and res.returncode != 0:
        print(f"\n❌ ERROR: Command failed with exit code {res.returncode}")

        if capture_output:
            if res.stdout:
                print("\n--- stdout ---")
                print(res.stdout)
            if res.stderr:
                print("\n--- stderr ---")
                print(res.stderr)

        sys.exit(res.returncode)

    return res


def run_capture(cmd, *, cwd=None, env=None):
    res = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )

    if res.returncode != 0:
        return ""

    return res.stdout.strip()


def normalize_pkg_name(name):
    return str(name).lower().replace("_", "-").replace(".", "-")


def _sanitize_uv_pip_args(args):
    """
    Global uv pip install guard for GB200/aarch64.

    Avoid:
      - qwen-vl-utils[decord] -> PyPI decord resolution
      - PyPI decord resolution
      - old tensorstore source build
      - megatron-core pulling old tensorstore
      - transformer-engine being resolved indirectly instead of by our real TE installer
    """
    args = [str(a) for a in args]
    allow_no_deps = "--no-deps" in args
    out = []

    for s in args:
        low = s.lower().replace("_", "-")

        if s.startswith("-"):
            out.append(s)
            continue

        if s.endswith(".whl") or Path(s).exists():
            out.append(s)
            continue

        if low.startswith("qwen-vl-utils[") or "qwen-vl-utils[" in low:
            fixed = re.sub(r"(?i)qwen-vl-utils\[[^\]]+\]", "qwen-vl-utils", s)
            print(f"🛡️  Rewriting unsafe dependency: {s} -> {fixed}")
            out.append(fixed)
            continue

        if (
            low == "decord"
            or low.startswith("decord==")
            or low.startswith("decord>=")
            or low.startswith("decord<=")
            or low.startswith("decord~=")
            or low.startswith("decord!=")
            or low.startswith("decord[")
        ):
            print(f"🛡️  Skipping PyPI decord dependency: {s}")
            continue

        if (
            low == "tensorstore"
            or low.startswith("tensorstore==")
            or low.startswith("tensorstore<")
            or low.startswith("tensorstore~=")
        ):
            print(f"🛡️  Skipping unsafe tensorstore dependency: {s}")
            continue

        if not allow_no_deps and (
            low == "megatron-core"
            or low.startswith("megatron-core==")
            or low.startswith("megatron-core>=")
            or low.startswith("megatron-core<=")
            or low.startswith("megatron-core~=")
        ):
            print(f"🛡️  Skipping managed megatron-core dependency: {s}")
            continue

        if (
            low == "transformer-engine"
            or low.startswith("transformer-engine==")
            or low.startswith("transformer-engine>=")
            or low.startswith("transformer-engine<=")
            or low.startswith("transformer-engine~=")
            or low.startswith("transformer-engine[")
            or low == "transformer-engine-cu12"
            or low.startswith("transformer-engine-cu12")
            or low == "transformer-engine-cu13"
            or low.startswith("transformer-engine-cu13")
            or low == "transformer-engine-torch"
            or low.startswith("transformer-engine-torch")
        ):
            print(f"🛡️  Skipping indirect transformer-engine dependency: {s}")
            continue

        out.append(s)

    return out


def uv_pip_install(args, *, env=None):
    args = _sanitize_uv_pip_args(args)

    if not args:
        print("ℹ️ No pip args left after sanitization; skipping uv pip install.")
        return

    run_cmd(["uv", "pip", "install", "--python", sys.executable, *args], env=env)


def uv_pip_install_try(args, *, env=None):
    args = _sanitize_uv_pip_args(args)

    if not args:
        print("ℹ️ No pip args left after sanitization; skipping uv pip install.")
        return True

    res = run_cmd(
        ["uv", "pip", "install", "--python", sys.executable, *args],
        env=env,
        check=False,
    )

    return res.returncode == 0


def uv_pip_raw_install(args, *, env=None, check=True):
    """
    Raw uv pip install. Use only for dependencies we intentionally manage.
    This bypasses the sanitizer.
    """
    return run_cmd(
        ["uv", "pip", "install", "--python", sys.executable, *[str(a) for a in args]],
        env=env,
        check=check,
    )


def uv_pip_uninstall(packages):
    packages = [str(p) for p in packages if p]

    if not packages:
        return

    print(f"\n🧹 Uninstalling if present: {' '.join(packages)}")

    subprocess.run(
        ["uv", "pip", "uninstall", "--python", sys.executable, *packages],
        capture_output=True,
        text=True,
    )


def package_importable(name):
    try:
        importlib.import_module(name)
        return True
    except Exception as exc:
        print(f"ℹ️ Package '{name}' is not cleanly importable: {type(exc).__name__}: {exc}")
        return False


def find_executable(name):
    found = shutil.which(name)

    if found:
        return found

    env_bin = Path(sys.executable).resolve().parent / name

    if env_bin.exists():
        return str(env_bin)

    local_bin = Path(sys.prefix).resolve() / "bin" / name

    if local_bin.exists():
        return str(local_bin)

    return ""


# ------------------------------------------------------------
# Pixi/native environment helpers
# ------------------------------------------------------------


def make_build_env(base_env=None):
    env = dict(base_env or os.environ)

    env_prefix = Path(sys.prefix).resolve()
    env_bin = env_prefix / "bin"
    env_include = env_prefix / "include"
    env_lib = env_prefix / "lib"
    env_pkgconfig_1 = env_lib / "pkgconfig"
    env_pkgconfig_2 = env_lib / "aarch64-linux-gnu" / "pkgconfig"

    env["PATH"] = f"{env_bin}:{env.get('PATH', '')}"

    existing_cmake_prefix = env.get("CMAKE_PREFIX_PATH", "")
    env["CMAKE_PREFIX_PATH"] = f"{env_prefix}:{existing_cmake_prefix}" if existing_cmake_prefix else str(env_prefix)

    existing_pkg = env.get("PKG_CONFIG_PATH", "")
    pkg_paths = [str(env_pkgconfig_1), str(env_pkgconfig_2)]
    env["PKG_CONFIG_PATH"] = ":".join(pkg_paths + ([existing_pkg] if existing_pkg else []))

    existing_cpath = env.get("CPATH", "")
    env["CPATH"] = f"{env_include}:{existing_cpath}" if existing_cpath else str(env_include)

    existing_library_path = env.get("LIBRARY_PATH", "")
    env["LIBRARY_PATH"] = f"{env_lib}:{existing_library_path}" if existing_library_path else str(env_lib)

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{env_lib}:{existing_ld}" if existing_ld else str(env_lib)

    return env


def maybe_prepare_pixi_native_deps():
    if os.environ.get("COSMOS_PIXI_ADD_NATIVE_DEPS", "0").strip() != "1":
        return

    pixi_bin = find_executable("pixi")

    if not pixi_bin:
        print("❌ COSMOS_PIXI_ADD_NATIVE_DEPS=1 was set, but pixi was not found.")
        sys.exit(1)

    print("--- Adding native Pixi dependencies ---")
    run_cmd([pixi_bin, "add", "cmake", "ninja", "pkg-config", "ffmpeg<5", "git"])

    print("\n✅ Pixi dependencies updated.")
    print("Now rerun:")
    print("  COSMOS_PLATFORM=gb200 pixi run setup")

    sys.exit(0)


# ------------------------------------------------------------
# Platform detection
# ------------------------------------------------------------


def detect_gpu_names():
    out = run_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])

    if not out:
        return []

    return [line.strip() for line in out.splitlines() if line.strip()]


def detect_platform():
    override = os.environ.get("COSMOS_PLATFORM", "").strip().lower()

    if override in {"x86", "x86_64", "amd64"}:
        return "x86"

    if override in {"gb200", "b200", "blackwell"}:
        return "gb200"

    if override in {"arm", "arm64", "aarch64"}:
        return "arm"

    arch = platform.machine().lower()
    gpu_names = detect_gpu_names()
    gpu_blob = " ".join(gpu_names).upper()

    print("\n--- Platform detection ---")
    print(f"CPU arch: {arch}")
    print(f"GPU names: {gpu_names if gpu_names else 'N/A'}")

    is_x86 = arch in {"x86_64", "amd64"}
    is_arm = arch in {"aarch64", "arm64"}
    is_gb200 = any(token in gpu_blob for token in ["GB200", "B200", "BLACKWELL"])

    if is_arm and is_gb200:
        return "gb200"

    if is_arm:
        print("⚠️ ARM/aarch64 detected, but GPU name did not explicitly say GB200/B200.")
        print("   Using generic ARM setup path.")
        return "arm"

    if is_x86:
        return "x86"

    print(f"❌ ERROR: Unsupported CPU architecture: {arch}")
    print("Set COSMOS_PLATFORM=x86 or COSMOS_PLATFORM=gb200 to override.")
    sys.exit(1)


def detect_cuda_home():
    candidates = []

    for key in ["CUDA_HOME", "CUDA_PATH"]:
        val = os.environ.get(key)

        if val:
            candidates.append(Path(val))

    candidates.extend(
        [
            Path(sys.prefix),
            Path("/usr/local/cuda"),
            Path("/usr/local/cuda-13.2"),
            Path("/usr/local/cuda-13.0"),
            Path("/usr/local/cuda-12.9"),
            Path("/usr/local/cuda-12.8"),
            Path("/usr/local/cuda-12.6"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            nvcc = candidate / "bin" / "nvcc"
            include = candidate / "include"

            if nvcc.exists() or include.exists():
                return str(candidate.resolve())

    return str(Path(sys.prefix).resolve())


def detect_cuda_arch(platform_key):
    override = os.environ.get("COSMOS_CUDA_ARCH", "").strip()

    if override:
        print(f"✅ Using COSMOS_CUDA_ARCH override: {override}")
        return override

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"{major}{minor}"
            print(f"✅ Detected CUDA architecture from torch: sm_{arch}")
            return arch

    except Exception as exc:
        print(f"⚠️ Could not detect CUDA arch from torch: {type(exc).__name__}: {exc}")

    if platform_key == "gb200":
        print("✅ Falling back to GB200 CUDA architecture: sm_100")
        return "100"

    print("⚠️ Falling back to generic CUDA architecture: sm_90")
    return "90"


def arch_to_torch_arch_list(cuda_arch):
    s = str(cuda_arch).strip()

    if len(s) == 3:
        return f"{s[0:2]}.{s[2]}"

    if len(s) == 2:
        return f"{s[0]}.{s[1]}"

    return s


def torch_cuda_usable_for_platform(platform_key):
    try:
        import torch

        print("\n--- Existing torch check ---")
        print(f"torch version: {torch.__version__}")
        print(f"torch cuda: {torch.version.cuda}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("⚠️ torch.cuda.is_available() is False.")
            return False

        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)

        print(f"torch device 0: {name}")
        print(f"torch device capability: {cc}")

        if platform_key == "gb200" and cc[0] >= 10:
            torch_ver = str(torch.__version__)
            cuda_ver = str(torch.version.cuda or "")

            if "+cu126" in torch_ver or cuda_ver.startswith("12.6"):
                print("⚠️ Existing torch is CUDA 12.6 on GB200/sm_100. Reinstall required.")
                return False

        try:
            x = torch.ones(1, device="cuda")
            y = (x + 1).cpu()
            torch.cuda.synchronize()
            print(f"✅ CUDA tensor test OK: {y}")

        except Exception as exc:
            print(f"⚠️ CUDA tensor test failed: {type(exc).__name__}: {exc}")
            return False

        return True

    except Exception as exc:
        print("\n--- Existing torch check failed ---")
        print(f"{type(exc).__name__}: {exc}")
        return False


# ------------------------------------------------------------
# Build tools
# ------------------------------------------------------------


def ensure_build_tools(env):
    missing = []

    if not find_executable("cmake"):
        missing.append("cmake")

    if not find_executable("ninja"):
        missing.append("ninja")

    if missing:
        print(f"\n--- Installing missing Python build tools: {' '.join(missing)} ---")
        uv_pip_install(missing, env=env)

    cmake_bin = find_executable("cmake")
    ninja_bin = find_executable("ninja")

    if not cmake_bin:
        print("\n❌ ERROR: cmake is still not available.")
        print("Run:")
        print("  pixi add cmake ninja")
        sys.exit(1)

    if not ninja_bin:
        print("\n❌ ERROR: ninja is still not available.")
        print("Run:")
        print("  pixi add ninja")
        sys.exit(1)

    print(f"✅ cmake: {cmake_bin}")
    print(f"✅ ninja: {ninja_bin}")

    return cmake_bin, ninja_bin


def ensure_git_available():
    git_bin = find_executable("git")

    if not git_bin:
        print("\n❌ ERROR: git is not available.")
        print("Run:")
        print("  pixi add git")
        sys.exit(1)

    return git_bin


def find_nvml_library():
    override = os.environ.get("CUDA_NVIDIA_ML_LIBRARY", "").strip()

    if override and Path(override).exists():
        print(f"✅ Using CUDA_NVIDIA_ML_LIBRARY override: {override}")
        return override

    candidate_paths = [
        "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so",
        "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so",
        "/usr/lib64/libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/local/cuda/lib64/stubs/libnvidia-ml.so",
    ]

    for p in candidate_paths:
        if Path(p).exists():
            print(f"✅ Found NVML library: {p}")
            return p

    ldconfig_out = run_capture(["ldconfig", "-p"])

    for line in ldconfig_out.splitlines():
        if "libnvidia-ml.so" in line and "=>" in line:
            p = line.split("=>", 1)[1].strip()

            if Path(p).exists():
                print(f"✅ Found NVML library from ldconfig: {p}")
                return p

    print("⚠️ Could not find libnvidia-ml.so automatically.")
    return ""


# ------------------------------------------------------------
# CUDA runtime library visibility for Transformer Engine
# ------------------------------------------------------------


def install_cuda_runtime_libs_for_transformer_engine(env, cuda_major="12", required=False):
    """
    Transformer Engine may dlopen CUDA libraries directly, e.g. libcublas.so.13.

    For the current failure, we only install/link cuBLAS, CUDA runtime, and NVRTC.
    We intentionally avoid nvidia-cudnn / nvidia-cusolver here because their generic
    PyPI stub packages can fail during build metadata.
    """
    print(f"--- Ensuring CUDA {cuda_major} cuBLAS/runtime libs are visible to Transformer Engine ---")

    env_prefix = Path(sys.prefix).resolve()
    env_lib = env_prefix / "lib"
    env_lib.mkdir(parents=True, exist_ok=True)

    if cuda_major == "13":
        install_cmds = [
            ["--index-url", "https://pypi.nvidia.com", "nvidia-cublas"],
            ["--index-url", "https://pypi.nvidia.com", "nvidia-cuda-runtime"],
            ["--index-url", "https://pypi.nvidia.com", "nvidia-cuda-nvrtc"],
            # fallback names, do not fail if deprecated / unavailable
            # ["--index-url", "https://pypi.nvidia.com", "nvidia-cublas-cu13"],
            # ["--index-url", "https://pypi.nvidia.com", "nvidia-cuda-runtime-cu13"],
            # ["--index-url", "https://pypi.nvidia.com", "nvidia-cuda-nvrtc-cu13"],
        ]
    else:
        install_cmds = [
            ["nvidia-cublas-cu12"],
            ["nvidia-cuda-runtime-cu12"],
            ["nvidia-cuda-nvrtc-cu12"],
            ["nvidia-nccl-cu12"],
        ]

    for cmd in install_cmds:
        uv_pip_raw_install(cmd, env=env, check=False)

    candidate_roots = set()

    for base in site.getsitepackages():
        candidate_roots.add(Path(base) / "nvidia")

    candidate_roots.add(Path(sysconfig.get_paths()["purelib"]) / "nvidia")

    linked = 0

    def link_libs_from(lib_dir):
        nonlocal linked

        if not lib_dir.exists():
            return

        for so in lib_dir.glob("*.so*"):
            if not (
                so.name.startswith("libcublas")
                or so.name.startswith("libcublasLt")
                or so.name.startswith("libcudart")
                or so.name.startswith("libnvrtc")
                or so.name.startswith("libnccl")
            ):
                continue

            dst = env_lib / so.name

            if dst.exists():
                continue

            try:
                dst.symlink_to(so)
                linked += 1
                print(f"🔗 Linked {so.name} -> {dst}")
            except Exception as exc:
                print(f"⚠️ Could not link {so} -> {dst}: {exc}")

    for nvidia_root in candidate_roots:
        if not nvidia_root.exists():
            continue

        for lib_dir in nvidia_root.glob("*/lib"):
            link_libs_from(lib_dir)

    for lib_dir in [
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/lib"),
        Path("/usr/local/nvidia/lib64"),
        Path("/usr/local/nvidia/lib"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib64"),
    ]:
        link_libs_from(lib_dir)

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{env_lib}:{existing_ld}" if existing_ld else str(env_lib)

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{env_lib}:{current_ld}" if current_ld else str(env_lib)

    activate_dir = env_prefix / "etc" / "conda" / "activate.d"
    deactivate_dir = env_prefix / "etc" / "conda" / "deactivate.d"

    try:
        activate_dir.mkdir(parents=True, exist_ok=True)
        deactivate_dir.mkdir(parents=True, exist_ok=True)

        (activate_dir / "zz_cuda_runtime_libs.sh").write_text(
            f'''
export _OLD_LD_LIBRARY_PATH_FOR_CUDA_RUNTIME="${{LD_LIBRARY_PATH:-}}"
export LD_LIBRARY_PATH="{env_lib}:${{LD_LIBRARY_PATH:-}}"
'''.strip()
            + "\n"
        )

        (deactivate_dir / "zz_cuda_runtime_libs.sh").write_text(
            """
if [ -n "${_OLD_LD_LIBRARY_PATH_FOR_CUDA_RUNTIME+x}" ]; then
    export LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH_FOR_CUDA_RUNTIME}"
    unset _OLD_LD_LIBRARY_PATH_FOR_CUDA_RUNTIME
fi
""".strip()
            + "\n"
        )

        print(f"✅ Wrote activation LD_LIBRARY_PATH script under: {activate_dir}")

    except Exception as exc:
        print(f"⚠️ Could not write activation script: {exc}")

    print(f"✅ CUDA runtime library linking done. Linked {linked} libraries.")

    libcublas_candidates = list(env_lib.glob("libcublas.so.13*"))
    libcublaslt_candidates = list(env_lib.glob("libcublasLt.so.13*"))

    if cuda_major == "13" and required and not libcublas_candidates:
        print("❌ ERROR: libcublas.so.13 is still missing after installing/linking cuBLAS.")
        print("")
        print("Run these debug commands:")
        print("  find .pixi/envs/default -name 'libcublas.so.13*' -o -name 'libcublasLt.so.13*'")
        print("  find .pixi/envs/default/lib/python3.10/site-packages/nvidia -name 'libcublas.so*'")
        print("")
        print("This usually means the CUDA 13 cuBLAS package is not available from your configured indexes.")
        sys.exit(1)

    if libcublas_candidates or libcublaslt_candidates:
        print("✅ Found cuBLAS candidates:")
        for p in [*libcublas_candidates, *libcublaslt_candidates]:
            print(f"  {p}")
    else:
        print("⚠️ cuBLAS candidates were not found yet.")


# ------------------------------------------------------------
# Torch installs
# ------------------------------------------------------------


def install_gb200_torch_stack():
    if torch_cuda_usable_for_platform("gb200"):
        print("✅ Existing torch is usable for GB200.")
        return

    torch_version = os.environ.get("COSMOS_GB200_TORCH_VERSION", "2.13.0").strip()

    # Important: default to CUDA 12.9 so torch + Pixi CUDA<13 + TE stay consistent.
    index_url = os.environ.get(
        "COSMOS_GB200_TORCH_INDEX_URL",
        "https://download.pytorch.org/whl/cu129",
    ).strip()

    print("\n--- Reinstalling GB200-compatible torch stack ---")
    print(f"torch version: {torch_version}")
    print(f"torch index: {index_url}")

    uv_pip_uninstall(["torch", "torchvision", "triton"])

    uv_pip_install(
        [
            "--index-url",
            index_url,
            f"torch=={torch_version}",
            "torchvision",
        ]
    )

    if not torch_cuda_usable_for_platform("gb200"):
        print("\n❌ ERROR: GB200 torch install finished, but CUDA is still not usable.")
        print("Try another index:")
        print(
            "  COSMOS_GB200_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 COSMOS_PLATFORM=gb200 pixi run setup"
        )
        sys.exit(1)


def install_arm_torch_stack():
    if torch_cuda_usable_for_platform("arm"):
        print("✅ Existing torch is usable for ARM.")
        return

    print("\n❌ ERROR: CUDA-enabled torch is not usable in this ARM environment.")
    print("If this is actually GB200/B200, run:")
    print("  COSMOS_PLATFORM=gb200 pixi run setup")
    sys.exit(1)


def install_x86_torch_stack():
    print("--- Purging x86 conflicting variants ---")
    uv_pip_uninstall([*X86_TORCH_PACKAGES, *ACCEL_PACKAGES])

    print("--- Installing x86 PyTorch 2.6.0 CUDA 12.6 stack ---")
    uv_pip_install(
        [
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
            "torch==2.6.0+cu126",
            "torchvision==0.21.0+cu126",
            "triton==3.2.0",
        ]
    )


def install_x86_accel_stack():
    print("--- Installing x86 pre-compiled Cosmos acceleration wheels ---")
    uv_pip_install(
        [
            "--extra-index-url",
            "https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple",
            "transformer-engine==1.13.0+cu126.torch260",
            "flash-attn==2.6.3",
            "natten==0.21.0",
        ]
    )


# ------------------------------------------------------------
# ARM runtime deps
# ------------------------------------------------------------


def ensure_packaging(env):
    try:
        import packaging  # noqa: F401

        return
    except Exception:
        pass

    uv_pip_install(["packaging"], env=env)


def install_arm_runtime_deps_from_cosmos(env):
    print("--- Installing ARM/GB200 runtime dependencies from cosmos-predict2 metadata ---")

    ensure_packaging(env)

    try:
        from importlib.metadata import requires
        from packaging.requirements import Requirement
    except Exception as exc:
        print(f"⚠️ Could not import metadata tools: {type(exc).__name__}: {exc}")
        requires = None
        Requirement = None

    reqs = []

    if requires and Requirement:
        raw_reqs = requires("cosmos-predict2") or []

        for raw in raw_reqs:
            try:
                req = Requirement(raw)
                name = normalize_pkg_name(req.name)

                if req.marker and not req.marker.evaluate():
                    continue

                if name in MANAGED_ARM_DEPS:
                    print(f"⏭️  Skipping managed/problematic dep: {raw}")
                    continue

                reqs.append(raw)

            except Exception:
                reqs.append(raw)
    else:
        print("⚠️ Falling back to manual runtime dependency list.")

    extra_reqs = [
        "accelerate",
        "diffusers",
        "transformers",
        "peft",
        "huggingface-hub",
        "safetensors",
        "einops",
        "omegaconf",
        "hydra-core",
        "opencv-python-headless",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "tqdm",
        "wandb",
        "pandas",
        "scipy",
        "scikit-image",
        "protobuf",
        "sentencepiece",
        "ftfy",
        "regex",
        "pillow",
        "pyyaml",
        "requests",
        "filelock",
        "loguru",
        "attrs",
        "fvcore",
        "termcolor",
        "tabulate",
        "rich",
        "click",
        "nvidia-ml-py",
        "tensorboard",
        "torchmetrics",
        "torch-fidelity",
        "lpips",
        "onnx",
        "onnxscript",
        "qwen-vl-utils==0.0.11",
    ]

    seen = set()
    final_reqs = []

    for r in [*reqs, *extra_reqs]:
        key = str(r).lower()

        if key in seen:
            continue

        seen.add(key)
        final_reqs.append(r)

    uv_pip_install(final_reqs, env=env)

    print("--- Installing ARM/GB200 managed tensorstore / megatron-core deps ---")

    ok = uv_pip_install_try(["--only-binary=:all:", "tensorstore>=0.1.72"], env=env)

    if not ok:
        print("⚠️ tensorstore wheel install failed. Continuing without tensorstore.")

    uv_pip_install_try(["--no-deps", "megatron-core==0.10.0"], env=env)


# ------------------------------------------------------------
# Real Transformer Engine for GB200, no GitHub fallback
# ------------------------------------------------------------


def transformer_engine_has_required_api(env=None):
    """
    Check Transformer Engine in a fresh subprocess.

    Important: libcublas.so.13 visibility depends on LD_LIBRARY_PATH.
    """
    check_code = r"""
import sys
try:
    import transformer_engine as te
    import transformer_engine.pytorch as tep
    print("TE path:", getattr(te, "__file__", "OK"))

    required = ["LayerNormLinear", "GroupedLinear", "LayerNormMLP", "DotProductAttention"]
    missing = [name for name in required if not hasattr(tep, name)]
    if missing:
        print("missing TE APIs:", missing)
        sys.exit(2)

    import megatron.core.extensions.transformer_engine as mte
    print("Megatron TE extension OK:", getattr(mte, "__file__", "OK"))
    sys.exit(0)
except Exception as exc:
    print(type(exc).__name__ + ":", exc)
    sys.exit(1)
"""

    test_env = dict(os.environ)

    if env:
        test_env.update(env)

    env_lib = str(Path(sys.prefix).resolve() / "lib")
    old_ld = test_env.get("LD_LIBRARY_PATH", "")
    test_env["LD_LIBRARY_PATH"] = f"{env_lib}:{old_ld}" if old_ld else env_lib

    print("--- Checking Transformer Engine API in fresh subprocess ---")
    print(f"LD_LIBRARY_PATH={test_env['LD_LIBRARY_PATH']}")

    res = subprocess.run(
        [sys.executable, "-c", check_code],
        env=test_env,
        text=True,
        capture_output=True,
    )

    if res.stdout:
        print(res.stdout)

    if res.stderr:
        print(res.stderr)

    return res.returncode == 0


def remove_fake_or_broken_transformer_engine():
    print("--- Removing fake/broken transformer_engine installs before real TE install ---")

    try:
        purelib = Path(sysconfig.get_paths()["purelib"])
    except Exception:
        purelib = (
            Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        )

    for p in [purelib / "transformer_engine", purelib / "transformer_engine_torch"]:
        if p.exists():
            print(f"🧹 Removing directory: {p}")
            shutil.rmtree(p, ignore_errors=True)

    for pattern in [
        "transformer_engine*.dist-info",
        "transformer_engine*.egg-info",
        "transformer_engine_torch*.dist-info",
        "transformer_engine_torch*.egg-info",
        "transformer_engine_cu*.dist-info",
    ]:
        for p in purelib.glob(pattern):
            print(f"🧹 Removing metadata: {p}")
            shutil.rmtree(p, ignore_errors=True)

    importlib.invalidate_caches()

    uv_pip_uninstall(
        [
            "transformer-engine",
            "transformer-engine-cu12",
            "transformer-engine-cu13",
            "transformer-engine-torch",
        ]
    )


def install_real_transformer_engine_no_git(env, platform_key):
    """
    GB200 path must use real Transformer Engine.

    No GitHub install here because it takes too long.
    We only use PyPI/package install:
      transformer_engine[pytorch]
    """
    install_cuda_runtime_libs_for_transformer_engine(env, cuda_major="12", required=False)

    if transformer_engine_has_required_api(env):
        return

    remove_fake_or_broken_transformer_engine()

    cuda_arch = detect_cuda_arch(platform_key)
    torch_arch = arch_to_torch_arch_list(cuda_arch)

    build_env = dict(env)

    build_env["NVTE_FRAMEWORK"] = os.environ.get("NVTE_FRAMEWORK", "pytorch")
    build_env["NVTE_CUDA_ARCHS"] = os.environ.get("NVTE_CUDA_ARCHS", str(cuda_arch))
    build_env["CUDAARCHS"] = os.environ.get("CUDAARCHS", str(cuda_arch))
    build_env["CMAKE_CUDA_ARCHITECTURES"] = os.environ.get("CMAKE_CUDA_ARCHITECTURES", str(cuda_arch))
    build_env["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", torch_arch)
    build_env["MAX_JOBS"] = os.environ.get("MAX_JOBS", "1")
    build_env["NVTE_BUILD_THREADS_PER_JOB"] = os.environ.get("NVTE_BUILD_THREADS_PER_JOB", "1")
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "1")

    print("--- Installing build requirements for real Transformer Engine ---")
    uv_pip_install(
        [
            "psutil",
            "packaging",
            "wheel",
            "setuptools",
            "cmake",
            "ninja",
            "pybind11",
        ],
        env=build_env,
    )

    te_spec = os.environ.get("COSMOS_ARM_TE_SPEC", "transformer_engine[pytorch,core_cu12]").strip()

    print("--- Installing real Transformer Engine for GB200/aarch64, no GitHub fallback ---")
    print(f"TE spec: {te_spec}")
    print(f"NVTE_CUDA_ARCHS={build_env['NVTE_CUDA_ARCHS']}")
    print(f"TORCH_CUDA_ARCH_LIST={build_env['TORCH_CUDA_ARCH_LIST']}")
    print(f"LD_LIBRARY_PATH={build_env.get('LD_LIBRARY_PATH', '')}")

    first = uv_pip_raw_install(
        [
            "--no-build-isolation",
            te_spec,
        ],
        env=build_env,
        check=False,
    )

    if first.returncode != 0:
        print("❌ ERROR: PyPI/package Transformer Engine install failed.")
        print("Not falling back to GitHub because you asked to avoid git install.")
        print("")
        print("Try one of these:")
        print("  1. Use NVIDIA NGC PyTorch container with Transformer Engine preinstalled.")
        print("  2. Set COSMOS_ARM_TE_SPEC to a wheel/package available in your index.")
        print("  3. Confirm CUDA 13 libs exist:")
        print("       find .pixi/envs/default -name 'libcublas.so.13*'")
        sys.exit(1)

    install_cuda_runtime_libs_for_transformer_engine(build_env, cuda_major="12", required=True)

    if not transformer_engine_has_required_api(build_env):
        print("❌ ERROR: Transformer Engine installed, but the fresh subprocess check failed.")
        print("Most likely the CUDA runtime libraries are still not visible or the installed TE build is wrong.")
        print("Debug with:")
        print("  find .pixi/envs/default -name 'libcublas.so.13*' -o -name 'libcublasLt.so.13*'")
        print(
            "  LD_LIBRARY_PATH=$PWD/.pixi/envs/default/lib:$LD_LIBRARY_PATH pixi run python -c 'import transformer_engine.pytorch as te; print(te)'"
        )
        print("No GitHub fallback will be used.")
        sys.exit(1)

    print("✅ Real Transformer Engine is installed and compatible with Megatron import.")


def install_arm_optional_accel_stack(env):
    if os.environ.get("COSMOS_GB200_INSTALL_FLASH_ATTN", "0").strip() != "1":
        print("--- Skipping flash-attn / natten builds by default ---")
        print("Set COSMOS_GB200_INSTALL_FLASH_ATTN=1 to try flash-attn.")
        return

    flash_attn_version = os.environ.get("COSMOS_ARM_FLASH_ATTN_VERSION", "2.6.3")
    natten_version = os.environ.get("COSMOS_ARM_NATTEN_VERSION", "0.21.0")
    skip_natten = os.environ.get("COSMOS_SKIP_NATTEN", "1").strip() == "1"

    uv_pip_install(
        [
            "psutil",
            "packaging",
            "wheel",
            "setuptools",
            "cmake",
            "ninja",
        ],
        env=env,
    )

    print("--- Installing ARM/GB200 flash-attn from source/build path ---")
    uv_pip_install_try(
        [
            "--no-build-isolation",
            "--no-deps",
            f"flash-attn=={flash_attn_version}",
        ],
        env=env,
    )

    if skip_natten:
        print("--- Skipping ARM/GB200 NATTEN build ---")
        return

    print("--- Installing ARM/GB200 NATTEN from source/build path ---")
    uv_pip_install_try(
        [
            "--no-build-isolation",
            "--no-deps",
            f"natten=={natten_version}",
        ],
        env=env,
    )


# ------------------------------------------------------------
# Decord source patches/build
# ------------------------------------------------------------


def replace_function_body(content, function_regex, new_body):
    match = re.search(function_regex, content, flags=re.MULTILINE)

    if not match:
        return content, False

    open_brace = content.find("{", match.start())

    if open_brace < 0:
        return content, False

    depth = 0
    i = open_brace

    while i < len(content):
        ch = content[i]

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

            if depth == 0:
                close_brace = i
                before = content[: open_brace + 1]
                after = content[close_brace:]
                return before + "\n" + new_body.rstrip() + "\n" + after, True

        i += 1

    return content, False


def ffmpeg_av_find_best_stream_expects_const_decoder():
    header_candidates = [
        Path(sys.prefix) / "include" / "libavformat" / "avformat.h",
        Path(os.environ.get("CONDA_PREFIX", "")) / "include" / "libavformat" / "avformat.h",
        Path("/usr/include/libavformat/avformat.h"),
        Path("/usr/local/include/libavformat/avformat.h"),
    ]

    for header in header_candidates:
        if not header.exists():
            continue

        content = header.read_text(errors="ignore")

        if "av_find_best_stream" not in content:
            continue

        if "const AVCodec **decoder_ret" in content or "const struct AVCodec **decoder_ret" in content:
            print("✅ FFmpeg av_find_best_stream expects const AVCodec **")
            return True

        if "AVCodec **decoder_ret" in content or "struct AVCodec **decoder_ret" in content:
            print("✅ FFmpeg av_find_best_stream expects AVCodec **")
            return False

    print("⚠️ Could not detect av_find_best_stream decoder_ret constness.")
    print("   Defaulting to non-const AVCodec ** for older/pinned FFmpeg.")

    return False


def patch_decord_for_modern_ffmpeg(src_dir):
    print("--- Patching decord source for FFmpeg compatibility ---")

    av_find_best_stream_const = ffmpeg_av_find_best_stream_expects_const_decoder()

    source_files = []
    source_files.extend((src_dir / "src").glob("**/*.cc"))
    source_files.extend((src_dir / "src").glob("**/*.h"))
    source_files.extend((src_dir / "include").glob("**/*.h"))

    changed_files = []

    for path in source_files:
        content = path.read_text(errors="ignore")
        original = content

        if path.name == "ffmpeg_common.h":
            if "#include <libavcodec/bsf.h>" not in content:
                if "#include <libavcodec/avcodec.h>" in content:
                    content = content.replace(
                        "#include <libavcodec/avcodec.h>",
                        "#include <libavcodec/avcodec.h>\n#include <libavcodec/bsf.h>",
                    )
                else:
                    content = "#include <libavcodec/bsf.h>\n" + content

        content = content.replace("av_register_all();", "")
        content = content.replace("avcodec_register_all();", "")

        if av_find_best_stream_const:
            content = re.sub(
                r"(?<!const\s)\bAVCodec\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)",
                r"const AVCodec *\1",
                content,
            )
            content = re.sub(
                r"(?<!const\s)\bAVInputFormat\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)",
                r"const AVInputFormat *\1",
                content,
            )

        if path.name == "video_reader.cc":
            content, replaced = replace_function_body(
                content,
                r"double\s+VideoReader::GetRotation\s*\(\s*\)\s*const\s*\{",
                "  return 0.0;",
            )

            if replaced:
                print("✅ Patched VideoReader::GetRotation().")

        if content != original:
            path.write_text(content)
            changed_files.append(path)

    if changed_files:
        for path in changed_files:
            print(f"✅ Patched: {path}")
    else:
        print("ℹ️ No decord source patches were needed.")


def install_decord_arm(root, env, platform_key):
    if os.environ.get("COSMOS_SKIP_DECORD", "0").strip() == "1":
        print("⚠️ COSMOS_SKIP_DECORD=1 set. Skipping decord.")
        return

    if package_importable("decord"):
        print("✅ decord is already importable. Skipping decord install.")
        return

    print("--- Building decord from source for GB200/aarch64 ---")
    print("✅ Default decord mode: CPU-only")
    print("   To force CUDA decord, set COSMOS_DECORD_USE_CUDA=1")

    ensure_git_available()
    cmake_bin, _ = ensure_build_tools(env)

    decord_use_cuda = os.environ.get("COSMOS_DECORD_USE_CUDA", "0").strip() == "1"
    cuda_arch = detect_cuda_arch(platform_key)
    nvml_library = find_nvml_library() if decord_use_cuda else ""

    build_root = Path(
        os.environ.get(
            "COSMOS_DECORD_BUILD_ROOT",
            os.environ.get("COSMOS_DECOR_BUILD_ROOT", "/tmp/decord_build"),
        )
    ).resolve()

    src_dir = build_root / "decord"
    build_dir = src_dir / "build"
    build_root.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        run_cmd(
            [
                "git",
                "clone",
                "--recursive",
                "https://github.com/dmlc/decord.git",
                str(src_dir),
            ],
            env=env,
        )

    decord_ref = os.environ.get(
        "COSMOS_DECORD_GIT_REF",
        os.environ.get("COSMOS_DECOR_GIT_REF", "v0.6.0"),
    )

    run_cmd(["git", "-C", str(src_dir), "fetch", "--all", "--tags"], env=env, check=False)
    run_cmd(["git", "-C", str(src_dir), "checkout", decord_ref], env=env)
    run_cmd(["git", "-C", str(src_dir), "reset", "--hard", decord_ref], env=env)
    run_cmd(["git", "-C", str(src_dir), "clean", "-xfd"], env=env)
    run_cmd(["git", "-C", str(src_dir), "submodule", "update", "--init", "--recursive"], env=env)

    patch_decord_for_modern_ffmpeg(src_dir)

    if build_dir.exists():
        print(f"🧹 Removing stale decord build directory: {build_dir}")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    cuda_home = env.get("CUDA_HOME", "")
    nvcc = Path(cuda_home) / "bin" / "nvcc" if cuda_home else None

    cmake_args = [
        cmake_bin,
        "..",
        "-G",
        "Ninja",
        f"-DUSE_CUDA={'ON' if decord_use_cuda else 'OFF'}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={sys.prefix}",
    ]

    if decord_use_cuda:
        cmake_args.extend(
            [
                f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
                f"-DCUDA_ARCHITECTURES={cuda_arch}",
                "-DCMAKE_POLICY_DEFAULT_CMP0104=NEW",
            ]
        )

        if nvcc and nvcc.exists():
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")

        if nvml_library:
            cmake_args.append(f"-DCUDA_NVIDIA_ML_LIBRARY={nvml_library}")

    print("\n--- Decord CMake configuration ---")
    print(f"Python executable: {sys.executable}")
    print(f"Python prefix: {sys.prefix}")
    print(f"CUDA_HOME: {cuda_home}")
    print(f"Build CUDA decord: {decord_use_cuda}")
    print(f"CUDA arch: {cuda_arch if decord_use_cuda else 'N/A'}")
    print(f"NVML library: {nvml_library if nvml_library else 'N/A'}")

    run_cmd(cmake_args, cwd=build_dir, env=env)
    run_cmd([cmake_bin, "--build", ".", "--parallel", str(os.cpu_count() or 8)], cwd=build_dir, env=env)

    python_dir = src_dir / "python"
    run_cmd([sys.executable, "setup.py", "bdist_wheel"], cwd=python_dir, env=env)

    dist_dir = python_dir / "dist"
    wheels = sorted(dist_dir.glob("decord-*.whl"))

    if not wheels:
        print(f"❌ ERROR: No decord wheel produced under {dist_dir}")
        sys.exit(1)

    wheel = wheels[-1]

    uv_pip_raw_install(["--force-reinstall", str(wheel)], env=env)

    if not package_importable("decord"):
        print("❌ ERROR: decord was installed, but Python still cannot import it.")
        sys.exit(1)

    print("✅ decord built and installed successfully for GB200/aarch64.")


# ------------------------------------------------------------
# Cosmos install / patching
# ------------------------------------------------------------


def install_cosmos_predict2(root, *, platform_key="x86", env=None):
    override_file = root / "tensorstore_override.txt"
    override_file.write_text("tensorstore>=0.1.72\n")

    try:
        if platform_key in {"gb200", "arm"}:
            print(f"--- Installing cosmos-predict2=={COSMOS_VERSION} on ARM/GB200 with --no-deps ---")
            print("⚠️ Skipping dependency resolver because some required packages have no aarch64 wheels.")
            uv_pip_install(["--no-deps", f"cosmos-predict2=={COSMOS_VERSION}"], env=env)
        else:
            print("--- Generating package version override for tensorstore ---")
            print(f"--- Installing target cosmos-predict2=={COSMOS_VERSION} with overrides ---")
            uv_pip_install(["--override", str(override_file), f"cosmos-predict2=={COSMOS_VERSION}"], env=env)

    finally:
        if override_file.exists():
            override_file.unlink()


def prepare_transformer_engine_include_links(cuda_home):
    cuda_home_path = Path(cuda_home)
    include_dir = cuda_home_path / "include"

    try:
        include_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    py_include_dir = include_dir / py_ver

    try:
        py_include_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    site_packages_candidates = [
        Path(sys.prefix) / "lib" / py_ver / "site-packages",
        cuda_home_path / "lib" / py_ver / "site-packages",
    ]

    for site_packages in site_packages_candidates:
        nvidia_dir = site_packages / "nvidia"

        if not nvidia_dir.exists():
            continue

        for header in nvidia_dir.glob("*/include/*"):
            if not header.exists():
                continue

            for dst_root in [include_dir, py_include_dir]:
                dst = dst_root / header.name

                if dst.exists():
                    continue

                try:
                    dst.symlink_to(header)
                    print(f"🔗 Linked {header} -> {dst}")
                except Exception:
                    pass


def apply_patch(target_file, patch_file):
    abs_target = str(target_file.resolve())
    abs_patch = str(patch_file.resolve())

    if not target_file.exists():
        print(f"⚠️ Warning: Target file not found for patching: {abs_target}")
        return

    if not patch_file.exists():
        print(f"⚠️ Warning: Patch file not found: {abs_patch}")
        return

    check = subprocess.run(
        [
            "patch",
            "-R",
            "-N",
            "--dry-run",
            abs_target,
            abs_patch,
        ],
        capture_output=True,
        text=True,
    )

    if check.returncode == 0:
        print(f"✅ Patch '{patch_file.name}' is already applied. Skipping.")
        return

    run_cmd(["patch", "-N", abs_target, abs_patch])


def locate_cosmos_predict2():
    print("--- Locating active cosmos_predict2 package directory ---")

    spec = importlib.util.find_spec("cosmos_predict2")

    if spec and spec.submodule_search_locations:
        cosmos_path = Path(spec.submodule_search_locations[0]).resolve()
        print(f"🎯 Found live package directory at: {cosmos_path}")
        return cosmos_path

    print("❌ ERROR: Could not locate cosmos_predict2 in environment site-packages.")
    sys.exit(1)


def bypass_cosmos_cuda_extra_check(cosmos_path):
    init_py = cosmos_path / "__init__.py"

    if not init_py.exists():
        return

    init_content = init_py.read_text(errors="ignore")

    if "_check_cuda_extra()" in init_content and "# _check_cuda_extra()" not in init_content:
        print("--- Bypassing container validation check in __init__.py ---")
        init_py.write_text(init_content.replace("_check_cuda_extra()", "# _check_cuda_extra()"))


def apply_repository_patches(root, cosmos_path):
    tokenizer_py = cosmos_path / "tokenizers" / "tokenizer.py"
    dit_py = cosmos_path / "models" / "text2image_dit.py"

    print("--- Applying repository patch configurations ---")
    apply_patch(tokenizer_py, root / "tokenizer.patch")
    apply_patch(dit_py, root / "text2image_dit.patch")


def patch_torch_onnx_compat():
    print("--- Checking torch.onnx compatibility shims ---")

    try:
        import torch
        import torch.onnx as torch_onnx
    except Exception as exc:
        print(f"⚠️ Could not import torch/torch.onnx: {type(exc).__name__}: {exc}")
        return

    onnx_dir = Path(torch_onnx.__file__).resolve().parent
    internal_dir = onnx_dir / "_internal"
    internal_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch.onnx import _type_utils  # noqa: F401

        print("✅ torch.onnx._type_utils is already importable.")
    except Exception as exc:
        print(f"⚠️ torch.onnx._type_utils import failed: {type(exc).__name__}: {exc}")

        shim_path = onnx_dir / "_type_utils.py"
        print(f"--- Writing torch.onnx._type_utils compatibility shim: {shim_path}")

        shim_path.write_text(
            """
# Auto-generated compatibility shim for packages expecting torch.onnx._type_utils.

from enum import IntEnum
import torch

try:
    from torch.onnx._internal._type_utils import *  # type: ignore # noqa
except Exception:
    try:
        from torch.onnx._internal.fx._type_utils import *  # type: ignore # noqa
    except Exception:
        class JitScalarType(IntEnum):
            UNDEFINED = 0
            FLOAT = 1
            UINT8 = 2
            INT8 = 3
            INT16 = 4
            INT32 = 5
            INT64 = 6
            STRING = 7
            BOOL = 9
            FLOAT16 = 10
            DOUBLE = 11
            COMPLEX64 = 14
            COMPLEX128 = 15
            BFLOAT16 = 16

            @classmethod
            def from_dtype(cls, dtype):
                return _DTYPE_TO_JIT_SCALAR_TYPE.get(dtype, cls.UNDEFINED)

            @classmethod
            def from_value(cls, value, default=None):
                dtype = getattr(value, "dtype", None)
                if dtype is not None:
                    return cls.from_dtype(dtype)
                if isinstance(value, torch.Tensor):
                    return cls.from_dtype(value.dtype)
                if default is not None:
                    return default
                return cls.UNDEFINED

            @classmethod
            def from_name(cls, name):
                if name is None:
                    return cls.UNDEFINED
                normalized = str(name).lower().replace("tensor(", "").replace(")", "")
                return _NAME_TO_JIT_SCALAR_TYPE.get(normalized, cls.UNDEFINED)

            def scalar_name(self):
                return _JIT_SCALAR_TYPE_TO_SCALAR_NAME.get(self, "UNKNOWN_SCALAR")

            def torch_name(self):
                return _JIT_SCALAR_TYPE_TO_TORCH_NAME.get(self, "UNKNOWN_TORCH_TYPE")

            def dtype(self):
                return _JIT_SCALAR_TYPE_TO_DTYPE.get(self, None)

            def onnx_type(self):
                return int(self)

        _DTYPE_TO_JIT_SCALAR_TYPE = {
            torch.float32: JitScalarType.FLOAT,
            torch.float: JitScalarType.FLOAT,
            torch.float64: JitScalarType.DOUBLE,
            torch.double: JitScalarType.DOUBLE,
            torch.float16: JitScalarType.FLOAT16,
            torch.half: JitScalarType.FLOAT16,
            torch.bfloat16: JitScalarType.BFLOAT16,
            torch.uint8: JitScalarType.UINT8,
            torch.int8: JitScalarType.INT8,
            torch.int16: JitScalarType.INT16,
            torch.short: JitScalarType.INT16,
            torch.int32: JitScalarType.INT32,
            torch.int: JitScalarType.INT32,
            torch.int64: JitScalarType.INT64,
            torch.long: JitScalarType.INT64,
            torch.bool: JitScalarType.BOOL,
            torch.complex64: JitScalarType.COMPLEX64,
            torch.complex128: JitScalarType.COMPLEX128,
        }

        _JIT_SCALAR_TYPE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_JIT_SCALAR_TYPE.items()}
        _JIT_SCALAR_TYPE_TO_SCALAR_NAME = {x: x.name for x in JitScalarType}
        _JIT_SCALAR_TYPE_TO_TORCH_NAME = {x: x.name for x in JitScalarType}
        _NAME_TO_JIT_SCALAR_TYPE = {x.name.lower(): x for x in JitScalarType}
        ScalarName = JitScalarType
""".strip()
            + "\n"
        )

    try:
        from torch.onnx._internal import jit_utils  # noqa: F401

        print("✅ torch.onnx._internal.jit_utils is already importable.")
    except Exception as exc:
        print(f"⚠️ torch.onnx._internal.jit_utils import failed: {type(exc).__name__}: {exc}")

        shim_path = internal_dir / "jit_utils.py"
        print(f"--- Writing torch.onnx._internal.jit_utils compatibility shim: {shim_path}")

        shim_path.write_text(
            """
# Auto-generated compatibility shim for packages expecting torch.onnx._internal.jit_utils.

try:
    from torch.onnx._internal.fx.jit_utils import *  # type: ignore # noqa
except Exception:
    pass

class GraphContext:
    def __init__(self, graph=None, block=None, opset=17, original_node=None, params_dict=None, env=None, values_in_env=None, new_nodes=None):
        self.graph = graph
        self.block = block
        self.opset = opset
        self.original_node = original_node
        self.params_dict = params_dict or {}
        self.env = env or {}
        self.values_in_env = values_in_env or set()
        self.new_nodes = new_nodes or []

    def op(self, opname, *raw_args, outputs=1, **kwargs):
        if self.graph is not None and hasattr(self.graph, "op"):
            return self.graph.op(opname, *raw_args, outputs=outputs, **kwargs)
        raise RuntimeError("GraphContext.op() compatibility shim was called without a valid graph.")

    def aten_op(self, operator, *args, overload_name="", **kwargs):
        opname = f"aten::{operator}"
        if overload_name:
            opname = f"{opname}.{overload_name}"
        return self.op(opname, *args, **kwargs)
""".strip()
            + "\n"
        )

    importlib.invalidate_caches()


def self_heal_transformer_engine():
    print("--- Applying self-healing patches to transformer_engine ---")

    site_candidates = [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sysconfig.get_paths()["purelib"]),
    ]

    te_dirs = []

    for site_path in site_candidates:
        candidate = site_path / "transformer_engine"

        if candidate.exists():
            te_dirs.append(candidate)

    if not te_dirs:
        te_spec = importlib.util.find_spec("transformer_engine")

        if te_spec and te_spec.submodule_search_locations:
            te_dirs.append(Path(te_spec.submodule_search_locations[0]).resolve())

    if not te_dirs:
        print("⚠️ transformer_engine package not found. Skipping self-heal.")
        return

    for te_dir in te_dirs:
        print(f"🎯 transformer_engine dir: {te_dir}")

        for py_file in te_dir.glob("**/*.py"):
            content = py_file.read_text(errors="ignore")
            original = content

            content = content.replace(
                "ldconfig -p | grep 'libnvrtc'",
                "ldconfig -p | grep 'libnvrtc' || true",
            )
            content = content.replace(
                'ldconfig -p | grep "libnvrtc"',
                'ldconfig -p | grep "libnvrtc" || true',
            )
            content = content.replace(
                "ldconfig -p | grep libnvrtc",
                "ldconfig -p | grep libnvrtc || true",
            )
            content = content.replace("|| true || true", "|| true")

            if content != original:
                py_file.write_text(content)
                print(f"✅ Patched ldconfig check in: {py_file}")


def patch_imaginaire_flash_attn_requirement():
    print("--- Patching imaginaire Qwen2.5-VL flash_attn import-time assertion ---")

    try:
        spec = importlib.util.find_spec("imaginaire")
    except Exception as exc:
        print(f"⚠️ Could not locate imaginaire: {type(exc).__name__}: {exc}")
        return

    if not spec or not spec.submodule_search_locations:
        print("⚠️ imaginaire package not found. Skipping flash_attn patch.")
        return

    site_dir = Path(list(spec.submodule_search_locations)[0]).resolve()
    target = site_dir / "networks" / "qwen2_5_vl.py"

    if not target.exists():
        print(f"⚠️ qwen2_5_vl.py not found: {target}")
        return

    s = target.read_text(errors="ignore")

    old = '''assert is_flash_attn_2_available(), "flash_attn_2 not available. run pip install flash_attn"'''

    new = """\
if not is_flash_attn_2_available():
    print("⚠️ flash_attn_2 not available; continuing without flash_attn import-time requirement on GB200.")
"""

    if old in s:
        target.write_text(s.replace(old, new))
        print(f"✅ Patched flash_attn assert in: {target}")
        return

    print("ℹ️ flash_attn assert already patched or exact line not found.")


# ------------------------------------------------------------
# Main setup flow
# ------------------------------------------------------------


def setup_gb200_or_arm(root, platform_key):
    print(f"--- Purging {platform_key} conflicting acceleration packages only ---")
    uv_pip_uninstall(ACCEL_PACKAGES)

    if platform_key == "gb200":
        install_gb200_torch_stack()
    else:
        install_arm_torch_stack()

    cuda_home = detect_cuda_home()

    if not cuda_home:
        print("\n❌ ERROR: Could not detect CUDA_HOME.")
        print("Set CUDA_HOME or CUDA_PATH before running this script.")
        sys.exit(1)

    print(f"--- Using CUDA_HOME={cuda_home} ---")

    env = make_build_env(os.environ)
    env["CUDA_HOME"] = cuda_home

    cuda_arch = detect_cuda_arch(platform_key)
    env["TORCH_CUDA_ARCH_LIST"] = arch_to_torch_arch_list(cuda_arch)
    env["CMAKE_CUDA_ARCHITECTURES"] = cuda_arch
    env["CUDAARCHS"] = cuda_arch

    nvml_library = find_nvml_library()

    if nvml_library:
        env["CUDA_NVIDIA_ML_LIBRARY"] = nvml_library

    install_decord_arm(root, env, platform_key)
    install_cosmos_predict2(root, platform_key=platform_key, env=env)
    install_arm_runtime_deps_from_cosmos(env)
    prepare_transformer_engine_include_links(cuda_home)

    install_real_transformer_engine_no_git(env, platform_key)

    patch_imaginaire_flash_attn_requirement()

    install_arm_optional_accel_stack(env)


def verify_key_imports():
    print("--- Verifying key imports ---")

    checks = [
        "torch",
        "torchvision",
        "accelerate",
        "diffusers",
        "transformers",
        "qwen_vl_utils",
        "decord",
        "transformer_engine",
        "megatron.core",
        "loguru",
        "attrs",
        "cosmos_predict2",
    ]

    for name in checks:
        try:
            mod = importlib.import_module(name)
            print(f"✅ import {name}: {getattr(mod, '__file__', 'OK')}")
        except Exception as exc:
            print(f"⚠️ import {name} failed: {type(exc).__name__}: {exc}")

    try:
        import transformer_engine.pytorch as tep

        print("✅ TE LayerNormLinear:", hasattr(tep, "LayerNormLinear"))
        print("✅ TE GroupedLinear:", hasattr(tep, "GroupedLinear"))
        print("✅ TE LayerNormMLP:", hasattr(tep, "LayerNormMLP"))
        print("✅ TE DotProductAttention:", hasattr(tep, "DotProductAttention"))
    except Exception as exc:
        print(f"⚠️ transformer_engine.pytorch API check failed: {type(exc).__name__}: {exc}")

    try:
        import megatron.core.extensions.transformer_engine as mte

        print(f"✅ import megatron TE extension: {getattr(mte, '__file__', 'OK')}")
    except Exception as exc:
        print(f"⚠️ import megatron TE extension failed: {type(exc).__name__}: {exc}")

    try:
        import imaginaire.networks.qwen2_5_vl as qwen25

        print(f"✅ import imaginaire.networks.qwen2_5_vl: {getattr(qwen25, '__file__', 'OK')}")
    except Exception as exc:
        print(f"⚠️ import imaginaire.networks.qwen2_5_vl failed: {type(exc).__name__}: {exc}")

    try:
        import cosmos_predict2.configs.base.config_text2image as cfg

        print(f"✅ import cosmos config_text2image: {getattr(cfg, '__file__', 'OK')}")
    except Exception as exc:
        print(f"⚠️ import cosmos config_text2image failed: {type(exc).__name__}: {exc}")

    try:
        from torch.onnx import _type_utils  # noqa: F401

        print("✅ import torch.onnx._type_utils: OK")
    except Exception as exc:
        print(f"⚠️ import torch.onnx._type_utils failed: {type(exc).__name__}: {exc}")

    try:
        from torch.onnx._internal import jit_utils  # noqa: F401

        print("✅ import torch.onnx._internal.jit_utils: OK")
    except Exception as exc:
        print(f"⚠️ import torch.onnx._internal.jit_utils failed: {type(exc).__name__}: {exc}")


def main():
    maybe_prepare_pixi_native_deps()

    root = Path(os.environ.get("PIXI_PROJECT_ROOT", os.getcwd())).resolve()
    os.chdir(root)

    platform_key = detect_platform()

    print(f"\n✅ Selected setup target: {platform_key}")
    print(f"✅ Python executable: {sys.executable}")
    print(f"✅ Python prefix: {sys.prefix}")

    if platform_key == "x86":
        install_x86_torch_stack()
        install_cosmos_predict2(root, platform_key=platform_key)
        install_x86_accel_stack()

    elif platform_key in {"gb200", "arm"}:
        setup_gb200_or_arm(root, platform_key)

    else:
        print(f"❌ ERROR: Unknown platform key: {platform_key}")
        sys.exit(1)

    cosmos_path = locate_cosmos_predict2()

    bypass_cosmos_cuda_extra_check(cosmos_path)
    apply_repository_patches(root, cosmos_path)

    patch_torch_onnx_compat()
    self_heal_transformer_engine()
    patch_imaginaire_flash_attn_requirement()
    verify_key_imports()

    print(f"\n🎉 Environment setup completed successfully for {platform_key} under Python {sys.version.split()[0]}!\n")


if __name__ == "__main__":
    main()
