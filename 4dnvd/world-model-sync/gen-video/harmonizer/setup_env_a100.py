import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def run_cmd(cmd):
    print(f"\n🚀 Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=False)
    if res.returncode != 0:
        print(f"\n❌ ERROR: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def apply_patch(target_file, patch_file):
    abs_target = str(target_file.resolve())
    abs_patch = str(patch_file.resolve())

    if not target_file.exists():
        print(f"⚠️ Warning: Target file not found for patching: {abs_target}")
        return

    # Dry-run a reversal to check if the patch is already present
    check = subprocess.run(["patch", "-R", "-N", "--dry-run", abs_target, abs_patch], capture_output=True)
    if check.returncode == 0:
        print(f"✅ Patch '{patch_file.name}' is already applied. Skipping.")
        return

    run_cmd(["patch", "-N", abs_target, abs_patch])


def main():
    root = Path(os.environ.get("PIXI_PROJECT_ROOT", os.getcwd())).resolve()
    os.chdir(root)

    # 1. Purge modern conflicting variants
    print("--- Purging modern variants ---")
    subprocess.run(
        [
            "uv",
            "pip",
            "uninstall",
            "--python",
            sys.executable,
            "torch",
            "torchvision",
            "triton",
            "flash-attn",
            "natten",
            "transformer-engine",
            "transformer-engine-cu12",
            "transformer-engine-torch",
        ],
        capture_output=True,
    )

    # 2. Force install stable production PyTorch 2.6.0 stack matching the Cosmos wheel layout
    print("--- Installing stable production PyTorch 2.6.0 stack ---")
    run_cmd(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
            "torch==2.6.0+cu126",
            "torchvision==0.21.0+cu126",
            "triton==3.2.0",
        ]
    )

    # 3. Generate package version override for tensorstore to avoid source compilation
    print("--- Generating package version override for tensorstore ---")
    override_file = root / "tensorstore_override.txt"
    override_file.write_text("tensorstore>=0.1.72\n")

    # 4. Install target cosmos-predict2 version required by Harmonizer (v1.0.9)
    print("--- Installing target cosmos-predict2==1.0.9 with overrides ---")
    run_cmd(
        ["uv", "pip", "install", "--python", sys.executable, "--override", str(override_file), "cosmos-predict2==1.0.9"]
    )

    # 5. Fetch the exact pre-compiled Python 3.10 standalone binary wheel from the custom Cosmos warehouse index
    print("--- Fetching pre-compiled Cosmos Python 3.10 hardware-accelerated engines ---")
    run_cmd(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--extra-index-url",
            "https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple",
            "transformer-engine==1.13.0+cu126.torch260",
            "flash-attn==2.6.3",
            "natten==0.21.0",
        ]
    )

    # 6. Dynamically locate the standard environment package directory
    print("--- Locating active cosmos_predict2 package directory ---")
    spec = importlib.util.find_spec("cosmos_predict2")
    if spec and spec.submodule_search_locations:
        cosmos_path = Path(spec.submodule_search_locations[0]).resolve()
        print(f"🎯 Found live package directory at: {cosmos_path}")
    else:
        print("❌ ERROR: Could not locate cosmos_predict2 in environment site-packages.")
        sys.exit(1)

    # 7. Bypass the strict docker-container check inside __init__.py
    init_py = cosmos_path / "__init__.py"
    if init_py.exists():
        init_content = init_py.read_text()
        if "_check_cuda_extra()" in init_content and "# _check_cuda_extra()" not in init_content:
            print("--- Bypassing container validation check in __init__.py ---")
            init_py.write_text(init_content.replace("_check_cuda_extra()", "# _check_cuda_extra()"))

    # 8. Apply standard repository patches to the 1.0.9 source files
    tokenizer_py = cosmos_path / "tokenizers" / "tokenizer.py"
    dit_py = cosmos_path / "models" / "text2image_dit.py"

    print("--- Applying repository patch configurations ---")
    apply_patch(tokenizer_py, (root / "tokenizer.patch").resolve())
    apply_patch(dit_py, (root / "text2image_dit.patch").resolve())

    # 9. Self-heal the ldconfig check loop inside transformer_engine 1.13.0
    print("--- Applying self-healing patches to transformer_engine ---")
    te_spec = importlib.util.find_spec("transformer_engine")
    if te_spec and te_spec.submodule_search_locations:
        te_dir = Path(te_spec.submodule_search_locations[0]).resolve()
        for py_file in te_dir.glob("**/*.py"):
            content = py_file.read_text(errors="ignore")
            if "ldconfig -p | grep" in content and "|| true" not in content:
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if "ldconfig -p | grep" in line and "|| true" not in line:
                        lines[i] = line.replace('", shell=True)', ' || true", shell=True)').replace(
                            "', shell=True)", " || true', shell=True)"
                        )
                py_file.write_text("\n".join(lines))
                print(f"🎯 Bypassed system ldconfig validation blocks in: {py_file.name}")

    if override_file.exists():
        override_file.unlink()

    print("\n🎉 Environment setup completed successfully under Python 3.10!\n")


if __name__ == "__main__":
    main()
