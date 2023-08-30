from setuptools import setup, Extension
from torch.utils import cpp_extension
import platform
import os
import subprocess
import torch

def get_cuda_version(cuda_home=os.environ.get('CUDA_PATH', '') if platform.system() == "Windows" else os.environ.get('CUDA_HOME', '')):
    if cuda_home == '' or not os.path.exists(os.path.join(cuda_home,"bin","nvcc.exe" if platform.system() == "Windows" else "nvcc")):
        return ''
    version_str = subprocess.check_output([os.path.join(cuda_home,"bin","nvcc"),"--version"])
    version_str=str(version_str).replace('\n', '').replace('\r', '')
    idx=version_str.find("release")
    return version_str[idx+len("release "):idx+len("release ")+4]
    
CUDA_VERSION = "".join(get_cuda_version().split(".")) if not os.environ.get('ROCM_VERSION', False) else False
ROCM_VERSION = os.environ.get('ROCM_VERSION', False) if torch.version.hip else False

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": ["-O3"],
}
if torch.version.hip:
    extra_compile_args["nvcc"].append("-U__HIP_NO_HALF_CONVERSIONS__")

version = "0.0.14" + (f"+cu{CUDA_VERSION}" if CUDA_VERSION else f"+rocm{ROCM_VERSION}" if ROCM_VERSION else "")
setup(
    name="exllama",
    version=version,
    install_requires=[
        "torch",
    ],
    packages=["exllama"],
    py_modules=["exllama"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "exllama_ext",
            [
                "exllama_ext/exllama_ext.cpp",
                "exllama_ext/cuda_buffers.cu",
                "exllama_ext/cuda_func/q4_matrix.cu",
                "exllama_ext/cuda_func/q4_matmul.cu",
                "exllama_ext/cuda_func/column_remap.cu",
                "exllama_ext/cuda_func/rms_norm.cu",
                "exllama_ext/cuda_func/rope.cu",
                "exllama_ext/cuda_func/half_matmul.cu",
                "exllama_ext/cuda_func/q4_attn.cu",
                "exllama_ext/cuda_func/q4_mlp.cu",
                "exllama_ext/cpu_func/rep_penalty.cpp",
            ],
            extra_compile_args=extra_compile_args,
            libraries=["cublas"] if platform.system() == "Windows" else [],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
