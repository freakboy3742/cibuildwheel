from __future__ import annotations

import importlib
import os
import pprint
import sys
from collections.abc import Sequence
from pathlib import Path

from .typing import PathOrStr
from .util import virtualenv


def cross_virtualenv(
    py_version: str,
    os_name: str,
    os_version: str,
    multiarch: str,
    arch: str,
    sdk: str,
    host_python: Path,
    build_python: Path,
    venv_path: Path,
    dependency_constraint_flags: Sequence[PathOrStr],
) -> dict[str, str]:
    """Create a cross-compilation virtual environment.

    In a cross-compilation environment, the *host* is the platform you're
    targeting the *build* is the platform where you're running the compilation.
    For example, when building iOS wheels, iOS is the host machine and macOS is
    the build machine.

    A cross-compilation virtualenv is an environment that is based on the
    *build* python (so that binaries can execute); but it's patched so that any
    request about platform details (such as `sys.platform` or
    `sysconfig.get_platform()`) return details of the host platform.

    :param py_version: The Python version (major.minor) in use
    :param os_name: The human readable name of the host operating system (i.e.,
        the value returned by platform.system())
    :param os_version: The version of the host operating system.
    :param multiarch: The multiarch tag for the host platform (i.e., the value
        of `sys.implementation._multiarch`)
    :param arch: The architecture for the host platform
    :param sdk: The SDK for the host platform
    :param host_python: The path to the python binary for the host platform
    :param build_python: The path to the python binary for the build platform
    :param venv_path: The path where the cross virtual environment should be
        created.
    :param dependency_constraint_flags: Any flags that should be used when
        constraining dependencies in the environment.
    """
    env = virtualenv(
        py_version,
        build_python,
        venv_path,
        dependency_constraint_flags,
        use_uv=False,
    )

    # Determine the kernel name and version, based on the OS name
    if os_name == "iOS":
        kernel_name = "Darwin"
        kernel_version = "23.5.0"
    else:
        raise ValueError(f"Can't build a cross-platform virtualenv for {os_name}")

    # Create the folder where the cross-platform configuratoin will be stored
    cross_path = venv_path / "cross"
    cross_path.mkdir()

    # Copy and patch the sysconfig module.
    # Start by loading the sysconfigdata module from the host Python
    host_prefix = host_python.parent.parent
    sysconfigdata_module = (
        host_prefix
        / "lib"
        / f"python{py_version}"
        / f"_sysconfigdata__{os_name.lower()}_{multiarch}.py"
    )
    spec = importlib.util.spec_from_file_location(sysconfigdata_module.stem, sysconfigdata_module)
    sysconfigdata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sysconfigdata)

    # The host's sysconfigdata will include references to build-time variables.
    # Update these to refer to the current known install location.
    orig_prefix = sysconfigdata.build_time_vars["prefix"]
    build_time_vars = {}
    for key, value in sysconfigdata.build_time_vars.items():
        if isinstance(value, str):
            # Replace any reference to the build installation prefix
            value = value.replace(orig_prefix, str(host_prefix))
            # Replace any reference to the build-time Framework location
            value = value.replace("-F .", f"-F {host_prefix}")
        build_time_vars[key] = value

    # Write the updated sysconfigdata module into the cross-platform site.
    with (cross_path / sysconfigdata_module.name).open("w") as f:
        f.write(f"# Generated from {sysconfigdata_module}\n")
        f.write("build_time_vars = ")
        pprint.pprint(build_time_vars, stream=f, compact=True)

    # Move the virtualenv's python binary to build-python
    (venv_path / "bin" / "python").rename(venv_path / "bin" / "build-python")
    (venv_path / "bin" / "build-python3").symlink_to(venv_path / "bin" / "build-python")
    (venv_path / "bin" / f"build-python{py_version}").symlink_to(venv_path / "bin" / "build-python")

    # Roll out the template for the cross-environment site folder
    cross_template = Path(__file__).parent / "resources" / "cross-site"
    for full_template_path in cross_template.glob("**/*.tmpl"):
        template_path = full_template_path.relative_to(cross_template)

        out_path = venv_path / template_path.parent / template_path.stem
        with full_template_path.open() as template_f, out_path.open("w") as out_f:
            while line := template_f.readline():
                out_f.write(
                    line.format(
                        host_python_home=str(host_python.parent.parent),
                        build_python_home=str(build_python.parent.parent),
                        venv_path=str(venv_path),
                        py_version=py_version,
                        py_version_tag="".join(py_version.split(".")[:2]),
                        os_name=os_name,
                        os_version=os_version,
                        platform=os_name.lower(),
                        kernel_name=kernel_name,
                        kernel_version=kernel_version,
                        multiarch=multiarch,
                        arch=arch,
                        sdk=sdk,
                    )
                )

    # Ensure the templated cross-platform python "binary" is executable
    (venv_path / "bin" / "python").chmod(0o755)

    # When running on macOS, it's easy for the build environment to leak into
    # the host environment, especially when building for ARM64 (because the
    # architecture is the same as the host architecture). The primary culprit
    # for this is homebrew libraries leaking in as dependencies for iOS
    # libraries.
    #
    # To prevent problems, isolate the build environment to only include the
    # system libraries, the host and build environments, and Cargo (to allow for
    # Rust compilation).
    if sys.platform == "darwin":
        env["PATH"] = os.pathsep.join(
            [
                str(host_python.parent),
                str(venv_path / "bin"),
                str(Path.home() / ".cargo" / "bin"),
                "/usr/bin",
                "/bin",
                "/usr/sbin",
                "/sbin",
                "/Library/Apple/usr/bin",
            ]
        )

    return env
