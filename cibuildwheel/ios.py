from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence, Set
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock

from . import errors
from ._compat.typing import assert_never
from .architecture import Architecture
from .cross import cross_virtualenv
from .environment import ParsedEnvironment
from .logger import log
from .options import Options
from .typing import PathOrStr
from .macos import install_cpython as install_build_cpython
from .util import (
    CIBW_CACHE_PATH,
    BuildFrontendConfig,
    BuildFrontendName,
    BuildSelector,
    call,
    combine_constraints,
    download,
    find_compatible_wheel,
    get_build_verbosity_extra_flags,
    get_pip_version,
    move_file,
    prepare_command,
    read_python_configs,
    shell,
    split_config_settings,
)


@dataclass(frozen=True)
class PythonConfiguration:
    version: str
    identifier: str
    url: str
    build_url: str

    @property
    def sdk(self):
        return self.identifier.split("-")[1].rsplit("_", 1)[1]

    @property
    def arch(self):
        return self.identifier.split("-")[1].split("_", 1)[1].rsplit("_", 1)[0]

    @property
    def multiarch(self):
        return f"{self.arch}-{self.sdk}"

    @property
    def slice(self):
        return {
            "iphoneos": "ios-arm64",
            "iphonesimulator": "ios-arm64_x86_64-simulator",
        }[self.sdk]


def get_python_configurations(
    build_selector: BuildSelector,
    architectures: Set[Architecture],
) -> list[PythonConfiguration]:
    # iOS builds are always cross builds; we need to install a macOS Python as
    # well. Load the macos configurations, and determine the macOS configuration
    # that matches the platform we're building.
    macos_python_configs = read_python_configs("macos")

    def build_url(item):
        # Extract the URL from the macOS configuration that matches
        # the provided iOS configuration item.
        macos_identifier = item["identifier"].rsplit("_", 1)[0].replace("ios", "macosx")
        matching = [
            config
            for config in macos_python_configs
            if config["identifier"] == macos_identifier
        ]
        return matching[0]["url"]

    # Load the platform configuration (iphoneos or iphonesimulator)
    full_python_configs = read_python_configs("ios")

    python_configurations = [
        PythonConfiguration(
            **item,
            build_url=build_url(item),
        )
        for item in full_python_configs
    ]
    # filter out configs that don't match any of the selected architectures
    python_configurations = [
        c
        for c in python_configurations
        if any(c.identifier.rsplit('_', 1)[0].endswith(a.value) for a in architectures)
    ]

    # skip builds as required by BUILD/SKIP
    python_configurations = [c for c in python_configurations if build_selector(c.identifier)]

    return python_configurations


def install_host_cpython(tmp: Path, config: PythonConfiguration) -> Path:
    # Install an iOS build of CPython
    ios_python_tar_gz = config.url.rsplit("/", 1)[-1]
    extension = ".tar.gz"
    assert ios_python_tar_gz.endswith(extension)
    installation_path = CIBW_CACHE_PATH / ios_python_tar_gz[: -len(extension)]
    with FileLock(str(installation_path) + ".lock"):
        if not installation_path.exists():
            downloaded_tar_gz = tmp / ios_python_tar_gz
            download(config.url, downloaded_tar_gz)
            installation_path.mkdir(parents=True, exist_ok=True)
            call("tar", "-C", installation_path, "-xf", downloaded_tar_gz)
            downloaded_tar_gz.unlink()

    return (
        installation_path
        / "Python.xcframework"
        / config.slice
        / "bin"
        / f"python{config.version}"
    )


def setup_python(
    tmp: Path,
    python_configuration: PythonConfiguration,
    dependency_constraint_flags: Sequence[PathOrStr],
    environment: ParsedEnvironment,
    build_frontend: BuildFrontendName,
) -> tuple[Path, dict[str, str]]:

    tmp.mkdir()

    implementation_id = python_configuration.identifier.split("-")[0]
    log.step(f"Installing Build Python {implementation_id}...")
    if implementation_id.startswith("cp"):
        free_threading = "t-iphone" in python_configuration.identifier
        build_python = install_build_cpython(
            tmp, python_configuration.version, python_configuration.build_url, free_threading
        )
    else:
        msg = "Unknown Python implementation"
        raise ValueError(msg)

    assert (
        build_python.exists()
    ), f"{build_python.name} not found, has {list(build_python.parent.iterdir())}"

    log.step(f"Installing Host Python {implementation_id}...")
    if implementation_id.startswith("cp"):
        host_python = install_host_cpython(tmp, python_configuration)
    else:
        msg = "Unknown Python implementation"
        raise ValueError(msg)

    assert (
        host_python.exists()
    ), f"{host_python.name} not found, has {list(host_python.parent.iterdir())}"

    log.step("Creating cross build environment...")

    venv_path = tmp / "venv"
    env = cross_virtualenv(
        py_version=python_configuration.version,
        os_name="iOS",
        os_version=os.getenv("IPHONEOS_DEPLOYMENT_TARGET", "13.0"),
        multiarch=python_configuration.multiarch,
        arch=python_configuration.arch,
        sdk=python_configuration.sdk,
        host_python=host_python,
        build_python=build_python,
        venv_path=venv_path,
        dependency_constraint_flags=dependency_constraint_flags,
    )
    venv_bin_path = venv_path / "bin"
    assert venv_bin_path.exists()

    # we version pip ourselves, so we don't care about pip version checking
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    # upgrade pip to the version matching our constraints
    call(
        "python",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        *dependency_constraint_flags,
        env=env,
        cwd=venv_path,
    )

    # Apply our environment after pip is ready
    env = environment.as_dictionary(prev_environment=env)

    # check what pip version we're on
    assert (venv_bin_path / "pip").exists()
    which_pip = call("which", "pip", env=env, capture_stdout=True).strip()
    if which_pip != str(venv_bin_path / "pip"):
        msg = (
            "cibuildwheel: pip available on PATH doesn't match our installed "
            "instance. If you have modified PATH, ensure that you don't "
            "overwrite cibuildwheel's entry or insert pip above it."
        )
        raise errors.FatalError(msg)

    # check what Python version we're on
    which_python = call("which", "python", env=env, capture_stdout=True).strip()
    if which_python != str(venv_bin_path / "python"):
        msg = (
            "cibuildwheel: python available on PATH doesn't match our "
            "installed instance. If you have modified PATH, ensure that you "
            "don't overwrite cibuildwheel's entry or insert python above it."
        )
        raise errors.FatalError(msg)

    log.step("Installing build tools...")
    if build_frontend == "pip":
        pass
        # call(
        #     "pip",
        #     "install",
        #     "--upgrade",
        #     *dependency_constraint_flags,
        #     env=env,
        # )
    elif build_frontend == "build":
        call(
            "pip",
            "install",
            "--upgrade",
            "build[virtualenv]",
            *dependency_constraint_flags,
            env=env,
        )
    else:
        assert_never(build_frontend)

    return build_python, env


def build(options: Options, tmp_path: Path) -> None:
    python_configurations = get_python_configurations(
        options.globals.build_selector, options.globals.architectures
    )

    if not python_configurations:
        return

    try:
        before_all_options_identifier = python_configurations[0].identifier
        before_all_options = options.build_options(before_all_options_identifier)

        if before_all_options.before_all:
            log.step("Running before_all...")
            env = before_all_options.environment.as_dictionary(prev_environment=os.environ)
            env.setdefault("IPHONEOS_DEPLOYMENT_TARGET", "13.0")
            before_all_prepared = prepare_command(
                before_all_options.before_all, project=".", package=before_all_options.package_dir
            )
            shell(before_all_prepared, env=env)

        built_wheels: list[Path] = []

        for config in python_configurations:
            build_options = options.build_options(config.identifier)
            build_frontend = build_options.build_frontend or BuildFrontendConfig("pip")
            log.build_start(config.identifier)

            identifier_tmp_dir = tmp_path / config.identifier
            identifier_tmp_dir.mkdir()
            built_wheel_dir = identifier_tmp_dir / "built_wheel"

            dependency_constraint_flags: Sequence[PathOrStr] = []
            if build_options.dependency_constraints:
                dependency_constraint_flags = [
                    "-c",
                    build_options.dependency_constraints.get_for_python_version(config.version),
                ]

            build_python, env = setup_python(
                identifier_tmp_dir / "build",
                config,
                dependency_constraint_flags,
                build_options.environment,
                build_frontend.name,
            )
            pip_version = get_pip_version(env)

            compatible_wheel = find_compatible_wheel(built_wheels, config.identifier)
            if compatible_wheel:
                log.step_end()
                print(
                    f"\nFound previously built wheel {compatible_wheel.name}, that's compatible with {config.identifier}. Skipping build step..."
                )
            else:
                if build_options.before_build:
                    log.step("Running before_build...")
                    before_build_prepared = prepare_command(
                        build_options.before_build, project=".", package=build_options.package_dir
                    )
                    shell(before_build_prepared, env=env)

                log.step("Building wheel...")
                built_wheel_dir.mkdir()

                extra_flags = split_config_settings(
                    build_options.config_settings, build_frontend.name
                )
                extra_flags += build_frontend.args

                build_env = env.copy()
                build_env["VIRTUALENV_PIP"] = pip_version
                if build_options.dependency_constraints:
                    constraint_path = build_options.dependency_constraints.get_for_python_version(
                        config.version
                    )
                    combine_constraints(build_env, constraint_path, None)

                if build_frontend.name == "pip":
                    extra_flags += get_build_verbosity_extra_flags(build_options.build_verbosity)
                    # Path.resolve() is needed. Without it pip wheel may try to fetch package from pypi.org
                    # see https://github.com/pypa/cibuildwheel/pull/369
                    call(
                        "python",
                        "-m",
                        "pip",
                        "wheel",
                        build_options.package_dir.resolve(),
                        f"--wheel-dir={built_wheel_dir}",
                        "--no-deps",
                        *extra_flags,
                        env=build_env,
                    )
                elif build_frontend.name == "build":
                    if not 0 <= build_options.build_verbosity < 2:
                        msg = f"build_verbosity {build_options.build_verbosity} is not supported for build frontend. Ignoring."
                        log.warning(msg)

                    call(
                        "python",
                        "-m",
                        "build",
                        build_options.package_dir,
                        "--wheel",
                        f"--outdir={built_wheel_dir}",
                        *extra_flags,
                        env=build_env,
                    )
                else:
                    assert_never(build_frontend)

                built_wheel = next(built_wheel_dir.glob("*.whl"))

                if built_wheel.name.endswith("none-any.whl"):
                    raise errors.NonPlatformWheelError()

                log.step_end()

            if build_options.test_command and build_options.test_selector(config.identifier):
                log.step("Testing wheel...")

                if build_options.before_test:
                    before_test_prepared = prepare_command(
                        build_options.before_test,
                        project=".",
                        package=build_options.package_dir,
                    )
                    shell_with_arch(before_test_prepared, env=virtualenv_env)

            #         # install the wheel
            #         if is_cp38 and python_arch == "x86_64":
            #             virtualenv_env_install_wheel = virtualenv_env.copy()
            #             virtualenv_env_install_wheel["SYSTEM_VERSION_COMPAT"] = "0"
            #             log.notice(
            #                 unwrap(
            #                     """
            #                     Setting SYSTEM_VERSION_COMPAT=0 to ensure CPython 3.8 can get
            #                     correct macOS version and allow installation of wheels with
            #                     MACOSX_DEPLOYMENT_TARGET >= 11.0.
            #                     See https://github.com/pypa/cibuildwheel/issues/1767 for the
            #                     details.
            #                     """
            #                 )
            #             )
            #         else:
            #             virtualenv_env_install_wheel = virtualenv_env

            #         pip_install(
            #             f"{repaired_wheel}{build_options.test_extras}",
            #             env=virtualenv_env_install_wheel,
            #         )

            #         # test the wheel
            #         if build_options.test_requires:
            #             pip_install(
            #                 *build_options.test_requires,
            #                 env=virtualenv_env_install_wheel,
            #             )

            #         # run the tests from a temp dir, with an absolute path in the command
            #         # (this ensures that Python runs the tests against the installed wheel
            #         # and not the repo code)
            #         test_command_prepared = prepare_command(
            #             build_options.test_command,
            #             project=Path(".").resolve(),
            #             package=build_options.package_dir.resolve(),
            #             wheel=repaired_wheel,
            #         )

            #         test_cwd = identifier_tmp_dir / "test_cwd"
            #         test_cwd.mkdir(exist_ok=True)
            #         (test_cwd / "test_fail.py").write_text(test_fail_cwd_file.read_text())

            #         shell_with_arch(test_command_prepared, cwd=test_cwd, env=virtualenv_env)

            # we're all done here; move it to output (overwrite existing)
            if compatible_wheel is None:
                output_wheel = build_options.output_dir.joinpath(built_wheel.name)
                moved_wheel = move_file(built_wheel, output_wheel)
                if moved_wheel != output_wheel.resolve():
                    log.warning(
                        "{built_wheel} was moved to {moved_wheel} instead of {output_wheel}"
                    )
                built_wheels.append(output_wheel)

            # clean up
            shutil.rmtree(identifier_tmp_dir)

            log.build_end()
    except subprocess.CalledProcessError as error:
        msg = f"Command {error.cmd} failed with code {error.returncode}. {error.stdout or ''}"
        raise errors.FatalError(msg) from error
