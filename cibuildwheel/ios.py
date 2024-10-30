from __future__ import annotations

import json
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
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
    copy_test_sources,
    download,
    extract_tar,
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
    def is_simulator(self):
        return self.sdk.endswith("simulator")

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
    # well. Rather than duplicate the location of the URL of macOS installers,
    # Load the macos configurations, and determine the macOS configuration
    # that matches the platform we're building, and embed that URL in the parsed
    # iOS configuration.
    macos_python_configs = read_python_configs("macos")

    def build_url(item):
        # The iOS item will be something like cp313-ios_arm64_iphoneos. Drop
        # the iphoneos suffix, then replace ios with macosx to yield
        # cp313-macosx_arm64, which will be a macOS configuration item.
        macos_identifier = item["identifier"].rsplit("_", 1)[0]
        macos_identifier = macos_identifier.replace("ios", "macosx")
        matching = [
            config for config in macos_python_configs if config["identifier"] == macos_identifier
        ]
        return matching[0]["url"]

    # Load the platform configuration
    full_python_configs = read_python_configs("ios")

    # Build the configurations, annotating with macOS URL details.
    python_configurations = [
        PythonConfiguration(
            **item,
            build_url=build_url(item),
        )
        for item in full_python_configs
    ]

    # Filter out configs that don't match any of the selected architectures
    python_configurations = [
        c
        for c in python_configurations
        if any(c.identifier.rsplit("_", 1)[0].endswith(a.value) for a in architectures)
    ]

    # Skip builds as required by BUILD/SKIP
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

    return installation_path


def setup_python(
    tmp: Path,
    python_configuration: PythonConfiguration,
    dependency_constraint_flags: Sequence[PathOrStr],
    environment: ParsedEnvironment,
    build_frontend: BuildFrontendName,
) -> tuple[Path, dict[str, str]]:
    # An iOS environment requires 2 python installs - one for the build machine
    # (macOS), and one for the host (iOS). We'll only ever interact with the
    # *host* python, but the build Python needs to exist to act as the base
    # for a cross venv.
    tmp.mkdir()

    implementation_id = python_configuration.identifier.split("-")[0]
    log.step(f"Installing Build Python {implementation_id}...")
    if implementation_id.startswith("cp"):
        free_threading = "t-iphone" in python_configuration.identifier
        build_python = install_build_cpython(
            tmp,
            python_configuration.version,
            python_configuration.build_url,
            free_threading,
        )
    else:
        msg = "Unknown Python implementation"
        raise ValueError(msg)

    assert (
        build_python.exists()
    ), f"{build_python.name} not found, has {list(build_python.parent.iterdir())}"

    log.step(f"Installing Host Python {implementation_id}...")
    if implementation_id.startswith("cp"):
        host_install_path = install_host_cpython(tmp, python_configuration)
        host_python = (
            host_install_path
            / "Python.xcframework"
            / python_configuration.slice
            / "bin"
            / f"python{python_configuration.version}"
        )
    else:
        msg = "Unknown Python implementation"
        raise ValueError(msg)

    assert (
        host_python.exists()
    ), f"{host_python.name} not found, has {list(host_install_path.iterdir())}"

    log.step("Creating cross build environment...")

    ios_deployment_target = os.getenv("IPHONEOS_DEPLOYMENT_TARGET", "13.0")

    venv_path = tmp / "venv"
    env = cross_virtualenv(
        py_version=python_configuration.version,
        os_name="iOS",
        os_version=ios_deployment_target,
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

    # We version pip ourselves, so we don't care about pip version checking
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    # Ensure that IPHONEOS_DEPLOYMENT_TARGET is set in the environment
    env["IPHONEOS_DEPLOYMENT_TARGET"] = ios_deployment_target

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
            "cibuildwheel: pip available on PATH doesn't match our installed instance. "
            "If you have modified PATH, ensure that you don't overwrite cibuildwheel's "
            "entry or insert pip above it."
        )
        raise errors.FatalError(msg)

    # check what Python version we're on
    which_python = call("which", "python", env=env, capture_stdout=True).strip()
    if which_python != str(venv_bin_path / "python"):
        msg = (
            "cibuildwheel: python available on PATH doesn't match our installed instance. "
            "If you have modified PATH, ensure that you don't overwrite cibuildwheel's "
            "entry or insert python above it."
        )
        raise errors.FatalError(msg)

    log.step("Installing build tools...")
    if build_frontend == "pip":
        # No additional build tools required
        pass
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

    return host_install_path, env


def extract_test_output(xcresult: Path) -> str:
    """Extract stdout content from an Xcode xcresult bundle."""
    try:
        # First, get the ID of the test run
        raw = call(
            "xcrun",
            "xcresulttool",
            "get",
            "--path",
            xcresult,
            "--format",
            "json",
            capture_stdout=True,
        )
        parsed = json.loads(raw)
        action_result = parsed["actions"]["_values"][0]["actionResult"]
        test_id = action_result["logRef"]["id"]["_value"]
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Unable to call xcresulttool on {xcresult}")
    except (KeyError, IndexError):
        raise RuntimeError(f"Unable to extract test ID from {xcresult}")

    # Then, extract the stdout content for the test ID.
    try:
        raw = call(
            "xcrun",
            "xcresulttool",
            "get",
            "--path",
            xcresult,
            "--id",
            test_id,
            "--format",
            "json",
            capture_stdout=True,
        )
        parsed = json.loads(raw)
        subsections = parsed["subsections"]["_values"][1]["subsections"]
        test_output = subsections["_values"][0]["emittedOutput"]["_value"]
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Unable to call xcresulttool on {xcresult}")
    except (KeyError, IndexError):
        raise RuntimeError(f"Unable to extract test output from {xcresult}")

    return test_output


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
                before_all_options.before_all,
                project=".",
                package=before_all_options.package_dir,
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

            host_install_path, env = setup_python(
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
                    f"\nFound previously built wheel {compatible_wheel.name} "
                    f"that is compatible with {config.identifier}. "
                    "Skipping build step..."
                )
                test_wheel = compatible_wheel
            else:
                if build_options.before_build:
                    log.step("Running before_build...")
                    before_build_prepared = prepare_command(
                        build_options.before_build,
                        project=".",
                        package=build_options.package_dir,
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
                        log.warning(
                            f"build_verbosity {build_options.build_verbosity} is "
                            "not supported for build frontend. Ignoring."
                        )

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

                test_wheel = built_wheel = next(built_wheel_dir.glob("*.whl"))

                if built_wheel.name.endswith("none-any.whl"):
                    raise errors.NonPlatformWheelError()

                log.step_end()

            if build_options.test_command and build_options.test_selector(config.identifier):
                if not config.is_simulator:
                    log.step("Skipping tests on non-simulator SDK")
                elif config.arch != os.uname().machine:
                    log.step("Skipping tests on non-native simulator architecture")
                else:
                    if build_options.before_test:
                        log.step("Running before_test...")
                        before_test_prepared = prepare_command(
                            build_options.before_test,
                            project=".",
                            package=build_options.package_dir,
                        )
                        shell(before_test_prepared, env=env)

                    log.step("Setting up test harness...")
                    # Copy the stub testbed project into the build directory
                    testbed_path = identifier_tmp_dir / "testbed"
                    shutil.copytree(
                        # FIXME: bundle the testbed with the host python
                        # host_install_path / testbed,
                        Path(__file__).parent / "resources/ios-testbed",
                        testbed_path,
                    )

                    # Install the Python XCframework into the stub testbed
                    (testbed_path / "Python.xcframework").symlink_to(
                        host_install_path / "Python.xcframework"
                    )

                    if build_options.test_sources:
                        copy_test_sources(
                            build_options.test_sources,
                            build_options.package_dir,
                            testbed_path / "iOSTestbed" / "app"
                        )
                    else:
                        # Copy *all* the test sources; however use the sdist
                        # to do this so that we avoid copying any .git or venv folders.

                        # Build a sdist of the project
                        call(
                            "python",
                            "-m",
                            "build",
                            build_options.package_dir,
                            "--sdist",
                            f"--outdir={identifier_tmp_dir}",
                            capture_stdout=True,
                        )
                        src_tarball = next(identifier_tmp_dir.glob("*.tar.gz"))

                        # Unpack the source tarball into the stub testbed
                        extract_tar(
                            src_tarball,
                            testbed_path / "iOSTestbed" / "app",
                            strip=1,
                        )

                    # Add the test runner arguments to the testbed's Info.plist file.
                    info_plist = testbed_path / "iOSTestbed" / "iOSTestbed-Info.plist"
                    with info_plist.open("rb") as f:
                        info = plistlib.load(f)

                    info["TestArgs"] = shlex.split(build_options.test_command)

                    with info_plist.open("wb") as f:
                        plistlib.dump(info, f)

                    log.step("Installing test requirements...")
                    # Install the compiled wheel (with any test extras), plus
                    # the test requirements. Use the --platform tag to force
                    # the installtion of iOS wheels; this requires the use of
                    # --only-binary=:all:.
                    ios_version = build_env["IPHONEOS_DEPLOYMENT_TARGET"]
                    platform_tag = f"ios_{ios_version.replace(".", "_")}_{config.arch}_{config.sdk}"

                    call(
                        "python",
                        "-m",
                        "pip",
                        "install",
                        "--only-binary=:all:",
                        "--platform",
                        platform_tag,
                        "--target",
                        testbed_path / "iOSTestbed" / "app_packages",
                        f"{test_wheel}{build_options.test_extras}",
                        *build_options.test_requires,
                        env=build_env,
                    )

                    log.step("Running test suite...")
                    xcresult = identifier_tmp_dir / "tests.xcresult"
                    try:
                        # Run the test suite. This will compile the testbed app,
                        # start a simulator, and run testbed in the simulator;
                        # but it won't display stdout of the test app as it runs.
                        # Xcode is really noisy; run in quiet mode unless verbosity
                        # has been requested.
                        if build_options.build_verbosity > 0:
                            test_cmd = ["xcodebuild", "test"]
                        else:
                            test_cmd = ["xcodebuild", "test", "-quiet"]

                        # The runtime behavior of the simulator doesn't change
                        # with the model - it only affects the screen size
                        # (which we don't care about). The iPhone SE 3rd gen is
                        # an "LTS" iPhone model, so we can rely on it existing.
                        simulator = "iPhone SE (3rd Generation)"

                        # Invoke xcodebuild. Provide a known location for the results
                        # (so we can process them later); also provide a derivedDataPath
                        # in the tmp folder so that it will be cleaned up on exit.
                        call(
                            *test_cmd,
                            "-project",
                            testbed_path / "iOSTestbed.xcodeproj",
                            "-scheme",
                            "iOSTestbed",
                            "-destination",
                            f"platform=iOS Simulator,name={simulator}",
                            "-resultBundlePath",
                            xcresult,
                            "-derivedDataPath",
                            identifier_tmp_dir / "DerivedData",
                        )
                        failed = False
                    except subprocess.CalledProcessError:
                        failed = True
                    finally:
                        # No matter whether the test passed or failed, extract
                        # stdout from the tests results, and display to the
                        # user.
                        print("\nExtracting test output...")
                        test_output = extract_test_output(xcresult)

                        # Write the test output to the screen
                        print("-" * 79)
                        print(test_output)

                    log.step_end(success=not failed)

                    if failed:
                        log.error(f"Test suite failed on {config.identifier}")
                        sys.exit(1)

            if compatible_wheel is None:
                output_wheel = build_options.output_dir.joinpath(built_wheel.name)
                moved_wheel = move_file(built_wheel, output_wheel)
                if moved_wheel != output_wheel.resolve():
                    log.warning(
                        f"{built_wheel} was moved to {moved_wheel} " f"instead of {output_wheel}"
                    )
                built_wheels.append(output_wheel)

            # Clean up
            shutil.rmtree(identifier_tmp_dir)

            log.build_end()
    except subprocess.CalledProcessError as error:
        msg = f"Command {error.cmd} failed with code {error.returncode}. {error.stdout or ''}"
        raise errors.FatalError(msg) from error
