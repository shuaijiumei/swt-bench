import logging
import re
import subprocess
import traceback
from typing import Literal

import docker
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from docker.models.containers import Container

from src.constants import (
    BASE_IMAGE_BUILD_DIR,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    MAP_VERSION_TO_INSTALL,
)
from src.test_spec import (
    make_test_spec,
    TestSpec
)
from src.docker_utils import (
    cleanup_container,
    remove_image,
    find_dependent_images
)

from src.exec_spec import (ExecSpec,
                                        get_exec_specs_from_dataset,)

from src.utils import Locker, setup_logger, close_logger

ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        log_msg = traceback.format_exc()
        self.logger.info(log_msg)
        return (
            f"{self.image_name}: {super().__str__()}\n"
            f"Check ({self.log_path}) for more information."
        )

BuildMode = Literal["cli", "api"]
ExecMode = Literal["unit_test", "reproduction_script"]

def docker_build_cli(
    build_dir: Path,
    image_name: str,
    platform: str,
    nocache: bool = False
):
    response = subprocess.Popen(
        ["docker", "build", "--tag", image_name, ".", "--platform", platform] + (["--no-cache"] if nocache else []),
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=build_dir,
    )
    for line in response.stdout:
        yield line.strip()
    response.wait()
    if response.returncode != 0:
        raise RuntimeError("Failed to build image")

def docker_build_api(
        build_dir: Path,
        image_name: str,
        platform: str,
        client: docker.DockerClient,
        nocache: bool = False,
):
    response = client.api.build(
        path=str(build_dir),
        tag=image_name,
        rm=True,
        forcerm=True,
        decode=True,
        platform=platform,
        nocache=nocache,
    )

    # Log the build process continuously
    buildlog = ""
    for chunk in response:
        if "stream" in chunk:
            # Remove ANSI escape sequences from the log
            chunk_stream = ansi_escape.sub("", chunk["stream"])
            yield chunk_stream.strip()
            buildlog += chunk_stream
        elif "errorDetail" in chunk:
            # Decode error message, raise BuildError
            raise docker.errors.BuildError(
                ansi_escape.sub("", chunk["errorDetail"]["message"]), buildlog
            )


def build_image(
        image_name: str,
        setup_scripts: dict,
        dockerfile: str,
        platform: str,
        client: docker.DockerClient,
        build_dir: Path,
        nocache: bool = False,
        build_mode: BuildMode = "api",
    ):
    """
    Builds a docker image with the given name, setup scripts, dockerfile, and platform.

    Args:
        image_name (str): Name of the image to build
        setup_scripts (dict): Dictionary of setup script names to setup script contents
        dockerfile (str): Contents of the Dockerfile
        platform (str): Platform to build the image for
        client (docker.DockerClient): Docker client to use for building the image
        build_dir (Path): Directory for the build context (will also contain logs, scripts, and artifacts)
        nocache (bool): Whether to use the cache when building
    """
    # Create a logger for the build process
    logger = setup_logger(image_name, build_dir / "build_image.log", "w")
    logger.info(
        f"Building image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
    try:
        # Write the setup scripts to the build directory
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )

        # Write the dockerfile to the build directory
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        # Build the image
        if build_mode == "cli":
            response = docker_build_cli(
                build_dir,
                image_name,
                platform,
                nocache,
            )
        else:
            response = docker_build_api(
                build_dir,
                image_name,
                platform,
                client,
                nocache
            )

        # Log the build process continuously
        for line in response:
            logger.info(line)
        logger.info("Image built successfully!")
    except docker.errors.BuildError as e:
        logger.error(f"docker.errors.BuildError during {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        close_logger(logger)  # functions that create loggers should close them


def build_base_images(
        client: docker.DockerClient,
        dataset: list,
        force_rebuild: bool = False,
        build_mode: BuildMode = "api",
    ):
    """
    Builds the base images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
    """
    # Get the base images to build from the dataset
    exec_specs = get_exec_specs_from_dataset(dataset)
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in exec_specs
    }
    if force_rebuild:
        for key in base_images:
            remove_image(client, key, "quiet")

    # Build the base images
    for image_name, (dockerfile, platform) in base_images.items():
        try:
            # Check if the base image already exists
            client.images.get(image_name)
            if force_rebuild:
                # Remove the base image if it exists and force rebuild is enabled
                remove_image(client, image_name, "quiet")
            else:
                print(f"Base image {image_name} already exists, skipping build.")
                continue
        except docker.errors.ImageNotFound:
            pass
        # Build the base image (if it does not exist or force rebuild is enabled)
        print(f"Building base image ({image_name})")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
            build_mode=build_mode,
        )
    print("Base images built successfully.")


def build_base_image_from_exec_spec(
        exec_spec: ExecSpec,
        force_rebuild: bool = False,
        build_mode: BuildMode = "api",
    ):
    """
    Builds the base image required for the exec_spec if it does not already exist.

    Args:
        exec_spec
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
    """
    client = docker.from_env()

    image_name = exec_spec.base_image_key
    dockerfile = exec_spec.base_dockerfile
    platform = exec_spec.platform

    if force_rebuild:
        remove_image(client, image_name, "quiet")

    with Locker(f"./locks/{image_name}_"): # prevent other processes from checking whether image exists while we are building. Allow for async wait (?)
        try:
            # Check if the base image already exists
            client.images.get(image_name)

            print(f"Base image {image_name} already exists, skipping build.")
            return
        except docker.errors.ImageNotFound:
            pass


        # Build the base image (if it does not exist or force rebuild is enabled)
        print(f"Building base image {image_name}")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
            build_mode=build_mode,
        )

        print(f"Base image {image_name} built successfully.")


def build_env_image_from_exec_spec(
        exec_spec: ExecSpec,
        force_rebuild: bool = False,
        build_mode: BuildMode = "api",
    ):
    """
    Builds the env image required for the exec_spec if it does not already exist.

    Args:
        exec_spec
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
    """
    client = docker.from_env()

    image_name = exec_spec.env_image_key

    if force_rebuild:
        remove_image(client, image_name, "quiet")

    build_base_image_from_exec_spec(exec_spec, force_rebuild, build_mode=build_mode)

    try:
        base_image = client.images.get(exec_spec.base_image_key)
    except docker.errors.ImageNotFound:
        raise Exception(
            f"Base image {exec_spec.base_image_key} not found for {exec_spec.env_image_key}\n."
            "Please build the base images first."
        )

    setup_script = exec_spec.env_script
    dockerfile = exec_spec.env_dockerfile
    platform = exec_spec.platform

    with Locker(f"./locks/{image_name}_"): # prevent other processes from checking whether image exists while we are building. Allow for async wait (?)
        try:
            env_image = client.images.get(exec_spec.env_image_key)
            if env_image.attrs["Created"] < base_image.attrs["Created"]:
                # Remove the environment image if it was built after the base_image
                for dep in find_dependent_images(env_image):
                    # Remove instance images that depend on this environment image
                    remove_image(client, dep.image_id, "quiet")
                remove_image(client, exec_spec.env_image_key, "quiet")
            print(f"Env image {image_name} already exists, skipping build.")
            return
        except docker.errors.ImageNotFound:
            pass

        # Build the env image (if it does not exist or force rebuild is enabled)
        print(f"Building env image {image_name}")
        build_image(
            image_name=image_name,
            setup_scripts={"setup_env.sh": setup_script},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
            build_mode=build_mode,
        )

        print(f"Env image {image_name} built successfully.")


def get_env_configs_to_build(
        client: docker.DockerClient,
        dataset: list,
    ):
    """
    Returns a dictionary of image names to build scripts and dockerfiles for environment images.
    Returns only the environment images that need to be built.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
    """
    image_scripts = dict()
    base_images = dict()
    exec_specs = get_exec_specs_from_dataset(dataset)

    for exec_spec in exec_specs:
        # Check if the base image exists
        try:
            if exec_spec.base_image_key not in base_images:
                base_images[exec_spec.base_image_key] = client.images.get(
                    exec_spec.base_image_key
                )
            base_image = base_images[exec_spec.base_image_key]
        except docker.errors.ImageNotFound:
            raise Exception(
                f"Base image {exec_spec.base_image_key} not found for {exec_spec.env_image_key}\n."
                "Please build the base images first."
            )

        # Check if the environment image exists
        image_exists = False
        try:
            env_image = client.images.get(exec_spec.env_image_key)
            image_exists = True

            if env_image.attrs["Created"] < base_image.attrs["Created"]:
                # Remove the environment image if it was built after the base_image
                for dep in find_dependent_images(env_image):
                    # Remove instance images that depend on this environment image
                    remove_image(client, dep.image_id, "quiet")
                remove_image(client, exec_spec.env_image_key, "quiet")
                image_exists = False
        except docker.errors.ImageNotFound:
            pass
        if not image_exists:
            # Add the environment image to the list of images to build
            image_scripts[exec_spec.env_image_key] = {
                "setup_script": exec_spec.env_script,
                "dockerfile": exec_spec.env_dockerfile,
                "platform": exec_spec.platform,
            }
    return image_scripts


def build_env_images(
        client: docker.DockerClient,
        dataset: list,
        force_rebuild: bool = False,
        max_workers: int = 4,
        build_mode: BuildMode = "api",
    ):
    """
    Builds the environment images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Get the environment images to build from the dataset
    if force_rebuild:
        env_image_keys = {x.env_image_key for x in get_exec_specs_from_dataset()}
        for key in env_image_keys:
            remove_image(client, key, "quiet")
    build_base_images(client, dataset, force_rebuild, build_mode)
    configs_to_build = get_env_configs_to_build(client, dataset)
    if len(configs_to_build) == 0:
        print("No environment images need to be built.")
        return
    print(f"Total environment images to build: {len(configs_to_build)}")

    # Build the environment images
    successful, failed = list(), list()
    with tqdm(
        total=len(configs_to_build), smoothing=0, desc="Building environment images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each image to build
            futures = {
                executor.submit(
                    build_image,
                    image_name,
                    {"setup_env.sh": config["setup_script"]},
                    config["dockerfile"],
                    config["platform"],
                    client,
                    ENV_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
                ): image_name
                for image_name, config in configs_to_build.items()
            }

            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if image built successfully
                    future.result()
                    successful.append(futures[future])
                except BuildImageError as e:
                    print(f"BuildImageError {e.image_name}")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue
                except Exception as e:
                    print(f"Error building image")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue

    # Show how many images failed to build
    if len(failed) == 0:
        print("All environment images built successfully.")
    else:
        print(f"{len(failed)} environment images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_images(
        client: docker.DockerClient,
        dataset: list,
        force_rebuild: bool = False,
        max_workers: int = 4,
        build_mode: BuildMode = "api",
    ):
    """
    Builds the instance images required for the dataset if they do not already exist.
    
    Args:
        dataset (list): List of test specs or dataset to build images for
        client (docker.DockerClient): Docker client to use for building the images
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Build environment images (and base images as needed) first
    test_specs = list(map(make_test_spec, dataset))
    if force_rebuild:
        for spec in test_specs:
            remove_image(client, spec.instance_image_key, "quiet")
    _, env_failed = build_env_images(client, test_specs, force_rebuild, max_workers, build_mode=build_mode)

    if len(env_failed) > 0:
        # Don't build images for instances that depend on failed-to-build env images
        dont_run_specs = [spec for spec in test_specs if spec.env_image_key in env_failed]
        test_specs = [spec for spec in test_specs if spec.env_image_key not in env_failed]
        print(f"Skipping {len(dont_run_specs)} instances - due to failed env image builds")
    print(f"Building instance images for {len(test_specs)} instances")
    successful, failed = list(), list()

    # Build the instance images
    with tqdm(
        total=len(test_specs), smoothing=0, desc="Building instance images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each image to build
            futures = {
                executor.submit(
                    build_instance_image,
                    test_spec,
                    client,
                    None,  # logger is created in build_instance_image, don't make loggers before you need them
                    False,
                ): test_spec
                for test_spec in test_specs
            }

            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if image built successfully
                    future.result()
                    successful.append(futures[future])
                except BuildImageError as e:
                    print(f"BuildImageError {e.image_name}")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue
                except Exception as e:
                    print(f"Error building image")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue

    # Show how many images failed to build
    if len(failed) == 0:
        print("All instance images built successfully.")
    else:
        print(f"{len(failed)} instance images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_image_from_exec_spec(
        exec_spec: ExecSpec,
        force_rebuild: bool,
        build_mode: BuildMode = "api",
    ):
    """
    Builds the instance image for the given test spec if it does not already exist.

    Args:
        test_spec (TestSpec): Test spec to build the instance image for
        client (docker.DockerClient): Docker client to use for building the image
        nocache (bool): Whether to use the cache when building
    """
    client = docker.from_env()

    # Get the image names and dockerfile for the instance image
    image_name = exec_spec.instance_image_key
    env_image_name = exec_spec.env_image_key
    dockerfile = exec_spec.instance_dockerfile

    if force_rebuild:
        remove_image(client, image_name, "quiet")

    build_env_image_from_exec_spec(exec_spec, force_rebuild, build_mode=build_mode)

    # Set up logging for the build process
    build_dir = INSTANCE_IMAGE_BUILD_DIR / exec_spec.instance_image_key.replace(":", "__")
    logger = setup_logger(exec_spec.instance_id, build_dir / "prepare_image.log", "w")

    # Check that the env. image the instance image is based on exists
    try:
        env_image = client.images.get(env_image_name)
    except docker.errors.ImageNotFound as e:
        raise BuildImageError(
            exec_spec.instance_id,
            f"Environment image {env_image_name} not found for {exec_spec.instance_id}",
            logger,
        ) from e
    logger.info(
        f"Environment image {env_image_name} found for {exec_spec.instance_id}\n"
        f"Building instance image {image_name} for {exec_spec.instance_id}"
    )

    with Locker(f"./locks/{image_name}_"): # prevent other processes from checking whether image exists while we are building. Allow for async wait (?)
        try:
            instance_image = client.images.get(image_name)
            if instance_image.attrs["Created"] < env_image.attrs["Created"]:
                # the environment image is newer than the instance image, meaning the instance image may be outdated
                remove_image(client, image_name, "quiet")
            else:
                logger.info(f"Image {image_name} already exists, skipping build.")
                close_logger(logger)
                return
        except docker.errors.ImageNotFound:
            pass

        # Build the instance image
        build_image(
            image_name=image_name,
            setup_scripts={
                "setup_repo.sh": exec_spec.repo_script,
            },
            dockerfile=dockerfile,
            platform=exec_spec.platform,
            client=client,
            build_dir=build_dir,
            build_mode=build_mode,
        )
        close_logger(logger)


def build_instance_image(
        exec_spec: ExecSpec,
        client: docker.DockerClient,
        logger: logging.Logger,
        nocache: bool,
        build_mode: BuildMode = "api",
    ):
    """
    Builds the instance image for the given test spec if it does not already exist.

    Args:
        test_spec (TestSpec): Test spec to build the instance image for
        client (docker.DockerClient): Docker client to use for building the image
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
    """

    # Get the image names and dockerfile for the instance image
    image_name = exec_spec.instance_image_key
    env_image_name = exec_spec.env_image_key
    dockerfile = exec_spec.instance_dockerfile

    # Set up logging for the build process
    build_dir = INSTANCE_IMAGE_BUILD_DIR / exec_spec.instance_image_key.replace(":", "__")
    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(exec_spec.instance_id, build_dir / "prepare_image.log","w")




    # Check that the env. image the instance image is based on exists
    try:
        env_image = client.images.get(env_image_name)
    except docker.errors.ImageNotFound as e:
        raise BuildImageError(
            exec_spec.instance_id,
            f"Environment image {env_image_name} not found for {exec_spec.instance_id}",
            logger,
        ) from e
    logger.info(
        f"Environment image {env_image_name} found for {exec_spec.instance_id}\n"
        f"Building instance image {image_name} for {exec_spec.instance_id}"
    )

    # Check if the instance image already exists
    image_exists = False
    try:
        instance_image = client.images.get(image_name)
        if instance_image.attrs["Created"] < env_image.attrs["Created"]:
            # the environment image is newer than the instance image, meaning the instance image may be outdated
            remove_image(client, image_name, "quiet")
            image_exists = False
        else:
            image_exists = True
    except docker.errors.ImageNotFound:
        pass

    # Build the instance image
    if not image_exists:
        build_image(
            image_name=image_name,
            setup_scripts={
                "setup_repo.sh": exec_spec.repo_script,
            },
            dockerfile=dockerfile,
            platform=exec_spec.platform,
            client=client,
            build_dir=build_dir,
            nocache=nocache,
            build_mode=build_mode,
        )
    else:
        logger.info(f"Image {image_name} already exists, skipping build.")

    if new_logger:
        close_logger(logger)


def build_container(
        exec_spec: ExecSpec,
        client: docker.DockerClient,
        logger: logging.Logger,
        nocache: bool,
        force_rebuild: bool = False,
        build_mode: BuildMode = "api",
    ):
    """
    Builds the instance image for the given test spec and creates a container from the image.

    Args:
        test_spec (TestSpec): Test spec to build the instance image and container for
        client (docker.DockerClient): Docker client for building image + creating the container
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
        force_rebuild (bool): Whether to force rebuild the image even if it already exists
    """
    # Build corresponding instance image
    if force_rebuild:
        remove_image(client, exec_spec.instance_image_key, "quiet")
    # build_instance_image(exec_spec, client, logger, nocache)
    build_instance_image_from_exec_spec(exec_spec, force_rebuild, build_mode)

    container = None
    try:
        # Get configurations for how container should be created
        config = MAP_VERSION_TO_INSTALL[exec_spec.repo][exec_spec.version]
        user = "root" if not config.get("execute_test_as_nonroot", False) else "nonroot"
        nano_cpus = config.get("nano_cpus")

        # Create the container
        logger.info(f"Creating container for {exec_spec.instance_id}...")
        container = client.containers.create(
            image=exec_spec.instance_image_key,
            name=exec_spec.get_instance_container_name(),
            user=user,
            detach=True,
            command="tail -f /dev/null",
            nano_cpus=nano_cpus,
            platform=exec_spec.platform,
        )
        logger.info(f"Container for {exec_spec.instance_id} created: {container.id}")
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f"Error creating container for {exec_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise BuildImageError(exec_spec.instance_id, str(e), logger) from e


def start_container(exec_spec: ExecSpec, client: docker.DockerClient, logger: logging.Logger, build_mode: BuildMode = "api") -> Container:
    # Build + start instance container (instance image should already be built)
    container = build_container(exec_spec, client, logger, exec_spec.rm_image, exec_spec.force_rebuild, build_mode)
    container.start()
    logger.info(f"Container for {exec_spec.instance_id} started: {container.id}")
    return container