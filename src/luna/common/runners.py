import spython.main

import docker


class DockerRunner:
    def __init__(self, image, command, volumes_map, num_cores, max_heap_size):
        self._image = image
        self._command = command
        self._volumes = {}
        self._num_cores = num_cores
        self._java_options = f"-Xmx{max_heap_size}"
        for k, v in volumes_map.items():
            self._volumes[k] = {"bind": v, "mode": "rw"}

    def run(self):
        client = docker.from_env()
        container = client.containers.run(
            volumes=self._volumes,
            nano_cpus=int(self._num_cores * 1e9),
            image=self._image,
            command=self._command,
            environment={"_JAVA_OPTIONS": self._java_options},
            detach=False,
            stream=True,
        )
        return container


class DockerRunnerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, image, command, volumes_map, num_cores, max_heap_size, **_ignored):
        if not self._instance:
            self._instance = DockerRunner(image, command, volumes_map, num_cores, max_heap_size)
        return self._instance


class SingularityRunner:
    def __init__(self, image, command, volumes_map, num_cores, use_gpu, max_heap_size):
        self._image = image
        self._command = command
        self._num_cores = num_cores
        self._use_gpu = use_gpu
        self._volumes = []
        self._java_options = f"-XX:ActiveProcessorCount={num_cores} -Xmx{max_heap_size}"
        for k, v in volumes_map.items():
            self._volumes.append(f"{k}:{v}")

    def run(self):
        executor = spython.main.Client.execute(
            image=self._image,
            command=self._command,
            bind=self._volumes,
            nv=self._use_gpu,
            options=["--env", f"_JAVA_OPTIONS={self._java_options}"],
            stream=True,
        )
        return executor


class SingularityRunnerBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, image, command, volumes_map, num_cores, use_gpu, max_heap_size, **_ignored):
        if not self._instance:
            self._instance = SingularityRunner(
                image, command, volumes_map, num_cores, use_gpu, max_heap_size
            )
        return self._instance


class RunnerFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


class RunnerProvider(RunnerFactory):
    def get(self, runner_type, **kwargs):
        return self.create(runner_type, **kwargs)


runner_provider = RunnerProvider()
runner_provider.register_builder("DOCKER", DockerRunnerBuilder())
runner_provider.register_builder("SINGULARITY", SingularityRunnerBuilder())
