---
title: Development Guide
---

A comprehensive guide for those interested in making contributions or
supporting the development of Luna.

**Warning:** This documentation is still a work in progress

# Development Setup Instructions

Follow the instructions below to install the latest version of Luna and
setup a development environment.

## Development Installation

Developers interested in using the newest features of Luna, or are
actively contributing to Luna should install the latest version
available via:

    git clone git@github.com:msk-mind/luna.git

The `Makefile` has a number of relevant targets. (Note: GNU
Make 3.82 or higher is required.)

See all targets:

    make help

Install virtual environment:

    make venv

OR use `setup_local.sh` to setup your python paths and `LUNA_HOME`
config:

    source ./setup_local.sh

### Troubleshooting

If the install process hangs on installing packages with poetry, you may have
to clear out your cache.

On MacOS:

    rm -rf ~/Library/Caches/pypoetry/cache/*

On Linux:

    rm -rf ~/.cache/pypoetry/*


## Development with Docker

Build docker image:

    make build-docker

The docker image is also available on DockerHub:
[luna](https://hub.docker.com/r/mskmind/luna).

This docker image includes the pre-requisites and python dependencies.

Once we have a stable release, docker images that includes pyluna
packages can be made available.

## Documentation Generation

Documentation in Luna is generated via `mkdocs`. See the `mkdocs.yml` file for
specific configuration details. Note that the `awesome-pages` plugin is
installed and navigation is specified in `.pages` files.

Docstrings are written according to the [Google Python Style
Guide](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).

In order to generate the documentation:

    # generate docs
    make build-docs

    # serve docs locally
    make serve-docs

When a branch is pushed to `master`, the documents are built and deployed to
the github pages site via [Github Actions](https://docs.github.com/en/actions).

# Technical Architecture

Here we provide an overview of the Luna technical architecture and
design methodologies.

# Resources

Here are some links to useful developer resources:

-   [Markdown Guide](https://www.markdownguide.org/) for
    [MkDocs](http://www.mkdocs.org/)
-   [pytest](https://docs.pytest.org/)
-   [docker](https://www.docker.com/)
-   [Github Actions](https://docs.github.com/en/actions)
