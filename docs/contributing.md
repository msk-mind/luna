# Contributing

Thank you for your interest in contributing to Luna!

This document is a quick guide to contributing.

## Creating a new issue
- **Enhancement**
  For new feature requests, create an issue with `enhancement` tag and describe the feature and its behavior in detail.
- **Bug**
  Describe the error, provide logs and code version. Apply `bug` tag.
- **Question**
  To ask a question or provide a suggestion, open an issues with `question` tag.

## Setting up local environment

Refer to the [development setup guide](dev.md#development-setup-instructions)
to:
- Get the latest code from git
- Setup environment with `make venv`
- See other available `make` targets with `make help`

## Code Development

We follow the [git branching workflow](https://git-scm.com/book/en/v2/Git-Branching-Branching-Workflows/).

Create your branch with the naming convention `IssueNumber-Description` (e.g. 123-annotation) from the default branch.

### Development guide
1. Use Google docstring format to clearly document all classes, methods,
   variables. In PyCharm, you can change the settting in `Settings > Tools >
   Python Integrated Tools > Docstring format`
2. Setup pre-commit Python linter for uniform code styles. In the cloned repo,
   run `pre-commit install`. When you attempt to make a commit, `black` will
   reformat your code and `flake8` will check for PEP8 compliance. Once these
   tests pass, you can re-add and commit properly formatted files.
3. Follow clean code principles
4. Add new dependencies to `pyproject.toml` using `poetry add`. If a dependency
   cannot be added using `poetry`, add it to the `environment.yml` so that it
   will be installed with `mamba` or `conda`. If that package is also
   managed by `poetry`, add it to the `pyproject.toml` using `poetry add
   --lock` with the same version added by `mamba` or `conda`.
5. Aim for 85% test coverage for new modules and ensure these tests pass.
6. Apply meaningful log statements.
7. Check that no sensitive information (credentials, hostnames, author info) is accidentally committed.
8. Add or update the tutorials in the documentation site.
  Refer to [mkdocs documentation generation guide](dev.md#documentation-generation)

### Unit tests
Luna uses [pytest](https://docs.pytest.org) for testing. Each code contribution
should have appropriate test coverage, as per the development guide. Tests
should follow the code package structure and any data required for testing
should be added under the same package under `testdata` directory.

To run all tests locally, run
```
make test
```

To run test coverage, run
```
make coverage
```

Individual tests can be run by specifying the test file and method.
```
pytest -s path/to/test_file.py::test_method
```

## Pull Requests (PR)

### Creating PR

Once you are done (see [development guide](dev.md)) with the changes in your branch, create a PR to merge your branch into the default **dev** branch.

1. Luna uses semantic release for version management. The PR title must start with one of these types specified in the [commit guidelines](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines).
Additionally, provide Luna subpackage and a brief description of the PR in the title.
For example, `perf: pathology – improve inference speed`
2. Add a description of the changes - list of changes, screenshots of functional tests, and any notes.
   This summary will help the reviewer engage in a more meaningful conversation with the author.
3. Link the PR to an issue with `Connect issue`.
4. Ensure all checks pass.
5. Add at least one person from the core development team as a reviewer.

### Reviewing PR

If you are assigned as a reviewer, add your comments based on these guidelines.

1. Check that the PR follows the [development guide](dev.md).
2. Is there any code/documentation update that is missed in the PR?

We encourage the author and reviewer to engage in active conversations to ensure all suggestions are clearly understood
and to improve the code quality of the overall library.

### Merging PR

PRs should be merged by the author who created the PR.
In order to merge a PR to **dev** branch, the PR should have
- All unit tests passed (via Github Actions)
- At least 1 approval from a reviewer
- No merge conflicts

Once all the requirements are met, double check that the PR title adheres to the commit guidelines.

**Squash and merge** your PR. This will add your branch as 1 commit in the dev branch.

Once the PR is merged, delete your branch.

## Additional Resources

### Release
#### Branches

- **dev**: default branch. All PRs are merged to dev branch
- **master**: release branch. Always stable
- other branches - created from dev branch. Contributors add and test their features, bug fixes etc.

**NOTE**: when merging dev to master branch for a release, use "Create a Merge commit" to show the PRs and commits.
Do not use squash and merge, as it will only make the release commit available for the semantic release parsing.

#### Release workflow

Release is automated via Github actions from the master branch. During the release:
-	Semantic release determines the new version and updates the version
-	Github tag/release is created with specification fo the new changes.
-	Pyluna is pushed to [pypi](https://pypi.org/project/pyluna/).

### Useful Links
- [Github pull requests docs](https://docs.github.com/en/pull-requests)
- [Python semantic release](https://python-semantic-release.readthedocs.io/en/latest/)
- [PyCharm python settings](https://www.jetbrains.com/help/pycharm/settings-tools-python-integrated-tools.html)
- [Clean code](https://learning.oreilly.com/library/view/clean-code/9780136083238/)
