# Contributing to 4C

Thank you for your willingness to contribute to 4C. We welcome all types of contributions, irrespective of size and complexity.

### Types of contributions
- [Issues](#Issues)
- [Pull requests](#Pull-requests)
- [Documentation](#Documentation)

**All contributions to 4C must adhere to [The European Code of Conduct for Research Integrity – Revised Edition 2023](http://www.doi.org/10.26356/ECOC).**


## Issues
Issues are generally used to remind or inform yourself or others about certain things in the
software. We use them to report bugs, start a feature request, or plan tasks. Please refer to [GitHub Discussions](https://github.com/4C-multiphysics/4C/discussions) to ask questions.

To create an issue, select one of our templates and provide a detailed description.

Before opening a new issue, please check whether your bug has
already been reported within the existing issues. Opening an issue is a valid contribution on its own and does not mean you
have to solve it yourself.


## Pull requests

### 1. Setup of 4C
Set up 4C as described in the [README.md](README.md)

### 2. Prepare your development environment

To ensure a high quality of the repository, we enforce some pre-commit hooks in the repository. You can activate them by executing the following command in your source directory:
```
./utilities/set_up_dev_env.sh
```

### 3. Code development

Add your changes to the codebase. Please add unit tests or input-file tests that capture your changes. For further information, see our documentation on [testing](https://4c-multiphysics.github.io/4C/documentation/4Ctesting.html).

#### Coding style
Your changes are checked for a number of coding style conventions when creating a commit. These conventions are also verified when you submit a pull request. Please annotate your C++ source code with [Doxygen](https://doxygen.nl/index.html) comments.

#### Commit messages
Please provide meaningful commit messages.

### 5. Submit a pull request
Once you submitted your pull request, checks will run automatically to verify that you changes do not break existing functionality. We will review your changes before it can be merged into the `main` branch.
We desire a clean commit history. This may require rebasing the commits before merging.

## Documentation
In 4C, we have two types of documentation. The general [4C documentation](https://4c-multiphysics.github.io/4C/documentation/) generated with [Sphinx](https://www.sphinx-doc.org/en/master/#) and the [4C source code documentation](https://4c-multiphysics.github.io/4C/doxygen/) generated with [Doxygen](https://doxygen.nl/index.html). We welcome any improvements to our documentation. More information can be found [here](doc/README.md).
