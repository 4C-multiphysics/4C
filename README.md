<p align="center">
  <picture>
    <source
      srcset="https://raw.githubusercontent.com/4C-multiphysics/4C-design/refs/heads/main/4C-logo/negative-white/4C-logo-landscape_negative.svg"
      media="(prefers-color-scheme: dark)">
    <img
      src="https://raw.githubusercontent.com/4C-multiphysics/4C-design/refs/heads/main/4C-logo/standard-color/4C-logo-landscape_rgb.svg"
      width="350"
      title="4C"
      alt="4C logo">
  </picture>
</p>

<div align="center">

[![website](./utilities/assets/badges/website_badge.svg)](https://4C-multiphysics.org)
[![docs/documentation](./utilities/assets/badges/documentation_documentation.svg)](https://4c-multiphysics.github.io/4C/documentation/)
[![docs/doxygen](./utilities/assets/badges/documentation_doxygen.svg)](https://4c-multiphysics.github.io/4C/doxygen/)
[![coverage report](https://4c-multiphysics.github.io/4C/coverage_report/badge_coverage.svg)](https://4c-multiphysics.github.io/4C/coverage_report)

</div>

<div align="center">

[![workflows/checkcode](https://github.com/4C-multiphysics/4C/actions/workflows/checkcode.yml/badge.svg?branch=main)](https://github.com/4C-multiphysics/4C/actions/workflows/checkcode.yml?query=branch%3Amain)
[![workflows/buildtest](https://github.com/4C-multiphysics/4C/actions/workflows/buildtest.yml/badge.svg?branch=main)](https://github.com/4C-multiphysics/4C/actions/workflows/buildtest.yml?query=branch%3Amain)
[![workflows/nightly_tests](https://github.com/4C-multiphysics/4C/actions/workflows/nightly_tests.yml/badge.svg?branch=main)](https://github.com/4C-multiphysics/4C/actions/workflows/nightly_tests.yml?query=branch%3Amain)
[![workflows/documentation](https://github.com/4C-multiphysics/4C/actions/workflows/documentation.yml/badge.svg?branch=main)](https://github.com/4C-multiphysics/4C/actions/workflows/documentation.yml?query=branch%3Amain)
[![workflows/coverage](https://github.com/4C-multiphysics/4C/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/4C-multiphysics/4C/actions/workflows/coverage.yml?query=branch%3Amain)

</div>

## What is 4C Multiphysics?
4C ("Comprehensive Computational Community Code") is a parallel multiphysics research code
to address a plethora of physical problems by means of _computational mechanics_.

Large parts of 4C are based on finite element methods (FEM),
but alternative discretization methods, such as discontinuous Galerkin methods (DG),
particle methods and mesh-free methods have also been successfully integrated.
The research software is implemented throughout in object-oriented programming (C++)
using modern software design and is parallelized with MPI for distributed memory hardware architectures.

- **Website**: https://4C-multiphysics.org
- **Documentation**: https://4c-multiphysics.github.io/4C/documentation/

## Vision

We aim to advance the frontiers of computational science and engineering by providing a versatile, extensible and open-source research software framework for the systematic development, analysis, and application of advanced numerical methods for modeling and simulation of complex multiphysics phenomena across scales and disciplines.

## Mission

4C Multiphysics is a modular, parallel and open-source simulation environment tailored to the needs of researchers and computational scientists to enable and accelerate research in computational science and engineering. Our mission is to:

- support the formulation and enable the rigorous study of complex single- and multiphysics models across spatial and temporal scales through a variety of physical models, numerical methods, and coupling algorithms with a strong focus on finite element methods and particle methods accompanied by comprehensive documentation, tutorials and a welcoming culture to ensure a low entry barrier;
- curate and advance a modular and extensible framework to develop mathematical models for challenging real-world problems in science, engineering and biomedicine described by differential equations and to devise and implement novel numerical methods with a clear focus on methodological innovation and practical usability;
- offer a platform for both simulation practitioners, studying real-world problems through numerical simulation, as well as researchers in numerical modeling and computational methods, aiming at the development of accurate models and innovative numerical methods and their efficient software implementation in the support of complex real-world scenarios;
- enable parallel and scalable computations on workstations and clusters to increase efficiency and utilization of available hardware resources with a strong focus on medium- and large-scale practical applications;
- foster a growing international research community in which engineers, scientists, and domain experts can cooperate, contribute, accelerate scientific discovery, and share advances in computational modeling and numerical method development and are committed to open scientific exchange, collaborative development, and sustainable software practices.

## Getting started

To quickly run 4C, you can use our docker image, where 4C comes pre-compiled.

```
docker run --interactive --tty ghcr.io/4c-multiphysics/4c:main
/home/user/4C/build/4C ../tests/input_files/<some-input-file>.4C.yaml output_name
```

See our [documentation](https://4c-multiphysics.github.io/4C/documentation/installation/installation.html) for more information on
how to build 4C from source on your machine.
Consult our [Tutorials](https://4c-multiphysics.github.io/4C/documentation/tutorials/tutorials.html) to get an overview of the
general workflow in 4C.
Also have a look at related [projects/tools](https://4c-multiphysics.github.io/4C/documentation/tools/tools.html) that work with 4C, e.g., tools that help with input generatation or visualization.

## Contributing

If you're interested in contributing to 4C, we welcome your collaboration.
Please follow [our contributing guidelines](https://github.com/4C-multiphysics/4C/blob/main/CONTRIBUTING.md)
and [Code of Conduct](https://github.com/4C-multiphysics/4C/blob/main/CODE_OF_CONDUCT.md).

If you need help with 4C, feel free to ask questions
in the [GitHub discussions](https://github.com/4C-multiphysics/4C/discussions).

## How to cite 4C

Please cite 4C as follows:

```
4C: A Comprehensive Multiphysics Simulation Framework, https://www.4c-multiphysics.org
```

You could use the following BibTeX entry:

```bibtex
@misc{4C,
  author       = {{4C}},
  title        = {{4C}: A {C}omprehensive {M}ultiphysics {S}imulation {F}ramework},
  howpublished = {\url{https://www.4c-multiphysics.org}},
  year         = {YEAR},
  note         = {Accessed: DATE}
}
```

We kindly ask you to also give credit to the individual methods and algorithms used in 4C.
References to the relevant publications can be found on the [4C website](https://4c-multiphysics.org) or throughout the
source code.
If you need any assistance with finding suitable references,
please feel free to reach out in
the [4C Slack workspace](https://join.slack.com/t/4c-multiphysics/shared_invite/zt-1oi61jgdd-5tZuHku3Tb_BH5UBgojbpQ).

## Disclaimer

4C is developed for research purposes in the field of numerical method development.
It is not intended for any use beyond this purpose and generally should not be used for any form of
safety-relevant or safety-critical calculations
or for an application in association with physical products in particular.
