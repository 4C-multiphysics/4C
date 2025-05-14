---
name: Release
about: Create a new release.
title: "Release 4C version YYYY.MINOR.PATCH"
type: Task
assignees: ''

---

## Steps for a new 4C release
> [!NOTE]
> For a PATCH-release, only some of the steps below are necessary. Commits are typically added to the
> ``main`` branch and cherry-picked to the release branch. That ensures that the main branch is fixed
> and the PATCH itself is still compatible with all projects consuming a specific version of |FOURC|.

- [ ] Check whether all changes that should be part of the release are merged into the ``main``-branch.
- [ ] Draft the release notes and post them as a comment to this issue (See notes below on release notes)
- [ ] Update the version number in the ``VERSION`` file in the repo. The version scheme is ``YYYY.MINOR.PATCH``, where ``YYYY`` is the year of the release, ``MINOR`` is the number of the release in the year starting at ``1``, and ``PATCH`` is the patch number. ``PATCH`` is typically ``0`` and incremented on demand. Typically, this step only involves removing the ``-dev`` suffix from the version.
- [ ] Pick a commit from ``main``-branch where all nightly tests have passed and the above points are satisfied. Commit: ``<insert the commit sha>``
- [ ] Create a new branch from the commit chosen in the previous step. The branch should be named ``release/YYYY.MINOR.x``. Note, since all patches are maintained on this branch, the name does not contain the respective patch number.
- [ ] Create a new release on GitHub. The tag name should be ``vYYYY.MINOR.PATCH`` and the release title should be ``4C version vYYYY.MINOR.PATCH``. Tick 'Set as latest release' and 'Create a discussion for this release'
- [ ] Upload the `4C_metadata.yaml` and `4C_schema.yaml` to the release. You can download them as artifact from the nightly pipeline.
- [ ] Update the tags of all Docker images, e.g., ``4c`` and ``4c-dependencies-ubuntu24.04``. Add the tag ``YYYY.MINOR.PATCH`` to the images, and ``latest`` if it is the latest release. You can run the script `./utilities/tag_images_for_release.sh`.
- [ ] Prepare the next release in the `VERSION` file in the repo by incrementing the number of the release, resetting the patch number to `0`, and adding the suffix `-dev`.


### Release notes
The release notes might contain a summary of the changes since the last release or a description of the fixed bugs in a PATCH release.

We do not yet have an automated way to collect the changelog. So, this requires some manual work. You can get a draft of the release notes from GitHub (Releases -> Draft a new release -> Generate release notes).

<details><summary>Changelog template</summary>
<p>

## What's Changed
<!-- Usually the commit message with pull request id, as provided by GitHub's generated release notes, is enough -->

### Breaking Changes
<!-- List of breaking changes: Filter issues with `breaking change` label -->

### Major changes
<!-- Any major changes. This can also contain self-written text on some major change -->

### Dependency changes
<!-- Any changes to our dependencies, e.g., Trilinos update, a new dependency -->

### Miscellaneous
<!-- Any other noteworthy changes -->

## New Contributors
<!-- GitHub's generated release notes contain a list of new contributors since the last release -->


<!-- link to full changelog provided by GitHub's generated release notes -->
**Full Changelog**: https://github.com/4C-multiphysics/4C/compare/v2025.1.0...v2025.2.0
</p>
</details>
