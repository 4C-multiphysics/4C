# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# The script requires the following variables to be set, e.g., with
# export RELEASE="2025.2.0"
# export COMMIT_SHA="87550fd36f7d057a47b67dba2226ed6cb6a8985b"
# export VERSION_DEPENDENCIES="49ffc8b8" # The dependencies hash
# export IMAGE_PREBUILT_4C="4c@sha256:8c313acdd3c1f96a5cc943b4239e63dd8557417c1c3f5fe85144a63bc4ed67ab"
# export IMAGE_PREBUILT_4C_MINIMAL="4c-minimal@sha256:170306ee11ca9f403dd8248949b748aba22180e8532a035c02a43f923217c22b"
#
# For the prebuilt 4C images it is better to use the digest `4c@sha256:...` instead of the `main` tag.
# The main tag can already point to a newer commit than the desired commit.
#
# Note: you need a working docker installation
set -e

echo "This script pushes images to the public 4C repository for a release."
read -sp "Have you checked that the script is up-to-date? Do you know what you are doing? (y/N): " REPLY
echo ""

if [ "$REPLY" != "y" ]; then
echo "Abort."
  exit 1
fi

REQUIRED_INPUT=(
  "RELEASE"
  "COMMIT_SHA"
  "VERSION_DEPENDENCIES"
  "IMAGE_PREBUILT_4C"
  "IMAGE_PREBUILT_4C_MINIMAL"
)

echo "Input:"
for VAR in ${REQUIRED_INPUT[@]};
do
  echo "${VAR}=${!VAR}"
  if [ "${!VAR}" = "" ]; then
    echo "Required input is empty!"
    exit 1
  fi
done
echo ""


# Dependencies image
docker pull ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:${VERSION_DEPENDENCIES}
docker tag ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:${VERSION_DEPENDENCIES} ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:${RELEASE}
docker tag ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:${VERSION_DEPENDENCIES} ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:latest


# Prebuilt 4C
docker pull ghcr.io/4c-multiphysics/${IMAGE_PREBUILT_4C}
# Check if the 4C build in the image matches the commit for the release
IMAGE_PREBUILT_4C_REVISION=`docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision"}}' ghcr.io/4c-multiphysics/${IMAGE_PREBUILT_4C}`
if [ "$COMMIT_SHA" != "$IMAGE_PREBUILT_4C_REVISION" ]; then
  echo "The prebuilt 4C is based on commit $IMAGE_PREBUILT_4C_REVISION. But you provided commit $COMMIT_SHA for the release."
  echo "The provided docker image might be wrong!"
  exit 1
fi
# Check if the dependencies hash matches the provided dependencies version
IMAGE_PREBUILT_4C_DEPENDENCIES_HASH=`docker inspect --format '{{ index .Config.Labels "org.4c-multiphysics.dependencies_hash"}}' ghcr.io/4c-multiphysics/${IMAGE_PREBUILT_4C}`
if [ "$VERSION_DEPENDENCIES" != "$IMAGE_PREBUILT_4C_DEPENDENCIES_HASH" ]; then
  echo "The prebuilt 4C is based on the dependencies image $IMAGE_PREBUILT_4C_DEPENDENCIES_HASH. But you provided $VERSION_DEPENDENCIES for the dependencies image for the release."
  echo "The provided docker image might be wrong!"
  exit 1
fi
docker tag ghcr.io/4c-multiphysics/${IMAGE_PREBUILT_4C} ghcr.io/4c-multiphysics/4c:${RELEASE}
docker tag ghcr.io/4c-multiphysics/${IMAGE_PREBUILT_4C} ghcr.io/4c-multiphysics/4c:latest


# Prebuilt 4C minimal
docker pull ghcr.io/4c-multiphysics/${IMAGE_PREBUILT_4C_MINIMAL}
IMAGE_PREBUILT_4C_MINIMAL_REVISION=`docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision"}}' ghcr.io/4c-multiphysics/${IMAGE_PREBUILT_4C_MINIMAL}`
if [ "$COMMIT_SHA" != "$IMAGE_PREBUILT_4C_MINIMAL_REVISION" ]; then
  echo "The prebuilt 4C is based on commit $IMAGE_PREBUILT_4C_MINIMAL_REVISION. But you provided commit $COMMIT_SHA for the release."
  echo "The provided docker image might be wrong!"
  exit 1
fi
docker tag ghcr.io/4c-multiphysics/${IMAGE_PREBUILT_4C_MINIMAL} ghcr.io/4c-multiphysics/4c-minimal:${RELEASE}
docker tag ghcr.io/4c-multiphysics/${IMAGE_PREBUILT_4C_MINIMAL} ghcr.io/4c-multiphysics/4c-minimal:latest


read -sp "You are about to push to the 4C repository. Continue? (y/N): " REPLY
echo ""
if [ "$REPLY" != "y" ]; then
  echo "Abort."
  exit 1
fi

# Push the images
docker push ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:${RELEASE}
docker push ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:latest

docker push ghcr.io/4c-multiphysics/4c:${RELEASE}
docker push ghcr.io/4c-multiphysics/4c:latest

docker push ghcr.io/4c-multiphysics/4c-minimal:${RELEASE}
docker push ghcr.io/4c-multiphysics/4c-minimal:latest

echo "You have released the images for 4C version $VERSION"
