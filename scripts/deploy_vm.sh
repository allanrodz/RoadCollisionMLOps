#!/usr/bin/env bash
set -euo pipefail

: "${VM_HOST:?Set VM_HOST}"
: "${VM_USER:?Set VM_USER}"
: "${VM_SSH_KEY_PATH:?Set VM_SSH_KEY_PATH}"
: "${IMAGE_NAME:?Set IMAGE_NAME}"

REMOTE_DIR="/home/${VM_USER}/road-collision-api"

ssh -i "$VM_SSH_KEY_PATH" -o StrictHostKeyChecking=no "${VM_USER}@${VM_HOST}" "mkdir -p ${REMOTE_DIR}"
scp -i "$VM_SSH_KEY_PATH" -o StrictHostKeyChecking=no docker-compose.prod.yml prometheus.yml "${VM_USER}@${VM_HOST}:${REMOTE_DIR}/"

ssh -i "$VM_SSH_KEY_PATH" -o StrictHostKeyChecking=no "${VM_USER}@${VM_HOST}" <<EOF
set -euo pipefail
cd ${REMOTE_DIR}
echo "IMAGE_NAME=${IMAGE_NAME}" > .env
if [ -n "${GHCR_USERNAME:-}" ] && [ -n "${GHCR_TOKEN:-}" ]; then
  echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USERNAME}" --password-stdin
fi
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker ps
EOF
