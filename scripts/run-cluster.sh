#!/bin/bash
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_ip --head|--worker path_to_hf_home [additional_args...]"
    exit 1
fi

DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"
PATH_TO_HF_HOME="$4"
shift 4
ADDITIONAL_ARGS=("$@")

if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

if [[ "${PATH_TO_HF_HOME}" != /* ]]; then
    echo "Error: path_to_hf_home must be absolute. Got: ${PATH_TO_HF_HOME}"
    exit 1
fi

CONTAINER_NAME="node-${RANDOM}"
cleanup() { docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true; docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Parse VLLM_HOST_IP from additional args if supplied using -e VLLM_HOST_IP=...
VLLM_HOST_IP=""
for ((i=0;i<${#ADDITIONAL_ARGS[@]};i++)); do
    arg="${ADDITIONAL_ARGS[i]}"
    if [[ "${arg}" == "-e" && $((i+1)) -lt ${#ADDITIONAL_ARGS[@]} ]]; then
        next="${ADDITIONAL_ARGS[i+1]}"
        if [[ "${next}" == VLLM_HOST_IP=* ]]; then
            VLLM_HOST_IP="${next#VLLM_HOST_IP=}"
        fi
    elif [[ "${arg}" == VLLM_HOST_IP=* ]]; then
        VLLM_HOST_IP="${arg#VLLM_HOST_IP=}"
    fi
done

RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
    [ -n "${VLLM_HOST_IP}" ] && RAY_START_CMD+=" --node-ip-address=${VLLM_HOST_IP}"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
    [ -n "${VLLM_HOST_IP}" ] && RAY_START_CMD+=" --node-ip-address=${VLLM_HOST_IP}"
fi

DOCKER_ENV_ARGS=()
if [ -n "${VLLM_HOST_IP}" ]; then
    DOCKER_ENV_ARGS+=( -e "RAY_NODE_IP_ADDRESS=${VLLM_HOST_IP}" -e "RAY_OVERRIDE_NODE_IP_ADDRESS=${VLLM_HOST_IP}" )
fi

# Auto-detect interface (attempt). Validate it is an interface name, not an IP.
IFACE=""
if [ -n "${VLLM_HOST_IP}" ]; then
    IFACE=$(ip route get "${VLLM_HOST_IP}" 2>/dev/null | sed -n 's/.* dev \([[:alnum:]][_[:alnum:]\.:+-]*\).*/\1/p' || true)
    if [ -z "${IFACE}" ]; then
        IFACE=$(ip -o -4 addr show scope global | awk '{print $2; exit}' || true)
    fi
fi

# Validate IFACE: must NOT look like an IPv4 address and must be non-empty.
if [ -n "${IFACE}" ]; then
    if [[ "${IFACE}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Detected IFACE looks like an IP (${IFACE}). Ignoring IFACE to avoid passing an IP to GLOO/NCCL."
        IFACE=""
    fi
fi

if [ -n "${IFACE}" ]; then
    echo "Using network interface: ${IFACE}"
    DOCKER_ENV_ARGS+=( -e "NCCL_SOCKET_IFNAME=${IFACE}" -e "GLOO_SOCKET_IFNAME=${IFACE}" -e "GLOO_DEVICE_TRANSPORT=tcp" )
else
    echo "No valid network interface detected; not setting NCCL/GLOO interface envs (allow auto-detect)."
fi

DOCKER_ADD_HOST_ARGS=()
if [ -n "${VLLM_HOST_IP}" ]; then
    HOSTNAME_ON_HOST=$(hostname)
    DOCKER_ADD_HOST_ARGS+=( --add-host "${HOSTNAME_ON_HOST}:${VLLM_HOST_IP}" )
fi

echo "Starting container ${CONTAINER_NAME} from image ${DOCKER_IMAGE}"
docker run \
    --entrypoint /bin/bash \
    --network host \
    --name "${CONTAINER_NAME}" \
    --shm-size 10.24g \
    --gpus all \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "${DOCKER_ADD_HOST_ARGS[@]}" \
    "${DOCKER_ENV_ARGS[@]}" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "/bin/bash -lc \"${RAY_START_CMD}\""