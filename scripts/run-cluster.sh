#!/bin/bash
#
# Launch a Ray cluster inside Docker for vLLM inference.
#
# Usage:
#  bash run_cluster.sh <docker_image> <head_node_ip> --head|--worker <abs_path_to_hf_home> [additional_args...]
# Example:
#  sudo bash run_cluster.sh vllm/vllm-openai:nightly-x86_64 192.168.0.101 --head /home/nima_ka/.cache/huggingface/hub -e VLLM_HOST_IP=192.168.0.101
#
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_ip --head|--worker path_to_hf_home [additional_args...]"
    exit 1
fi

DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # --head or --worker
PATH_TO_HF_HOME="$4"
shift 4

ADDITIONAL_ARGS=("$@")

if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Require absolute path for HF cache (tilde won't expand under sudo reliably).
if [[ "${PATH_TO_HF_HOME}" != /* ]]; then
    echo "Error: path_to_hf_home must be absolute. Got: ${PATH_TO_HF_HOME}"
    exit 1
fi

CONTAINER_NAME="node-${RANDOM}"

cleanup() {
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Parse VLLM_HOST_IP from additional args if given as: -e VLLM_HOST_IP=1.2.3.4
VLLM_HOST_IP=""
for ((i=0;i<${#ADDITIONAL_ARGS[@]};i++)); do
    arg="${ADDITIONAL_ARGS[i]}"
    if [[ "${arg}" == "VLLM_HOST_IP="* ]] || [[ "${arg}" == "VLLM_HOST_IP"* ]]; then
        VLLM_HOST_IP="${arg#VLLM_HOST_IP=}"
        # If the user passed it with a preceding -e, we may have -e then VLLM_HOST_IP=...; handle that:
    fi
    if [[ "${arg}" == "-e" && $((i+1)) -lt ${#ADDITIONAL_ARGS[@]} ]]; then
        next="${ADDITIONAL_ARGS[i+1]}"
        if [[ "${next}" == VLLM_HOST_IP=* ]]; then
            VLLM_HOST_IP="${next#VLLM_HOST_IP=}"
        fi
    fi
done

# Fall back to an env var if user exported it to shell
if [ -z "${VLLM_HOST_IP}" ] && [ -n "${VLLM_HOST_IP:-}" ]; then
    VLLM_HOST_IP="${VLLM_HOST_IP}"
fi

# Ray start command
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
    if [ -n "${VLLM_HOST_IP}" ]; then
        RAY_START_CMD+=" --node-ip-address=${VLLM_HOST_IP}"
    fi
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
    if [ -n "${VLLM_HOST_IP}" ]; then
        RAY_START_CMD+=" --node-ip-address=${VLLM_HOST_IP}"
    fi
fi

# Build docker env args
DOCKER_ENV_ARGS=()
if [ -n "${VLLM_HOST_IP}" ]; then
    DOCKER_ENV_ARGS+=( -e "RAY_NODE_IP_ADDRESS=${VLLM_HOST_IP}" -e "RAY_OVERRIDE_NODE_IP_ADDRESS=${VLLM_HOST_IP}" )
fi

# Auto-detect network interface to set NCCL/GLOO envs (optional but helpful)
IFACE=""
if [ -n "${VLLM_HOST_IP}" ]; then
    # Attempt to detect the interface used to reach VLLM_HOST_IP; fallback to first non-loopback
    IFACE=$(ip route get "${VLLM_HOST_IP}" 2>/dev/null | awk -F 'dev ' '{print $2}' | awk '{print $1}' || true)
    if [ -z "${IFACE}" ]; then
        IFACE=$(ip -o -4 addr show scope global | awk '{print $2; exit}')
    fi
fi

if [ -n "${IFACE}" ]; then
    DOCKER_ENV_ARGS+=( -e "NCCL_SOCKET_IFNAME=${IFACE}" -e "GLOO_SOCKET_IFNAME=${IFACE}" -e "GLOO_DEVICE_TRANSPORT=tcp" )
fi

# If VLLM_HOST_IP is provided, add a container-only hosts mapping so hostname resolves to LAN IP inside container
DOCKER_ADD_HOST_ARGS=()
if [ -n "${VLLM_HOST_IP}" ]; then
    HOSTNAME_ON_HOST=$(hostname)
    DOCKER_ADD_HOST_ARGS+=( --add-host "${HOSTNAME_ON_HOST}:${VLLM_HOST_IP}" )
fi

# Compose and run docker
echo "Starting container ${CONTAINER_NAME} from image ${DOCKER_IMAGE}"
echo "Mapped HF cache: ${PATH_TO_HF_HOME} -> /root/.cache/huggingface"

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