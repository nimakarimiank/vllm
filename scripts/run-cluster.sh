#!/bin/bash
#
# Launch a Ray cluster inside Docker for vLLM inference.
#
# This script can start either a head node or a worker node, depending on the
# --head or --worker flag provided as the third positional argument.
#
# Usage:
# 1. Designate one machine as the head node and execute:
#    bash run_cluster.sh \
#         vllm/vllm-openai \
#         <head_node_ip> \
#         --head \
#         /abs/path/to/huggingface/cache \
#         -e VLLM_HOST_IP=<head_node_ip>
#
# 2. On every worker machine, execute:
#    bash run_cluster.sh \
#         vllm/vllm-openai \
#         <head_node_ip> \
#         --worker \
#         /abs/path/to/huggingface/cache \
#         -e VLLM_HOST_IP=<worker_node_ip>
# 
# Each worker requires a unique VLLM_HOST_IP value.
# Keep each terminal session open. Closing a session stops the associated Ray
# node and thereby shuts down the entire cluster.
# Every machine must be reachable at the supplied IP address.
#
# The container is named "node-<random_suffix>". To open a shell inside
# a container after launch, use:
#       docker exec -it node-<random_suffix> /bin/bash
#
# Then, you can execute vLLM commands on the Ray cluster as if it were a
# single machine, e.g. vllm serve ...
#
# To stop the container, use:
#       docker stop node-<random_suffix>

# Check for minimum number of required arguments.
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_ip --head|--worker path_to_hf_home [additional_args...]"
    exit 1
fi

# Extract the mandatory positional arguments and remove them from $@.
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker.
PATH_TO_HF_HOME="$4"
shift 4

# Preserve any extra arguments so they can be forwarded to Docker.
ADDITIONAL_ARGS=("$@")

# Validate the NODE_TYPE argument.
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Generate a unique container name with random suffix.
# Docker container names must be unique on each host.
# The random suffix allows multiple Ray containers to run simultaneously on the same machine,
# for example, on a multi-GPU machine.
CONTAINER_NAME="node-${RANDOM}"

# Define a cleanup routine that removes the container when the script exits.
# This prevents orphaned containers from accumulating if the script is interrupted.
cleanup() {
    docker stop "${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}"
}
trap cleanup EXIT

# Build the Ray start command based on the node role.
# The head node manages the cluster and accepts connections on port 6379, 
# while workers connect to the head's address.
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# Parse VLLM_HOST_IP from additional args if present.
# This is needed for multi-NIC configurations where Ray needs explicit IP bindings.
VLLM_HOST_IP=""
AUTO_SERVE_SCRIPT=""
for arg in "${ADDITIONAL_ARGS[@]}"; do
    if [[ $arg == "-e" ]]; then
        continue
    fi
    if [[ $arg == VLLM_HOST_IP=* ]]; then
        VLLM_HOST_IP="${arg#VLLM_HOST_IP=}"
        break
    fi
    if [[ $arg == AUTO_SERVE_SCRIPT=* ]]; then
        AUTO_SERVE_SCRIPT="${arg#AUTO_SERVE_SCRIPT=}"
    fi
done

# Build Ray IP environment variables if VLLM_HOST_IP is set.
# These variables ensure Ray binds to the correct network interface on multi-NIC systems.
RAY_IP_VARS=()
if [ -n "${VLLM_HOST_IP}" ]; then
    RAY_IP_VARS=(
        -e "RAY_NODE_IP_ADDRESS=${VLLM_HOST_IP}"
        -e "RAY_OVERRIDE_NODE_IP_ADDRESS=${VLLM_HOST_IP}"
    )
fi
#############################################
# Detect correct network interface for NCCL/GLOO
#############################################
NCCL_IFACE="enp4s0"
GLOO_IFACE="enp4s0"

# # If user already set NCCL/GLOO interface, respect it
# for arg in "${ADDITIONAL_ARGS[@]}"; do
#     if [[ $arg == "-e" ]]; then
#         continue
#     fi
#     if [[ $arg == NCCL_SOCKET_IFNAME=* ]]; then
#         NCCL_IFACE="${arg#NCCL_SOCKET_IFNAME=}"
#     fi
#     if [[ $arg == GLOO_SOCKET_IFNAME=* ]]; then
#         GLOO_IFACE="${arg#GLOO_SOCKET_IFNAME=}"
#     fi
# done

# # Auto-detect only if not manually set
# if [ -z "$NCCL_IFACE" ]; then
#     NCCL_IFACE=$(ip -o -4 route get "$VLLM_HOST_IP" | sed -n 's/.* dev \([^ ]*\).*/\1/p')
# fi
# if [ -z "$GLOO_IFACE" ]; then
#     GLOO_IFACE="$NCCL_IFACE"
# fi

echo "Using NCCL interface: $NCCL_IFACE"
echo "Using GLOO interface: $GLOO_IFACE"

DIST_IFACE_ENV=(
    -e "NCCL_SOCKET_IFNAME=${NCCL_IFACE}"
    -e "NCCL_DEBUG=INFO"
    -e "NCCL_SOCKET_NTHREADS=4"
    -e "TORCH_DISTRIBUTED_DEFAULT_BACKEND=nccl"
    -e "GLOO_SOCKET_IFNAME=${GLOO_IFACE}"
)

# If this is the head node and an AUTO_SERVE_SCRIPT environment variable was provided
# (e.g. via: -e AUTO_SERVE_SCRIPT=/app/ray_serve.py) then automatically launch the
# Ray Serve application after the Ray head process starts. NOTE: This will start
# the serve script immediately; for multi-GPU tensor parallel loading you should
# ensure worker nodes are already up, otherwise the model may only use the head GPU.
if [ "${NODE_TYPE}" == "--head" ] && [ -n "${AUTO_SERVE_SCRIPT}" ]; then
    RAY_START_CMD="ray start --head --port=6379 && python3 ${AUTO_SERVE_SCRIPT}"
fi

# Launch the container with the assembled parameters.
# --network host: Allows Ray nodes to communicate directly via host networking
# --shm-size 10.24g: Increases shared memory
# --gpus all: Gives container access to all GPUs on the host
# -v HF_HOME: Mounts HuggingFace cache to avoid re-downloading models
docker run \
    --runtime nvidia \
    --entrypoint /bin/bash \
    --network host \
    --name "${CONTAINER_NAME}" \
    --shm-size 10.24g \
    --ipc=host \
    --gpus all \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "${RAY_IP_VARS[@]}" \
    "${DIST_IFACE_ENV[@]}" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"
