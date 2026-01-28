#!/bin/bash
# vLLM Ray cluster launcher (see scripts/README.md for full workflow).

# Step 1: Validate required arguments.
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_ip --head|--worker path_to_hf_home [additional_args...]"
    exit 1
fi

# Step 2: Extract the mandatory positional arguments.
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"
PATH_TO_HF_HOME="$4"
shift 4

# Step 3: Preserve extra Docker arguments.
ADDITIONAL_ARGS=("$@")

# Step 4: Validate the node role flag.
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Step 5: Allocate a unique container name per launch.
CONTAINER_NAME="node-${RANDOM}"

# Step 6: Ensure the container is removed on exit.
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

# Step 7: Build the Ray start command for the chosen role.
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# Step 8: Parse optional env settings forwarded via -e.
VLLM_HOST_IP=""
AUTO_SERVE_SCRIPT=""
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

# Step 9: Align Ray IP bindings when VLLM_HOST_IP is supplied.
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

# Step 10: Program NCCL/GLOO defaults (overrides remain available via -e flags).
NCCL_IFACE="enp4s0"
GLOO_IFACE="enp4s0"

echo "Using NCCL interface: $NCCL_IFACE"
echo "Using GLOO interface: $GLOO_IFACE"

DIST_IFACE_ENV=(
    -e "NCCL_SOCKET_IFNAME=${NCCL_IFACE}"
    -e "NCCL_DEBUG=INFO"
    -e "NCCL_SOCKET_NTHREADS=4"
    -e "TORCH_DISTRIBUTED_DEFAULT_BACKEND=nccl"
    -e "GLOO_SOCKET_IFNAME=${GLOO_IFACE}"
)

# Step 11: Optionally chain a Serve script after the head starts.
if [ "${NODE_TYPE}" == "--head" ] && [ -n "${AUTO_SERVE_SCRIPT}" ]; then
    RAY_START_CMD="ray start --head --port=6379 && python3 ${AUTO_SERVE_SCRIPT}"
fi

# Step 12: Launch the container with the assembled parameters.
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