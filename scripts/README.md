# scripts/ README

Helper utilities for launching and operating **vLLM** Ray clusters live here. The primary entry point is `run-cluster.sh`, which creates Ray head/worker nodes inside NVIDIA-enabled Docker containers and wires up networking + distributed training environment variables.

### Navigation
- [Step 0 – Requirements](#step-0--requirements)
- [Step 1 – Prepare the Hugging Face cache path](#step-1--prepare-the-hugging-face-cache-path)
- [Step 2 – Launch the Ray head node](#step-2--launch-the-ray-head-node)
- [Step 3 – Launch each Ray worker node](#step-3--launch-each-ray-worker-node)
- [Step 4 – Manage and inspect containers](#step-4--manage-and-inspect-containers)
- [Step 5 – Enter the head container and run vLLM Serve](#step-5--enter-the-head-container-and-run-vllm-serve)
- [Step 6 – Confirm the running models](#step-6--confirm-the-running-models)
- [Step 7 – Send a chat completion request](#step-7--send-a-chat-completion-request-outside-docker)
- [Reference – Script arguments and environment variables](#reference--script-arguments-and-environment-variables)

---

## Step 0 – Requirements
- Docker with the NVIDIA Container Toolkit installed on every machine.
- Access to the target container image (default: `vllm/vllm-openai`).
- All machines can reach each other on the Ray port `6379` and whatever ports you expose for inference (default `8000`).
- A shared or replicated Hugging Face cache path that each node can mount (e.g., an NFS mount or a local directory kept in sync).

---

## Step 1 – Prepare the Hugging Face cache path
1. Pick an absolute directory on every machine (for example `/data/hf-cache`).
2. Ensure it is writable by the user running Docker.
3. Populate it with any pre-downloaded models you need, or let vLLM download on demand (the directory is mounted into the container as `/root/.cache/huggingface`).

---

## Step 2 – Launch the Ray head node
```bash
sudo bash run-cluster.sh \
        vllm/vllm-openai \
        192.168.0.101 \
        --head \
        /data/hf-cache \
        -e VLLM_HOST_IP=192.168.0.101
```
**What happens:**
- A container named `node-<random_suffix>` starts with `ray start --head --port=6379 --block`.
- `VLLM_HOST_IP` automatically programs Ray’s IP binding variables so multi-NIC machines pick the right interface.

Optional extras:
- Pass `-e AUTO_SERVE_SCRIPT=/app/ray_serve.py` to auto-run a Serve script once Ray is up.
- Override network fabrics via `-e NCCL_SOCKET_IFNAME=eth0` or `-e GLOO_SOCKET_IFNAME=ib0` if the defaults (`enp4s0`) are wrong.

---

## Step 3 – Launch each Ray worker node
Run the same script on every worker machine, pointing to the head IP and giving each worker a unique host IP:
```bash
sudo bash run-cluster.sh \
        vllm/vllm-openai \
        192.168.0.101 \
        --worker \
        /data/hf-cache \
        -e VLLM_HOST_IP=192.168.0.102
```
Repeat with `VLLM_HOST_IP=...103`, `...104`, etc. Workers connect to `ray start --address=192.168.0.101:6379` inside the container.

---

## Step 4 – Manage and inspect containers
- Containers are auto-named `node-<random_suffix>` (separate per host so multiple clusters can coexist).
- The script installs a trap; stopping the script (Ctrl+C) calls `docker stop` followed by `docker rm` to avoid orphaned containers.
- To inspect logs or run commands:
    ```bash
    docker exec -it node-<suffix> /bin/bash
    ```

---

## Step 5 – Enter the head container and run vLLM Serve
### 5.1 Attach to the head container
```bash
docker exec -it node-<head_suffix> /bin/bash
```
Wait until all worker nodes have printed messages indicating they joined the Ray cluster before moving on.

### 5.2 Choose a model to serve (Hugging Face or local)
You can host any Hugging Face model or a local folder that follows the same structure. Inside the container:

#### Option A – Serve directly from Hugging Face Hub
1. (Optional) Authenticate if you need private models:
     ```bash
     huggingface-cli login --token <HF_TOKEN>
     ```
     or pass `-e HUGGING_FACE_HUB_TOKEN=<HF_TOKEN>` when launching the container.
2. Run vLLM Serve with the repo ID:
     ```bash
     vllm serve Qwen/Qwen3-0.6B \
             --tensor-parallel-size 2 \
             --download-dir /root/.cache/huggingface
     ```
     - Replace `Qwen/Qwen3-0.6B` with any other Hugging Face repo (`meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, etc.).
     - Use `--hf-token <token>` if you prefer passing credentials per invocation.

#### Option B – Serve a pre-downloaded local model directory
```bash
vllm serve /root/.cache/huggingface/mistral-7b-instruct \
        --served-model-name mistral-7b \
        --max-model-len 4096
```
Point the argument to any folder you staged in Step 1. `--served-model-name` lets you expose a friendly alias while keeping the on-disk folder name unchanged.

#### Useful knobs (either option)
- `--tensor-parallel-size N`: number of GPUs to split the model across; match the number of Ray actors/GPUs participating.
- `--pipeline-parallel-size`, `--max-model-len`, `--dtype float16`, `--quantization awq/gptq`, etc., depending on the model’s requirements.

When startup finishes you should see: `Uvicorn running on http://0.0.0.0:8000`. The Serve HTTP server now listens on the host’s port 8000 through host networking.

---

## Step 6 – Confirm the running models
From **outside** the container (or another shell on the host), list the published models:
```bash
curl http://<head_host>:8000/v1/models
```
Example response:
```json
{
    "object": "list",
    "data": [
        {
            "id": "Qwen/Qwen3-0.6B",
            "object": "model",
            "max_model_len": 40960,
            "owned_by": "vllm"
        }
    ]
}
```
Use the `id` field from this list when sending inference requests.

---

## Step 7 – Send a chat completion request (outside Docker)
```bash
curl http://<head_host>:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
                    "model": "Qwen/Qwen3-0.6B",
                    "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Who won the world series in 2020?"}
                    ]
                }'
```
Swap in any other model ID returned by `/v1/models`, and run the curl command from your laptop or a non-container shell on the head machine. Successful responses follow the OpenAI-compatible schema.

---

## Reference – Script arguments and environment variables
- `--head | --worker`: controls whether `ray start` runs in head or worker mode.
- `-e VLLM_HOST_IP=<ip>`: required so Ray binds to the correct NIC; every worker must have a unique value.
- `-e AUTO_SERVE_SCRIPT=/path/to/script.py`: auto-launch a Ray Serve script once the head node starts.
- NCCL/GLOO overrides: `-e NCCL_SOCKET_IFNAME=<iface>`, `-e GLOO_SOCKET_IFNAME=<iface>`.
- Additional `docker run` flags (volumes, env vars, GPU partitioning) can be appended to the command; the script forwards any remaining arguments untouched.
