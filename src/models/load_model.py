from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="RedHatAI/gemma-3-12b-it-quantized.w8a8",
    local_dir="./models/base/gemma-3-12b-it-quantized.w8a8",
    local_dir_use_symlinks=False,
    token="token"
)