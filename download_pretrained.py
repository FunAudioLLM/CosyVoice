from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    local_dir="pretrained_models/Fun-CosyVoice3-0.5B",
    token=False,
)