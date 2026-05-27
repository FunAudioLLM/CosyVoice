"""Frontend prompt-cache patch for CosyVoice3.

Reuses (speech_token, speech_feat, embedding, prompt_text_token) when the
same (prompt_text, prompt_wav) combination is requested again.

Usage:
    from fe_cache import enable_fe_cache
    enable_fe_cache(model)
"""
import threading

_cache_lock = threading.Lock()
_cache = {}


def _key(prompt_text, prompt_wav):
    return (prompt_text, prompt_wav)


def enable_fe_cache(model):
    fe = model.frontend
    orig = fe.frontend_zero_shot

    def cached_frontend_zero_shot(tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id):
        # When using a registered speaker id, original code already takes a fast path.
        if zero_shot_spk_id != '':
            return orig(tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id)

        k = _key(prompt_text, prompt_wav)
        with _cache_lock:
            cached = _cache.get(k)

        if cached is None:
            # Cold path: do full work, then cache the prompt-side outputs.
            model_input = orig(tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id)
            cached = {
                'prompt_text': model_input['prompt_text'],
                'prompt_text_len': model_input['prompt_text_len'],
                'llm_prompt_speech_token': model_input['llm_prompt_speech_token'],
                'llm_prompt_speech_token_len': model_input['llm_prompt_speech_token_len'],
                'flow_prompt_speech_token': model_input['flow_prompt_speech_token'],
                'flow_prompt_speech_token_len': model_input['flow_prompt_speech_token_len'],
                'prompt_speech_feat': model_input['prompt_speech_feat'],
                'prompt_speech_feat_len': model_input['prompt_speech_feat_len'],
                'llm_embedding': model_input['llm_embedding'],
                'flow_embedding': model_input['flow_embedding'],
            }
            with _cache_lock:
                _cache[k] = cached
            return model_input

        # Warm path: tokenize tts_text only, splice with cached prompt-side.
        tts_text_token, tts_text_token_len = fe._extract_text_token(tts_text)
        model_input = dict(cached)
        model_input['text'] = tts_text_token
        model_input['text_len'] = tts_text_token_len
        return model_input

    fe.frontend_zero_shot = cached_frontend_zero_shot
    return _cache  # expose for inspection
