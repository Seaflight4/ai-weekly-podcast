"""TNG (SkaiNet) TTS provider implementation.

Calls the ``/synthesize`` endpoint exposed by the internal TNG TTS service
(https://tts.model.tngtech.com/api). Authentication uses a bearer token
(the shared SKAINET_API_KEY).

Voice cloning is supported via the ``voice_to_clone`` field: a reference
audio file (WAV) uploaded as multipart/form-data alongside the text. The
provider reads the reference clips into memory ONCE at construction time
and reuses the same bytes for every per-turn call, so a 93-turn episode
incurs 93 server-side reference uploads but zero per-call file I/O.

The endpoint returns a JSON body of the form ``{"audio": "<base64 WAV>"}``;
we decode the base64 WAV and re-encode to MP3 so the downstream merge
step keeps working.
"""

import base64
import io
import json
import logging
import os
from typing import Dict, List, Optional

import requests
from pydub import AudioSegment

from ..base import TTSProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://tts.model.tngtech.com/api"
DEFAULT_TNG_MODEL = "qwen3"
# Vendored reference clips shipped under pipeline/podcastfy/reference/.
DEFAULT_REFERENCE_VOICES = {
    "person1": "reference/host_m.wav",
    "person2": "reference/host_f.wav",
}

# Directory holding this file's package (pipeline/podcastfy/), used to
# resolve package-relative reference voice paths.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_reference_path(path: str) -> str:
    """Resolve a reference-voice path.

    Absolute paths are used as-is. Relative paths are resolved against the
    vendored package directory (pipeline/podcastfy/), falling back to the
    current working directory, so the shipped reference clips are found
    regardless of where the pipeline is invoked from.
    """
    if os.path.isabs(path):
        return path
    pkg_path = os.path.join(_PACKAGE_DIR, path)
    if os.path.exists(pkg_path):
        return pkg_path
    return os.path.join(os.getcwd(), path)


class TNGTTS(TTSProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.base_url = DEFAULT_BASE_URL
        # `model` here is the provider name passed by the factory (e.g. "tng").
        # It is only used by the orchestrator to detect multi-speaker mode
        # ("multi" in self.model.lower()); the actual TNG model name (e.g.
        # "qwen3") is supplied per-call via generate_audio(model=...).
        self.model = model or "tng"
        # qwen3 voice design instructions (e.g. pace, mood). Populated from
        # the `tng.instructions` config block by TextToSpeech.__init__.
        self.instructions = ""
        # Synthesis parameters, overridable from config.yaml. Defaults keep
        # the prior hardcoded values so behaviour is unchanged until tuned.
        self.speed = "1.0"        # qwen3 accepts a float (1.0 = normal)
        self.temperature = "0.5"  # lower = more deterministic
        # Reference voice clips for cloning, keyed by speaker tag
        # ("person1" / "person2"). Each value is the raw WAV bytes read ONCE
        # at init and reused for every call. Populated by
        # `set_reference_voices()` from the orchestrator; defaults are
        # attempted in `load_reference_voices()`.
        self.reference_voices: Dict[str, bytes] = {}
        self._reference_filenames: Dict[str, str] = {}

    def load_reference_voices(
        self, reference_voices: Optional[Dict[str, str]] = None
    ) -> None:
        """Read the reference voice clips into memory once.

        Args:
            reference_voices: Mapping of speaker tag ("person1"/"person2")
                to a WAV file path. If None, uses DEFAULT_REFERENCE_VOICES
                resolved relative to the vendored package directory.
        """
        paths = reference_voices or DEFAULT_REFERENCE_VOICES
        for speaker, path in paths.items():
            if not path:
                continue
            resolved = _resolve_reference_path(path)
            if not os.path.exists(resolved):
                logger.warning(
                    "Reference voice for %s not found at %s; "
                    "voice cloning for this speaker will be disabled.",
                    speaker,
                    resolved,
                )
                continue
            with open(resolved, "rb") as f:
                self.reference_voices[speaker] = f.read()
            self._reference_filenames[speaker] = os.path.basename(resolved)
            logger.info(
                "Loaded reference voice for %s from %s (%d bytes)",
                speaker,
                resolved,
                len(self.reference_voices[speaker]),
            )

    def set_reference_voices(self, reference_voices: Dict[str, str]) -> None:
        """Public hook for the orchestrator to pass config-driven paths."""
        self.load_reference_voices(reference_voices)

    def generate_audio(
        self,
        text: str,
        voice: str,
        model: str,
        voice2: str = None,
    ) -> bytes:
        """Synthesize one turn of speech.

        Args:
            text: The text to synthesize.
            voice: Speaker tag ("person1" / "person2") used to look up the
                reference voice clip for cloning. Falls back to a built-in
                voice name if no reference clip is registered for the tag.
            model: The TNG model name (e.g. "qwen3").
            voice2: Unused (kept for base-class compatibility).
        """
        if not self.api_key:
            raise RuntimeError(
                "TNG TTS requires an API key. Set SKAINET_API_KEY in your .env "
                "(it is read as TNG_API_KEY by Config)."
            )

        tng_model = model or DEFAULT_TNG_MODEL

        # Build the form fields. `voice` in the API is a built-in voice name
        # for qwen3; when cloning we send `voice_to_clone` (the reference WAV)
        # and omit `voice` so the server uses the cloned timbre.
        ref_bytes = self.reference_voices.get(voice) if voice else None
        ref_name = self._reference_filenames.get(voice) if voice else None

        # Validate: if no reference clip, require a built-in voice name.
        if not ref_bytes:
            if not voice:
                raise ValueError(
                    "Either a reference voice clip or a built-in voice name "
                    "must be provided."
                )
            self.validate_parameters(text, voice, tng_model)
        # When cloning we don't need a built-in voice name; skip the strict
        # voice-non-empty validation in that case by passing a placeholder.
        else:
            self.validate_parameters(text, voice or "clone", tng_model)

        data = {
            "input": text,
            "model": tng_model,
            "temperature": str(self.temperature),
            "speed": str(self.speed),
            "instructions": self.instructions or "",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/synthesize"

        files = None
        if ref_bytes:
            files = {
                "voice_to_clone": (
                    ref_name or "voice.wav",
                    ref_bytes,
                    "audio/wav",
                )
            }
        else:
            # No reference clip: use the built-in voice name field.
            data["voice"] = voice

        try:
            response = self._post_with_retry(url, data, files, headers)
        except requests.RequestException as e:
            raise RuntimeError(f"TNG TTS request failed: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"TNG TTS returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        # The endpoint returns JSON: {"audio": "<base64-encoded WAV>"}.
        try:
            resp_data = response.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"TNG TTS returned non-JSON body: {e}") from e

        if isinstance(resp_data, dict) and "detail" in resp_data:
            raise RuntimeError(f"TNG TTS error: {resp_data['detail']}")

        audio_b64 = resp_data.get("audio") if isinstance(resp_data, dict) else None
        if not audio_b64:
            raise RuntimeError(
                f"TNG TTS response missing 'audio' field: {str(resp_data)[:200]}"
            )

        try:
            wav_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            raise RuntimeError(f"TNG TTS audio is not valid base64: {e}") from e

        # Normalise WAV -> MP3 so the orchestrator's merge step works.
        try:
            segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        except Exception as e:
            raise RuntimeError(f"TNG TTS audio could not be decoded as WAV: {e}") from e

        buf = io.BytesIO()
        segment.export(buf, format="mp3", codec="libmp3lame")
        return buf.getvalue()

    # HTTP statuses that are likely transient and worth retrying with backoff.
    _RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def _post_with_retry(self, url, data, files, headers, max_attempts: int = 4):
        """POST to the TTS endpoint, retrying transient failures with backoff.

        Retries on connection errors and on the retryable HTTP statuses in
        ``_RETRYABLE_STATUS`` (e.g. 502/503/429). Non-retryable error responses
        raise immediately. Backoff is 2s, 4s, 8s between attempts.
        """
        import time as _time
        last_exc = None
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    url, data=data, files=files, headers=headers, timeout=120
                )
            except requests.RequestException as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "TNG TTS connection error (attempt %d/%d): %s; retrying in %ds",
                        attempt + 1, max_attempts, e, wait,
                    )
                    _time.sleep(wait)
                    continue
                raise
            if response.status_code in self._RETRYABLE_STATUS:
                if attempt < max_attempts - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "TNG TTS HTTP %d (attempt %d/%d); retrying in %ds: %.200s",
                        response.status_code, attempt + 1, max_attempts, wait,
                        response.text,
                    )
                    _time.sleep(wait)
                    continue
            return response
        # Should not reach here, but propagate the last exception if we do.
        if last_exc:
            raise last_exc
        return response

    def health_check(self, timeout: int = 15) -> bool:
        """Probe the ``/health`` endpoint. Returns True on HTTP 200."""
        url = f"{self.base_url}/health"
        try:
            r = requests.get(url, timeout=timeout)
            return r.status_code == 200
        except requests.RequestException as e:
            logger.warning("TNG TTS health check failed: %s", e)
            return False

    @classmethod
    def wait_until_healthy(
        cls, api_key: str, max_wait: float = 300.0, poll_interval: float = 5.0
    ) -> bool:
        """Block until the TTS service reports healthy, or raise after ``max_wait``.

        Polls ``GET /api/health`` every ``poll_interval`` seconds. The first poll
        is immediate (no initial delay). Returns True once healthy. Raises
        ``RuntimeError`` if the service does not recover within ``max_wait``,
        so the caller fails fast instead of burning the per-piece retry budget
        on a down service.
        """
        import time as _time
        # Reuse a lightweight instance just for the base_url + requests; no
        # need to load reference voices for a health probe.
        probe = cls(api_key=api_key)
        deadline = _time.monotonic() + max_wait
        attempt = 0
        while _time.monotonic() < deadline:
            attempt += 1
            if probe.health_check():
                if attempt > 1:
                    logger.info(
                        "TNG TTS healthy after %d probe(s) (~%.0fs)",
                        attempt, (attempt - 1) * poll_interval,
                    )
                else:
                    logger.info("TNG TTS healthy")
                return True
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                break
            sleep = min(poll_interval, remaining)
            logger.warning(
                "TNG TTS not healthy (probe %d); waiting %.0fs (%.0fs remaining)",
                attempt, sleep, remaining,
            )
            _time.sleep(sleep)
        raise RuntimeError(
            f"TNG TTS service did not become healthy within {max_wait:.0f}s "
            f"({attempt} probes to GET {probe.base_url}/health)"
        )

    def get_supported_tags(self) -> List[str]:
        return self.COMMON_SSML_TAGS
