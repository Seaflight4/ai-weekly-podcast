"""
Text-to-Speech Module for converting text into speech using the TNG TTS provider.

Handles cleaning of input text, concurrent per-piece audio generation, and
merging of audio files into a single output.
"""

import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any

from pydub import AudioSegment

from .tts.factory import TTSProviderFactory
from .utils.config import load_config
from .utils.config_conversation import load_conversation_config

logger = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(
        self,
        model: str = "tng",
        api_key: Optional[str] = None,
        conversation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the TextToSpeech class.

        Args:
            model (str): The TTS provider to use. Defaults to 'tng'.
            api_key (Optional[str]): API key for the TTS service.
            conversation_config (Optional[Dict]): Configuration for conversation settings.
        """
        self.config = load_config()
        self.conversation_config = load_conversation_config(conversation_config)
        self.tts_config = self.conversation_config.get("text_to_speech", {})

        # Get API key from config if not provided
        if not api_key:
            api_key = getattr(self.config, f"{model.upper().replace('MULTI', '')}_API_KEY", None)

        # Initialize provider using factory
        self.provider = TTSProviderFactory.create(
            provider_name=model, api_key=api_key, model=model
        )

        # Inject provider-specific config fields that aren't passed per-call
        # (e.g. qwen3 voice design instructions, reference voice clips,
        # synthesis speed/temperature).
        provider_key = self.provider.__class__.__name__.lower().replace("tts", "")
        provider_cfg = self.tts_config.get(provider_key, {})
        if hasattr(self.provider, "instructions"):
            self.provider.instructions = provider_cfg.get("instructions", "")
        if hasattr(self.provider, "speed"):
            self.provider.speed = provider_cfg.get("speed", self.provider.speed)
        if hasattr(self.provider, "temperature"):
            self.provider.temperature = provider_cfg.get(
                "temperature", self.provider.temperature
            )
        # Load reference voice clips for cloning into the provider ONCE.
        # The provider reads the WAV files into memory and reuses the bytes
        # for every per-turn call (no per-call file I/O).
        reference_voices = provider_cfg.get("reference_voices")
        if reference_voices and hasattr(self.provider, "set_reference_voices"):
            self.provider.set_reference_voices(reference_voices)

        # Setup directories and config
        self._setup_directories()
        self.audio_format = self.tts_config.get("audio_format", "mp3")
        self.ending_message = self.tts_config.get("ending_message", "")
        # Concurrency for per-piece audio generation. 1 = sequential.
        self.max_workers = int(self.tts_config.get("max_workers", 1))

    def _get_provider_config(self) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        provider_name = self.provider.__class__.__name__.lower().replace("tts", "")
        provider_config = self.tts_config.get(provider_name, {})
        logger.debug(f"Using provider config: {provider_config}")
        return provider_config

    def convert_to_speech(self, text: str, output_file: str) -> None:
        """
        Convert input text to speech and save as an audio file.

        Args:
            text (str): Input text to convert to speech.
            output_file (str): Path to save the output audio file.

        Raises:
            ValueError: If the input text is not properly formatted
            RuntimeError: If audio generation fails
        """
        cleaned_text = text

        try:
            t0 = time.perf_counter()
            with tempfile.TemporaryDirectory(dir=self.temp_audio_dir) as temp_dir:
                audio_segments = self._generate_audio_segments(cleaned_text, temp_dir)
                t_gen = time.perf_counter() - t0

                t1 = time.perf_counter()
                self._merge_audio_files(audio_segments, output_file)
                t_merge = time.perf_counter() - t1

                logger.info(
                    "[tts] generated %d pieces in %.2fs (max_workers=%d); merge %.2fs; total %.2fs",
                    len(audio_segments),
                    t_gen,
                    self.max_workers,
                    t_merge,
                    t_gen + t_merge,
                )
                logger.info("Audio saved to %s", output_file)

        except Exception as e:
            logger.error("Error converting text to speech: %s", str(e))
            raise

    def _generate_audio_segments(self, text: str, temp_dir: str) -> List[str]:
        """Generate audio segments for each speaker turn.

        Collects all (piece, speaker_tag, model, temp_path) tasks first, then
        executes them with a ThreadPoolExecutor (or sequentially when
        max_workers <= 1). Order is restored at merge time via the
        filename sort key, so concurrent completion order does not matter.

        Voice is assigned by SPEAKER (Person1/Person2), not by Q/A role.
        Each speaker keeps one cloned voice for the entire episode.
        """
        qa_pairs = self.provider.split_qa(
            text, self.ending_message, self.provider.get_supported_tags()
        )
        provider_config = self._get_provider_config()
        model = provider_config.get("model")

        # Map Q/A role to speaker tag for voice-clone lookup. Person1 =
        # question (explainer), Person2 = answer (skeptic). The tag is passed
        # as `voice` to the provider, which resolves it to a reference clip.
        role_to_speaker = {"question": "person1", "answer": "person2"}

        # Phase 1: collect all tasks. Each task is (temp_path, piece, speaker_tag, model).
        tasks: List[Tuple[str, str, str, str]] = []
        for idx, (question, answer) in enumerate(qa_pairs, 1):
            if not question and not answer:
                continue
            sub_idx = 0
            for speaker_type, content in (("question", question), ("answer", answer)):
                if not content:
                    continue
                speaker_tag = role_to_speaker.get(speaker_type, speaker_type)
                for piece in self._split_long_turn(content):
                    if not re.search(r"[A-Za-z0-9]", piece):
                        logger.warning("Skipping non-verbal piece: %r", piece)
                        continue
                    sub_idx += 1
                    temp_file = os.path.join(
                        temp_dir, f"{idx}_{speaker_type}_{sub_idx}.{self.audio_format}"
                    )
                    tasks.append((temp_file, piece, speaker_tag, model))

        if not tasks:
            return []

        # Phase 2: execute (sequentially or in parallel).
        audio_files: List[str] = [None] * len(tasks)
        failed: List[int] = []

        def _run(i: int, temp_path: str, piece: str, voice: str, mdl: str) -> None:
            label = os.path.splitext(os.path.basename(temp_path))[0]
            t_start = time.perf_counter()
            try:
                audio_data = self.provider.generate_audio(piece, voice, mdl)
            except Exception as e:
                logger.warning(
                    "[tng] piece %s FAILED after %.0f ms: %s (skipping turn)",
                    label, (time.perf_counter() - t_start) * 1000, e,
                )
                failed.append(i)
                return
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            with open(temp_path, "wb") as f:
                f.write(audio_data)
            logger.info(
                "[tng] piece %s: %.0f ms, %d bytes",
                label,
                elapsed_ms,
                len(audio_data),
            )
            audio_files[i] = temp_path

        if self.max_workers <= 1:
            for i, (temp_path, piece, voice, mdl) in enumerate(tasks):
                _run(i, temp_path, piece, voice, mdl)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_to_idx = {
                    pool.submit(_run, i, tp, pc, v, m): i
                    for i, (tp, pc, v, m) in enumerate(tasks)
                }
                for fut in as_completed(future_to_idx):
                    fut.result()  # _run swallows exceptions; surface any other bug

        if failed:
            logger.warning(
                "[tts] %d/%d pieces failed and were skipped; episode will have "
                "gaps at those points", len(failed), len(tasks),
            )

        # Filter out any None (failed or produced no audio without raising).
        return [f for f in audio_files if f is not None]

    @staticmethod
    def _split_long_turn(content: str, max_chars: int = 3000) -> List[str]:
        """
        Split a single spoken turn into sub-pieces no longer than max_chars,
        breaking at sentence boundaries where possible.

        Args:
            content (str): The full turn text.
            max_chars (int): Max characters per sub-piece.

        Returns:
            List[str]: Sub-pieces of the turn.
        """
        if len(content) <= max_chars:
            return [content]

        pieces = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                pieces.append(current.strip())
                current = sentence
            else:
                current = (" " + sentence) if current else sentence
        if current:
            pieces.append(current.strip())
        return pieces

    def _merge_audio_files(self, audio_files: List[str], output_file: str) -> None:
        """
        Merge the provided audio files sequentially, ensuring questions come
        before answers within the same Q&A index.

        Args:
            audio_files: List of paths to audio files to merge.
            output_file: Path to save the merged audio file.

        Raises:
            RuntimeError: If no valid audio segments are produced.
        """
        try:

            def get_sort_key(file_path: str) -> Tuple[int, int, int]:
                """Sort key from filename: questions (0) before answers (1)."""
                basename = os.path.splitext(os.path.basename(file_path))[0]
                parts = basename.split("_")
                idx = int(parts[0])
                is_answer = 1 if parts[1].startswith("answer") else 0
                sub_idx = int(parts[2]) if len(parts) > 2 else 0
                return (idx, is_answer, sub_idx)

            combined = AudioSegment.empty()

            for file_path in sorted(audio_files, key=get_sort_key):
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    logger.warning("Skipping empty audio file: %s", file_path)
                    continue
                try:
                    segment = AudioSegment.from_file(file_path, format=self.audio_format)
                    combined += segment
                except Exception as e:
                    logger.warning("Skipping unreadable audio file %s: %s", file_path, str(e))

            if len(combined) == 0:
                raise RuntimeError("No valid audio segments produced")

            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            combined.export(output_file, format=self.audio_format)
            logger.info("Merged audio saved to %s", output_file)

        except Exception as e:
            logger.error("Error merging audio files: %s", str(e))
            raise

    def _setup_directories(self) -> None:
        """Setup required directories for audio processing.

        An absolute ``temp_audio_dir`` (from conversation_config) is used as-is
        so callers can place per-run temp pieces outside the package. A relative
        path is resolved under the vendored package directory.
        """
        self.output_directories = self.tts_config.get("output_directories", {})
        temp_dir = self.tts_config.get("temp_audio_dir", "data/audio/tmp/")

        if os.path.isabs(temp_dir):
            self.temp_audio_dir = temp_dir
        else:
            base_dir = os.path.abspath(os.path.dirname(__file__))
            self.temp_audio_dir = os.path.join(base_dir, temp_dir.rstrip("/"))

        os.makedirs(self.temp_audio_dir, exist_ok=True)

        for dir_path in [
            self.output_directories.get("transcripts"),
            self.output_directories.get("audio"),
        ]:
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
