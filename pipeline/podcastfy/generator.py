"""Long-form podcast generator (vendored from podcastfy-long).

Parses a ``podcast_brief.md`` (the format emitted by ``pipeline.generate``),
fetches full arXiv PDFs and Hacker-News/web pages, generates a two-host
transcript via an OpenAI-compatible LLM, and synthesizes voice-cloned audio
via the TNG qwen3 TTS service.

The entry point is :class:`SimplePodcastGenerator`.
"""

import json
import logging
import os
import re
from typing import List, Optional

from langchain_core.runnables import RunnableLambda
from openai import OpenAI

import pymupdf

from .content_generator import LLMBackend, LongFormContentStrategy
from .text_to_speech import TextToSpeech
from .sources import parse_brief, download_arxiv_pdfs, fetch_web_content

logger = logging.getLogger(__name__)

SKAINET_BASE = "https://chat.model.tngtech.com/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro-0813"


# Self-contained long-form podcast prompt (modelled on the souzatharsis
# longform prompt). No network / LangChain Hub dependency.
#
# The model is called with response_format=json_object, which suppresses the
# reasoning/thinking that DeepSeek otherwise leaks into its content. The model
# returns JSON {"dialogue": [{"speaker": "1"|"2", "text": "..."}, ...]} which we
# parse back into <PersonN>...</PersonN> tagged lines.
LONGFORM_PROMPT = """You are producing a part of a two-host podcast that gives a high-level overview of recent AI research and industry news.

Audience: AI researchers at a leading software consulting firm. They know ML fundamentals (RL, transformers, MoE, quantization, diffusion, autoregressive decoding, tokenization). Explain ONLY what is novel to each paper. Do not define basics they already know.

Podcast: {podcast_name} - {podcast_tagline}
Language: {output_language}
Roles: {roles_person1} (Person1) and {roles_person2} (Person2)

SPEAKER DYNAMIC:
- Two knowledgeable AI researchers co-hosting. They take turns leading topics and react to each other naturally. Genuine curiosity, surprise, or occasional skepticism when a claim truly warrants it.
- No one is the designated skeptic. The goal is a clear, big-picture explanation the listener can follow, not a debate. Pushback happens only when genuinely warranted by the content, never as a structural rule.
- Both hosts can teach as well as ask. Alternate who LEADS each topic so the conversation has natural variety.

SPOKEN STYLE:
- This is a spoken podcast, NOT a written article. Do NOT use colons (:), em-dashes, or semicolons (;). They sound like written text read aloud. Use periods to end sentences. Where you would write a colon or em-dash, start a new sentence instead. For example, instead of "the cost story: $15 versus $574", write "The cost story is striking. Fifteen dollars versus 574."
- Avoid written transition phrases like "Case in point:" or "Or put differently:". Just say it naturally.
- Keep each turn SHORT. Most turns 1 to 3 sentences, max ~30 words. A few turns per topic can be very short (1-5 words) if they are genuine reactions or questions. NEVER write a monologue turn.
- Cover ONLY what is in this part's INPUT. No "as we discussed earlier" padding.

CONVERSATIONAL INTERPLAY:
- This is a live dialogue, not alternating monologues. React to what the other host just said before making your point. Start with genuine reactions like "Right.", "Exactly.", "Yeah.", "Wait.", "Wow.", or "Oh." when the other host said something surprising or important. These are NOT filler. They show the listener someone is listening.
- Do NOT use empty filler as a standalone turn. "Absolutely." alone adds nothing. But "Right. And that connects to..." or "Wait, how does that actually work?" are genuine reactions that drive the conversation forward.
- The listening host should ask genuine questions when something is surprising, unclear, or needs deeper explanation. "How does that work?", "Wait, really?", "What does that mean in practice?" Questions create natural back-and-forth. Aim for 1-2 genuine questions per topic.
- Build on what the other host said. Connect your point to their point ("That is exactly it. And the benchmark numbers prove..."). Do not just state your next fact independently.

OUTPUT FORMAT:
- Return a single JSON object with this exact schema:
  {{"dialogue": [{{"speaker": "1", "text": "spoken words"}}, {{"speaker": "2", "text": "spoken words"}}]}}
- "speaker" must be "1" (Person1) or "2" (Person2).
- Output ONLY the JSON object. No thinking, no reasoning, no preamble, no
  markdown, no code fences, no closing commentary, no extra keys.

INSTRUCTIONS:
{instruction}

CONTEXT:
{context}

INPUT:
{input_text}
"""


def _extract_balanced_json(text: str):
    """
    Return the first balanced JSON object substring parsed to Python, or None.

    Tries every '{' position as a potential object start (so stray leading
    junk like a duplicated '{"' before the real object is simply skipped).
    """
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return None


def parse_dialogue_json(raw: str) -> str:
    """
    Convert a JSON dialogue response into <PersonN>...</PersonN> tagged lines.

    Accepts either {"dialogue": [...]} or a bare list of turn objects. Each turn
    is {"speaker": "1"|"2", "text": "..."}. Output lines follow the tag-per-line
    format used by the podcastfy TTS stack:
        <Person1>
        spoken text
        </Person1>
    """
    if not raw or not raw.strip():
        return ""
    text = raw.strip()
    # If the model wrapped the JSON in code fences, strip them.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()

    # Extract the outermost balanced JSON object if json.loads fails (handles
    # stray prefixes/suffixes without trusting rfind for nested braces).
    data = None
    try:
        data = json.loads(text)
    except Exception:
        data = _extract_balanced_json(text)
    if data is None:
        logger.warning("Could not find JSON in model response")
        return ""

    if isinstance(data, list):
        turns = data
    elif isinstance(data, dict):
        turns = data.get("dialogue") or data.get("turns") or []
        # Handle truncated JSON where the balanced extractor recovered the
        # inner turn object instead of the outer wrapper. A single turn
        # dict (has "speaker"/"text" but no "dialogue"/"turns" key) is treated
        # as a one-element list so the content is not lost.
        if not turns and "speaker" in data and ("text" in data or "t" in data):
            turns = [data]
    else:
        turns = []

    lines = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker", "")).strip()
        spoken = str(turn.get("text", "") or turn.get("t", "") or "").strip()
        if not spoken:
            continue
        if _is_filler_turn(spoken):
            continue
        if speaker == "2":
            person = 2
        elif speaker == "1":
            person = 1
        else:
            # Infer from existing tags if speaker missing/weird.
            person = 1
        lines.append(f"<Person{person}>")
        lines.append(spoken)
        lines.append(f"</Person{person}>")
    return "\n".join(lines)


# Short conversational fillers that add duration without content. Kept to a
# tight allow-list so real short answers (e.g. a quick clarifying reply) are
# not accidentally removed.
_FILLER_TURNS = {
    "absolutely", "absolutely!", "absolutely.",
    "exactly", "exactly!", "exactly.",
    "right", "right!", "right.",
    "great question", "great question!",
    "great point", "great point!",
    "great idea", "that's a great idea",
    "that's a mouthful", "that's a mouthful!",
    "okay", "okay.", "ok", "ok.",
    "yes", "yes!", "yeah", "yeah!",
    "sure", "sure.",
    "let's get into it", "lets get into it",
    "let's dive in", "lets dive in",
    "let's jump in", "lets jump in",
    "definitely", "definitely!",
    "now that's interesting",
    "wow", "wow!",
}


def _is_filler_turn(spoken: str) -> bool:
    """Return True if a spoken turn is mostly padding with no real content."""
    t = " ".join(spoken.split()).strip().rstrip(".").lower()
    if not t:
        return True
    # A very short turn made only of interjections/punctuation.
    if len(t) <= 2:
        return True
    return t in _FILLER_TURNS


class SimplePDFExtractor:
    """Lightweight PDF extractor for arXiv papers"""

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF with minimal processing.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content
        """
        try:
            doc = pymupdf.open(pdf_path)
            content = " ".join(page.get_text() for page in doc)
            doc.close()
            return content
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {str(e)}")
            raise


class SimplePodcastGenerator:
    def __init__(
        self,
        model_name: str = None,
        api_key_label: str = "SKAINET_API_KEY",
        api_base_label: str = "LLM_API_BASE",
        tts_model: str = "tng",
        papers_dir: Optional[str] = None,
        web_dir: Optional[str] = None,
    ):
        """
        Minimal setup: LLM + strategy + TTS.

        Defaults to the OpenAI-compatible SkaiNet endpoint (no Gemini key
        required) and the internal TNG TTS provider (reuses SKAINET_API_KEY
        as a bearer token). Set GEMINI_API_KEY and pass
        model_name="gemini-2.5-flash" to use Gemini for the LLM instead.

        Args:
            papers_dir: Directory to cache downloaded arXiv PDFs. Defaults to
                ``"papers"`` (relative to CWD). Pass an absolute path for a
                per-run cache so re-runs don't collide.
            web_dir: Directory to cache fetched web/blog page text. Defaults
                to ``"web"``.
        """
        if model_name is None:
            model_name = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

        self.papers_dir = papers_dir or "papers"
        self.web_dir = web_dir or "web"

        # Cap per-part output. DeepSeek-V4-Pro is a reasoning model, so the
        # thinking preamble counts toward the output token budget. 16000 gives
        # reasoning + JSON response enough room even with complex prompts
        # (conversational interplay, signposts, delta-first numbers, etc.).
        self.content_generator_config = {
            "max_output_tokens": 16000,
        }

        # Initialize LLM
        llm_backend = LLMBackend(
            is_local=False,
            temperature=0.7,
            max_output_tokens=self.content_generator_config["max_output_tokens"],
            model_name=model_name,
            api_key_label=api_key_label,
            api_base_label=api_base_label,
        )
        self.llm = llm_backend.llm
        self.tts_model = tts_model
        self.model_name = model_name

        # Raw OpenAI-compatible client for transcript generation. Using the SDK
        # directly (instead of langchain-openai's ChatOpenAI) avoids langchain's
        # structured-output parser, which raises LengthFinishReasonError when the
        # eventual JSON is truncated. Our own tolerant parser handles that case.
        self._use_raw_client = not (
            "gemini" in model_name.lower() or llm_backend.is_local
        )
        self._raw_client = None
        self._max_output_tokens = self.content_generator_config["max_output_tokens"]
        if self._use_raw_client:
            self._raw_client = OpenAI(
                api_key=os.environ.get(api_key_label),
                base_url=os.environ.get(api_base_label) or SKAINET_BASE,
                timeout=120,
                max_retries=2,
            )

        # Minimal config (replace with your values)
        self.config_conversation = {
            "podcast_name": "AI News Weekly",
            "podcast_tagline": "Latest AI research and news",
            "conversation_style": ["informative", "engaging", "punchy"],
            "roles_person1": "AI researcher",
            "roles_person2": "AI researcher",
            "dialogue_structure": ["conversation", "exchange"],
            "output_language": "English",
            "engagement_techniques": ["analogies", "examples", "specific numbers"],
            # Overview episode: ~10 concise parts (~2000 words total)
            "max_num_chunks": 10,
            "min_chunk_size": 4000,
            "per_paper_chars": 10000,      # abstract + intro per paper
            "per_paper_tail_chars": 3000,  # conclusion tail per paper
            "per_web_chars": 22000,        # full text fetched for blog sources
        }

        # Strategy (only long-form for you)
        self.strategy = LongFormContentStrategy(
            self.llm,
            self.content_generator_config,
            self.config_conversation,
        )

    def _call_llm_json(self, params: dict) -> str:
        """
        Call the LLM with the long-form prompt params and return tagged
        <PersonN> output (parsed from JSON).
        """
        if not self._use_raw_client:
            raise NotImplementedError("Gemini/local backends use the langchain chain; set use_raw_client=True")
        # Validate the TEMPLATE (not the formatted output) so that literal
        # braces in user content (e.g. code snippets in fetched web pages or
        # PDFs like {August}) don't trip a false "missing parameter" error.
        self._check_required_params(LONGFORM_PROMPT, params)
        prompt = LONGFORM_PROMPT.format(**params)
        resp = self._raw_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You return only valid JSON objects. Never include any "
                        "text outside the JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=self._max_output_tokens,
            response_format={"type": "json_object"},
            presence_penalty=0.75,
            frequency_penalty=0.75,
        )
        content = resp.choices[0].message.content or ""
        tagged = parse_dialogue_json(content)
        if len(tagged.strip()) < 40:
            finish_reason = resp.choices[0].finish_reason
            logger.warning(
                "LLM returned short/empty dialogue. finish_reason=%s, "
                "content_len=%d, content_preview=%.500s",
                finish_reason, len(content), content[:500]
            )
        # Return empty (not raise) so the strategy's retry loop catches it.
        return tagged

    def _check_required_params(self, prompt: str, params: dict) -> None:
        """Ensure the prompt template has no missing placeholders."""
        missing = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", prompt)
        for name in set(missing):
            if name not in params:
                raise KeyError(f"Missing prompt parameter: {name}")

    def generate_transcript(self, markdown_content: str) -> str:
        """
        Generate podcast transcript from markdown.

        Args:
            markdown_content: Full markdown file content

        Returns:
            Transcript with <Person1> and <Person2> tags
        """
        # The strategy expects a chain-like object it can .invoke(params) with.
        # We provide a RunnableLambda backed by the raw OpenAI client so we keep
        # full control over JSON parsing (our parser is tolerant of truncation
        # and DeepSeek's doubled-brace quirk, unlike langchain-openai's parser).
        chain = RunnableLambda(self._call_llm_json)

        # Prepare parameters
        prompt_params = {
            "roles_person1": self.config_conversation["roles_person1"],
            "roles_person2": self.config_conversation["roles_person2"],
            "podcast_name": self.config_conversation["podcast_name"],
            "podcast_tagline": self.config_conversation["podcast_tagline"],
            "output_language": self.config_conversation["output_language"],
        }

        # Generate using long-form strategy
        transcript = self.strategy.generate(chain, markdown_content, prompt_params)

        # Clean output
        transcript = self.strategy.clean(transcript, self.content_generator_config)

        return transcript

    def generate_audio(self, transcript: str, output_path: str,
                        tts_model: str = None,
                        temp_audio_dir: Optional[str] = None):
        """Convert transcript to audio.

        Args:
            transcript: Tagged transcript text.
            output_path: Where to write the merged MP3.
            tts_model: TTS provider name (defaults to the instance default).
            temp_audio_dir: Optional absolute directory for per-turn audio
                pieces. When None, the package-default temp dir is used.
        """
        conversation_config = None
        if temp_audio_dir is not None:
            conversation_config = {
                "text_to_speech": {"temp_audio_dir": temp_audio_dir},
            }
        tts = TextToSpeech(
            model=tts_model or self.tts_model,
            conversation_config=conversation_config,
        )
        tts.convert_to_speech(transcript, output_path)
        print(f"[podcastfy] audio saved to {output_path}")

    def load_markdown_and_pdfs(self, markdown_path: str, pdf_paths: List[str]) -> str:
        """
        Load markdown file and extract text from PDF papers.
        Concatenate all into single input for transcript generation.

        Args:
            markdown_path: Path to podcast_brief.md
            pdf_paths: List of paths to arXiv PDF files

        Returns:
            Combined text content
        """
        # Load markdown
        with open(markdown_path, "r") as f:
            combined_content = f.read()

        # Extract and append PDF content
        pdf_extractor = SimplePDFExtractor()
        per_paper_chars = self.config_conversation.get("per_paper_chars", 20000)
        tail_chars = self.config_conversation.get("per_paper_tail_chars", 5000)
        for pdf_path in pdf_paths:
            print(f"[podcastfy] extracting {os.path.basename(pdf_path)}...")
            try:
                pdf_text = pdf_extractor.extract_text(pdf_path)
                # Collapse whitespace so the dense PDF text is readable.
                pdf_text = " ".join(pdf_text.split())
                # Overview input: abstract+intro (head) and conclusion (tail).
                body = pdf_text[:per_paper_chars]
                if len(pdf_text) > per_paper_chars:
                    body += "\n\n[EXCERPT END]\n\n" + pdf_text[-tail_chars:]
                combined_content += (
                    f"\n\n=== PAPER: {os.path.basename(pdf_path)} ===\n\n{body}"
                )
            except Exception as e:
                logger.warning(f"Skipping {pdf_path}: {e}")

        return combined_content

    # Thematic ordering of topics for narrative flow. Maps source titles
    # (substring match) to their position in the podcast. Grouped into 3 themes:
    #   Model architectures -> Agent frameworks -> Industry & security
    _THEMATIC_ORDER = [
        # Model architectures
        "DiffusionGemma Technical Report",
        "Qwen 3.8 27B",
        "Ornith-1.5: From Self-Scaffolding to Self-Improvement",
        # Agent frameworks
        "StateM: Reaching 95.3% Raw Accuracy",
        "UI-Mate: Advancing Open-Weight Foundation GUI Agents",
        "Twin: Playing an Unknown Game with a Test-Time Digital Twin",
        # Industry & security / research
        "Cursor is now a part of SpaceX",
        "Self-Supervised Visual On-Policy Distillation",
        "What Aggregate Scores Miss",
        "Every Model Cheats",
    ]

    def load_brief_and_sources(self, brief_path: str = "source/podcast_brief.md") -> str:
        """
        Parse a podcast brief, auto-download referenced arXiv PDFs, and
        assemble topic-delimited blocks for transcript generation.

        Produces a sequence of === TOPIC: <title> === blocks (one per source,
        in thematic order), bracketed by === INTRO === and === RECAP === blocks.
        The topic-aware chunker in content_generator.py splits on these markers
        so each LLM call handles one coherent topic.

        Args:
            brief_path: Path to the podcast brief markdown file.

        Returns:
            Combined text content for the LLM.
        """
        # Read the brief for the intro block (title + description).
        with open(brief_path, "r") as f:
            brief_text = f.read()

        # Parse sources and download arXiv PDFs.
        sources = parse_brief(brief_path)
        arxiv_sources = [s for s in sources if s.kind == "arxiv"]
        print(f"[podcastfy] {len(sources)} sources ({len(arxiv_sources)} arXiv, {len(sources) - len(arxiv_sources)} blog)")

        pdf_paths = download_arxiv_pdfs(sources, target_dir=self.papers_dir)
        print(f"[podcastfy] {len(pdf_paths)} arXiv PDF(s) available")

        # Map PDF paths by arxiv_id for quick lookup.
        pdf_by_id = {os.path.splitext(os.path.basename(p))[0]: p for p in pdf_paths}

        # Fetch full main-article text for blog (non-arXiv) sources so the
        # LLM gets the full narrative (e.g. the Dreadnode case studies).
        web_by_url = fetch_web_content(sources, target_dir=self.web_dir)
        print(f"[podcastfy] {len(web_by_url)} blog source(s) fetched")

        # Order sources thematically.
        ordered = self._order_sources_thematically(sources)

        # Build the combined content as topic-delimited blocks.
        # INTRO block: brief title + description (for the opening overview).
        intro_lines = brief_text.strip().split("\n")[:3]
        combined_content = "=== INTRO ===\n"
        combined_content += "\n".join(intro_lines)
        combined_content += "\n=== END INTRO ===\n"

        # Topic blocks.
        per_paper_chars = self.config_conversation.get("per_paper_chars", 20000)
        tail_chars = self.config_conversation.get("per_paper_tail_chars", 5000)
        per_web_chars = self.config_conversation.get("per_web_chars", 12000)
        pdf_extractor = SimplePDFExtractor()
        topic_titles = []
        for src in ordered:
            topic_titles.append(src.title)
            combined_content += f"\n=== TOPIC: {src.title} ===\n"
            combined_content += f"SOURCE: {src.kind} · URL: {src.url}\n"
            combined_content += f"EXCERPT: {src.excerpt}\n"

            # Append full PDF text if this is an arXiv source with a downloaded PDF.
            if src.kind == "arxiv" and src.arxiv_id and src.arxiv_id in pdf_by_id:
                pdf_path = pdf_by_id[src.arxiv_id]
                print(f"[podcastfy] extracting {os.path.basename(pdf_path)}...")
                try:
                    pdf_text = pdf_extractor.extract_text(pdf_path)
                    pdf_text = " ".join(pdf_text.split())
                    body = pdf_text[:per_paper_chars]
                    if len(pdf_text) > per_paper_chars:
                        body += "\n\n[EXCERPT END]\n\n" + pdf_text[-tail_chars:]
                    combined_content += f"\nFULL PAPER TEXT:\n{body}\n"
                except Exception as e:
                    logger.warning(f"Skipping {pdf_path}: {e}")
            elif src.kind == "blog" and src.url in web_by_url:
                # Blog source: append the fetched full main-article text.
                web_text = web_by_url[src.url]
                body = web_text[:per_web_chars]
                if len(web_text) > per_web_chars:
                    body += "\n\n[EXCERPT END]\n\n" + web_text[-2000:]
                combined_content += f"\nFULL PAGE TEXT:\n{body}\n"

            combined_content += "=== END TOPIC ===\n"

        # RECAP block: list of all topic titles (for the closing summary).
        combined_content += "\n=== RECAP ===\n"
        combined_content += "Topics covered in this episode:\n"
        for i, title in enumerate(topic_titles, 1):
            combined_content += f"{i}. {title}\n"
        combined_content += "=== END RECAP ===\n"

        return combined_content

    def _order_sources_thematically(self, sources) -> list:
        """Order sources by the predefined thematic ordering.

        Sources are matched to _THEMATIC_ORDER by substring match on title.
        Unmatched sources are appended at the end.
        """
        ordered = []
        remaining = list(sources)
        for target in self._THEMATIC_ORDER:
            for i, src in enumerate(remaining):
                if target.lower() in src.title.lower():
                    ordered.append(src)
                    remaining.pop(i)
                    break
        # Append any unmatched sources at the end.
        ordered.extend(remaining)
        return ordered
