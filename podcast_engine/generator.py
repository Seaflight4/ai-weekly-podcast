"""Long-form podcast generator (vendored from podcastfy-long).

Fetches full arXiv PDFs and web pages, generates a two-host transcript via
an OpenAI-compatible LLM, and synthesizes voice-cloned audio via the TNG
qwen3 TTS service.

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
from .sources import download_arxiv_pdfs, fetch_web_content

logger = logging.getLogger(__name__)

SKAINET_BASE = "https://chat.model.tngtech.com/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

# Hard cap on how much of a fetched blog/web page is fed to the transcript
# LLM per part, regardless of depth. A deep-dive blog can exceed 35 KB, and
# re-processing that much input per part is the dominant transcript cost
# (measured ~17 min for a 7-blog episode). The head + a short tail is enough
# for a ~330-word topic discussion and cuts per-part input ~3x.
MAX_WEB_CHARS = 12000
MAX_WEB_TAIL_CHARS = 1500


# Self-contained long-form podcast prompt (modelled on the souzatharsis
# longform prompt). No network / LangChain Hub dependency.
#
# The model is called with response_format=json_object, which suppresses the
# reasoning/thinking that DeepSeek otherwise leaks into its content. The model
# returns JSON {"dialogue": [{"speaker": "1"|"2", "text": "..."}, ...]} which we
# parse back into <PersonN>...</PersonN> tagged lines.
LONGFORM_PROMPT = """You are producing a part of a two-host podcast about {podcast_topic}.

Audience: {audience}

Familiar topics the listener already knows and needs NO explanation of: {familiar_topics}

Podcast: {podcast_name} - {podcast_tagline}
Language: {output_language}
Roles: {host1_name} — {roles_person1} (Person1) and {host2_name} — {roles_person2} (Person2). Hosts introduce themselves by name and address each other by name; never refer to yourselves as "Person 1" or "Person 2".

SPEAKER DYNAMIC:
- Two knowledgeable co-hosts exploring the topic together. They take turns leading topics and react to each other naturally. Genuine curiosity, surprise, or occasional skepticism when a claim truly warrants it.
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
    Return the best balanced JSON object substring parsed to Python, or None.

    Scans every '{' position as a potential object start. When the model
    emits chain-of-thought reasoning before the JSON answer (common with
    reasoning models when response_format is not set), the reasoning may
    contain example JSON fragments. To avoid grabbing a reasoning artifact
    instead of the real answer, we prefer candidates that parse to a dict
    with a "dialogue" or "turns" key (the expected schema). If none qualify,
    fall back to the last successfully-parsed balanced object.
    """
    best_schema_match = None
    last_parsed = None
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
                        parsed = json.loads(candidate)
                    except Exception:
                        break
                    last_parsed = parsed
                    if (isinstance(parsed, dict)
                            and ("dialogue" in parsed or "turns" in parsed)):
                        best_schema_match = parsed
                    break
    return best_schema_match if best_schema_match is not None else last_parsed


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
        depth_factor: float = 1.0,
        max_num_chunks: int = 10,
        per_source_words: int | None = None,
        intro_words: int | None = None,
        recap_words: int | None = None,
        memory_words: int | None = None,
        audience_prompt: str | None = None,
        familiar_clause: str = "none",
        memory_context: Optional[dict] = None,
        items: Optional[list] = None,
        podcast_topic: str = "the provided sources",
        podcast_name: str = "Podcast",
        podcast_tagline: str = "",
        host1_name: str = "Brian",
        host2_name: str = "Tina",
        roles_person1: str = "co-host",
        roles_person2: str = "co-host",
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
            # Transcript generation is pinned to DeepSeek-V4-Flash. A heavier
            # reasoning model (e.g. Pro) spends far more of max_output_tokens on
            # a thinking preamble whose truncation kills the JSON and forces the
            # empty/1-turn retry path. Always use Flash unless model_name is
            # passed explicitly (programmatic override); LLM_MODEL must not
            # silently redirect it.
            model_name = DEFAULT_MODEL

        self.papers_dir = papers_dir or "papers"
        self.web_dir = web_dir or "web"

        # Cross-episode memory: {topic: ["episode_date: summary", ...]} from
        # prior runs (see pipeline.memory). When a topic in this episode closes
        # a theme present in a prior episode, a === MEMORY === block is injected
        # so the transcript model can reference the real continuation.
        self.memory_context: dict = memory_context or {}
        # The run's chosen RankedItems, so topic-aware ordering can group
        # same-theme sources adjacently (within-episode cross-referencing).
        self.items: list = items or []

        # Cap per-part output. DeepSeek-V4-Flash is a lighter-reasoning model,
        # so the thinking preamble is shorter than Pro's. 16000 still gives
        # reasoning + JSON response ample room; Flash will simply finish
        # faster when less reasoning is needed.
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
                # Big-context part calls (deep-dive blog sources) can run >120s;
                # a too-tight timeout kills them mid-generation and forces a
                # full retry (each wasted attempt ~= the timeout). 300s leaves
                # headroom for the largest parts.
                timeout=300,
                max_retries=2,
            )

        # Minimal config (replace with your values); the run config overrides
        # per-source input depth and the per-source transcript budget.
        self.depth_factor = depth_factor
        self.audience_prompt = audience_prompt
        self.familiar_clause = familiar_clause
        self.podcast_topic = podcast_topic
        # Output word budget (derived by RunConfig.budget). These cap each
        # part's transcript length so the finished episode conforms to the
        # requested length; they do NOT scale with depth_factor.
        self.per_source_words = per_source_words or 240
        self.intro_words = intro_words or 200
        self.recap_words = recap_words or 280
        self.memory_words = memory_words or 90
        self.config_conversation = {
            "podcast_name": podcast_name,
            "podcast_tagline": podcast_tagline,
            "conversation_style": ["informative", "engaging", "punchy"],
            "host1_name": host1_name,
            "host2_name": host2_name,
            "roles_person1": roles_person1,
            "roles_person2": roles_person2,
            "dialogue_structure": ["conversation", "exchange"],
            "output_language": "English",
            "engagement_techniques": ["analogies", "examples", "specific numbers"],
            "max_num_chunks": max(2, min(30, int(max_num_chunks))),
            "min_chunk_size": 4000,
            # Word caps for intro / mid / recap parts.
            "intro_words": self.intro_words,
            "per_source_words": self.per_source_words,
            "recap_words": self.recap_words,
            # Word cap for the cross-episode MEMORY part (prior-coverage sync).
            "memory_words": self.memory_words,
            # Per-source input context scales with depth: brief feeds less so
            # the LLM stays concise, deep-dive feeds more so it can go deeper.
            # Web (blog) text is capped hard: a blog page can be 35KB+ under
            # deep-dive, and re-reading that per part is the dominant transcript
            # cost; the head + a short tail is enough for a 330-word discussion.
            "per_paper_chars": round(10000 * depth_factor),      # abstract + intro per paper
            "per_paper_tail_chars": round(3000 * depth_factor),  # conclusion tail per paper
            "per_web_chars": min(round(22000 * depth_factor), MAX_WEB_CHARS),  # blog full text
            "depth_factor": depth_factor,
            # The mid/recap part instructions gain the grounded continuity
            # reference rules whenever cross-episode memory is available
            # (see pipeline.memory) or topic-aware ordering ran.
            "continuity_reference_enabled": bool(self.memory_context),
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
        # Generous token ceiling (per-part word budgets are enforced by the
        # prompt + deterministic trim in content_generator; a tight cap would
        # truncate DeepSeek's pre-JSON reasoning and cancel the part).
        max_tokens = int(params.get("max_output_tokens") or self._max_output_tokens)
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
            max_tokens=max_tokens,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            # DeepSeek honours response_format=json_object, which keeps its
            # reasoning out of `content`. Without it, complex parts emit a long
            # thinking preamble that eats the whole token budget and truncation
            # yields finish_reason=length, no JSON, and a wasted retry.
            response_format={"type": "json_object"},
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

    def generate_transcript(self, markdown_content: str, on_part=None) -> str:
        """
        Generate podcast transcript from markdown.

        Args:
            markdown_content: Full markdown file content
            on_part: Optional callback ``on_part(idx, text)`` fired after each
                part is cleaned, enabling pipeline overlap (e.g. starting TTS
                for a part while later parts are still generating).

        Returns:
            Transcript with <Person1> and <Person2> tags
        """
        # The strategy expects a chain-like object it can .invoke(params) with.
        # We provide a RunnableLambda backed by the raw OpenAI client so we keep
        # full control over JSON parsing (our parser is tolerant of truncation
        # and DeepSeek's doubled-brace quirk, unlike langchain-openai's parser).
        chain = RunnableLambda(self._call_llm_json)

        # Prepare parameters
        audience = (self.audience_prompt or (
            "a general audience interested in the topic. Define specialized "
            "terms where helpful and keep explanations concrete and "
            "self-contained."
        ))
        prompt_params = {
            "podcast_topic": self.podcast_topic,
            "roles_person1": self.config_conversation["roles_person1"],
            "roles_person2": self.config_conversation["roles_person2"],
            "host1_name": self.config_conversation["host1_name"],
            "host2_name": self.config_conversation["host2_name"],
            "podcast_name": self.config_conversation["podcast_name"],
            "podcast_tagline": self.config_conversation["podcast_tagline"],
            "output_language": self.config_conversation["output_language"],
            "audience": audience,
            "familiar_topics": self.familiar_clause or "none",
        }

        # Generate using long-form strategy
        transcript = self.strategy.generate(chain, markdown_content, prompt_params, on_part=on_part)

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

    def build_combined_content(self, sources: list, intro_text: str = "") -> str:
        """
        Fetch full source content and assemble topic-delimited blocks for
        transcript generation from a list of structured :class:`Source`
        objects. Callers pass sources directly; the engine has no dependency
        on any brief markdown format.
        """
        arxiv_sources = [s for s in sources if s.kind == "arxiv"]
        pdf_sources = [s for s in sources if s.kind == "pdf"]
        blog_sources = [s for s in sources if s.kind == "blog"]
        print(f"[podcastfy] {len(sources)} sources "
              f"({len(arxiv_sources)} arXiv, {len(pdf_sources)} pdf, {len(blog_sources)} blog)")

        pdf_paths = download_arxiv_pdfs(sources, target_dir=self.papers_dir)
        print(f"[podcastfy] {len(pdf_paths)} arXiv PDF(s) available")

        # Map PDF paths by arxiv_id for quick lookup.
        pdf_by_id = {os.path.splitext(os.path.basename(p))[0]: p for p in pdf_paths}

        # Fetch full main-article text for blog (non-arXiv) sources so the
        # LLM gets the full narrative (e.g. the Dreadnode case studies).
        web_by_url = fetch_web_content(sources, target_dir=self.web_dir)
        print(f"[podcastfy] {len(web_by_url)} blog source(s) fetched")

        # Order sources thematically (taxonomy-aware when labels are known).
        ordered = self._order_sources_thematically(sources)

        # Build the combined content as topic-delimited blocks.
        # INTRO block: intro text (for the opening overview).
        intro_lines = intro_text.strip().split("\n")[:3] if intro_text.strip() else []
        combined_content = "=== INTRO ===\n"
        if intro_lines:
            combined_content += "\n".join(intro_lines)
        combined_content += "\n=== END INTRO ===\n"

        # MEMORY block: prior-episode summaries for topics this episode covers
        # (cross-episode continuity — see pipeline.memory). It becomes its own
        # chunk/part, listing what was covered before so a topic part can
        # reference a genuine continuation (and only a genuine one).
        mem_section = self._memory_section()
        if mem_section:
            combined_content += "\n" + mem_section

        # Topic blocks.
        per_paper_chars = self.config_conversation.get("per_paper_chars", 20000)
        tail_chars = self.config_conversation.get("per_paper_tail_chars", 5000)
        per_web_chars = self.config_conversation.get("per_web_chars", 12000)
        pdf_extractor = SimplePDFExtractor()
        topic_titles = []
        for src in ordered:
            # A source with NO content is dropped entirely (never aired): no
            # excerpt in the brief AND no fetched full text. The collect stage
            # already drops empty-body items, so this only fires for hand-
            # edited briefs that keep a bare bullet with no excerpt.
            has_full = ((src.kind == "arxiv" and src.arxiv_id and src.arxiv_id in pdf_by_id)
                        or (src.kind == "blog" and src.url in web_by_url)
                        or (src.kind == "pdf" and src.local_path))
            if not src.excerpt.strip() and not has_full:
                print(f"[podcastfy] no content for topic, dropping: {src.title} ({src.url})")
                continue
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
            elif src.kind == "pdf" and src.local_path:
                # A user-provided PDF already on disk (no download/fetch).
                print(f"[podcastfy] extracting {os.path.basename(src.local_path)}...")
                try:
                    pdf_text = pdf_extractor.extract_text(src.local_path)
                    pdf_text = " ".join(pdf_text.split())
                    body = pdf_text[:per_paper_chars]
                    if len(pdf_text) > per_paper_chars:
                        body += "\n\n[EXCERPT END]\n\n" + pdf_text[-tail_chars:]
                    combined_content += f"\nFULL PAPER TEXT:\n{body}\n"
                except Exception as e:
                    logger.warning(f"Skipping {src.local_path}: {e}")
            elif src.kind == "blog" and src.url in web_by_url:
                # Blog source: append the fetched full main-article text.
                web_text = web_by_url[src.url]
                body = web_text[:per_web_chars]
                if len(web_text) > per_web_chars:
                    body += "\n\n[EXCERPT END]\n\n" + web_text[-MAX_WEB_TAIL_CHARS:]
                combined_content += f"\nFULL PAGE TEXT:\n{body}\n"

            combined_content += "=== END TOPIC ===\n"

        # RECAP block: list of all topic titles (for the closing summary).
        combined_content += "\n=== RECAP ===\n"
        combined_content += "Topics covered in this episode:\n"
        for i, title in enumerate(topic_titles, 1):
            combined_content += f"{i}. {title}\n"
        combined_content += "=== END RECAP ===\n"

        return combined_content

    def _memory_section(self) -> str:
        """A ``=== MEMORY ===`` block listing prior-episode coverage for topics
        this episode covers. Empty string when there is no continuity memory."""
        if not self.memory_context:
            return ""
        lines = ["=== MEMORY: PRIOR EPISODES ===",
                 "Prior coverage from earlier episodes, matched by topic label. "
                 "This is the COMPLETE set of prior coverage you may reference. "
                 "Reference a listed item ONLY when this episode's story "
                 "directly continues it — a new development or successor release "
                 "of the same product, model, paper, or news thread. A listed "
                 "prior item that merely shares the same broad topic/theme but "
                 "is a different story is NOT a continuation and must NOT be "
                 "referenced. Never invent prior coverage not listed here."]
        for topic in sorted(self.memory_context):
            entries = self.memory_context[topic]
            if not entries:
                continue
            lines.append(f"TOPIC: {topic}")
            for entry in entries:
                lines.append(f"- {entry}")
        lines.append("=== END MEMORY ===")
        return "\n".join(lines)

    def _order_sources_thematically(self, sources) -> list:
        """Order sources by topic-aware grouping when per-item labels are
        available; otherwise fall back to the legacy hardcoded thematic order.

        Within-episode cross-referencing depends on related items being ADJACENT
        (each part sees only its own input + prior dialogue). So same-theme
        items are clustered, the theme order follows a stable canonical order,
        and within a theme items keep their original (rank) order. Sources the
        run has no label for keep the legacy ``_THEMATIC_ORDER`` behavior, so
        nothing regresses when labels are absent.
        """
        items = self.items or []
        label_by_url = _label_by_url(items)
        if not label_by_url:
            return _thematic_order_legacy(sources)

        unmatched = []
        by_label_text = {}
        for src in sources:
            label = label_by_url.get(_src_key(src))
            if label is None:
                unmatched.append(src)
                continue
            by_label_text.setdefault(label, []).append(src)

        ordered = []
        for label in _THEME_ORDER:
            group = by_label_text.pop(label, None)
            if group:
                ordered.extend(group)
        # Remaining labeled buckets (not in the canonical list) append in a
        # stable (label-alphabetical) order.
        for label in sorted(by_label_text):
            ordered.extend(by_label_text[label])
        # Unlabeled sources keep the legacy fallback ordering.
        if unmatched:
            ordered.extend(_thematic_order_legacy(unmatched))
        return ordered


# Canonical THEME order (overrides the hardcoded _THEMATIC_ORDER whenever the
# run carries per-item taxonomy labels). It lists the taxonomy buckets in
# narrative order, so items of the same (or adjacent) buckets are ordered
# adjacently — the enabler of within-episode cross-referencing, which needs
# related items in consecutive parts. A bucket not listed, or a source with no
# label, keeps the legacy behavior (rank order / substring match).
_THEME_ORDER: list[str] = [
    "post_training", "pretraining", "architecture_world_models",
    "model_release", "agents", "robotics", "multimodal", "safety_alignment",
    "incident", "policy", "business_economics", "ai_for_science",
    "benchmarks", "research_theory", "interpretability",
    "inference_infrastructure", "retrieval_rag",
]


def _src_key(src) -> str:
    return (src.url or "").strip().rstrip("/").lower()


def _label_by_url(items: list) -> dict:
    """Run-item url (normalized) -> single taxonomy id, from the judges' labels
    carried on the chosen RankedItems (see pipeline.generate/_brief_text)."""
    out = {}
    for it in items:
        t = getattr(it, "topics", None) or {}
        tid = next((k for k in t if t[k] > 0), None)
        if tid:
            out[(getattr(it, "url", "") or "").strip().rstrip("/").lower()] = tid
    return out


def _thematic_order_legacy(sources) -> list:
    """The legacy ordering: match sources to _THEMATIC_ORDER by title substring;
    anything unmatched is appended in its original order (kept for runs without
    per-item labels and for unmatched/label-less sources)."""
    ordered = []
    remaining = list(sources)
    for target in SimplePodcastGenerator._THEMATIC_ORDER:
        for i, src in enumerate(remaining):
            if target.lower() in src.title.lower():
                ordered.append(src)
                remaining.pop(i)
                break
    ordered.extend(remaining)
    return ordered
