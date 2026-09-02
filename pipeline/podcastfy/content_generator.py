"""
Content Generator Module

This module is responsible for generating Q&A content based on input texts using
LangChain and various LLM backends. It handles the interaction with the AI model and
provides methods to generate and save the generated content.
"""

import os
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List


logger = logging.getLogger(__name__)

# Default LLM endpoint when LLM_API_BASE is unset — the same SkaiNet gateway the
# core pipeline (pipeline/llm.py) uses, so podcastfy works with just
# SKAINET_API_KEY, matching the rest of the pipeline.
_SKAINET_BASE_URL = "https://chat.model.tngtech.com/v1"


class LLMBackend:
    def __init__(
        self,
        is_local: bool,
        temperature: float,
        max_output_tokens: int,
        model_name: str,
        api_key_label: str = "GEMINI_API_KEY",
        api_base_label: str = "LLM_API_BASE",
        orchestrator: str = "openai",
    ):
        """
        Initialize the LLMBackend.

        Args:
            is_local (bool): Whether to use a local LLM or not.
            temperature (float): The temperature for text generation.
            max_output_tokens (int): The maximum number of output tokens.
            model_name (str): The name of the model to use.
            api_key_label (str): Env var name holding the API key.
            api_base_label (str): Env var name holding an OpenAI-compatible base URL.
            orchestrator (str): 'openai' (default, OpenAI-compatible endpoint) or 'gemini'.
        """
        self.is_local = is_local
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model_name = model_name
        self.is_multimodal = not is_local  # Does not assume local LLM is multimodal

        common_params = {
            "temperature": temperature,
            "presence_penalty": 0.1,  # Encourage diverse content
            "frequency_penalty": 0.1,  # Avoid repetition
        }

        if is_local:
            # Lazy import: only needed for the local-LLM path.
            from langchain_community.llms.llamafile import Llamafile
            self.llm = Llamafile()  # replace with ollama
        elif orchestrator == "gemini" or "gemini" in self.model_name.lower():
            # Lazy import: only needed for the Gemini path.
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                api_key=os.environ.get("GEMINI_API_KEY"),
                model=model_name,
                max_output_tokens=max_output_tokens,
                **common_params,
            )
        else:
            # OpenAI-compatible endpoint (e.g. SkaiNet / tngtech) — the default.
            from langchain_openai import ChatOpenAI
            base_url = os.environ.get(api_base_label) or _SKAINET_BASE_URL
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.environ.get(api_key_label),
                base_url=base_url,
                max_tokens=max_output_tokens,
                timeout=120,
                max_retries=2,
                # DeepSeek honours response_format=json_object, which suppresses
                # the "thinking"/reasoning that otherwise leaks into content.
                response_format={"type": "json_object"},
            )

class LongFormContentGenerator:
    """
    Handles generation of long-form podcast conversations by breaking content into manageable chunks.
    
    Uses a "Content Chunking with Contextual Linking" strategy to maintain context between segments
    while generating longer conversations.
    
    Attributes:
        llm_chain: The LangChain chain used for content generation
    """
    
    def __init__(self, chain, llm, config_conversation: Dict[str, Any], ):
        """
        Initialize ConversationGenerator.
        
        Args:
            llm_chain: The LangChain chain to use for generation
            config_conversation: Conversation configuration dictionary
        """
        self.llm_chain = chain
        self.llm = llm
        self.max_num_chunks = config_conversation.get("max_num_chunks", 10)  # Default if not in config
        self.min_chunk_size = config_conversation.get("min_chunk_size", 200)  # Default if not in config
        # Per-source input reference scale (brief < 1.0 < deep-dive). This
        # scales how much source TEXT the LLM sees — never the output budget.
        self.depth_factor = float(config_conversation.get("depth_factor", 1.0))
        # Output word budget per part (derived by RunConfig.budget). These set
        # the hard ceiling each LLM part must stay under.
        self.intro_words = int(config_conversation.get("intro_words", 200))
        self.per_source_words = int(config_conversation.get("per_source_words", 240))
        self.recap_words = int(config_conversation.get("recap_words", 280))

    def __calculate_chunk_size(self, input_content: str) -> int:
        """
        Calculate chunk size based on input content length.
        Returns 0 when topic-aware markers are present (the chunker will
        split on === TOPIC === / === INTRO === / === RECAP === boundaries
        instead of by character count).

        Args:
            input_content: Input text content

        Returns:
            Calculated chunk size, or 0 for topic-aware mode.
        """
        # Topic-aware mode: no char-based chunking needed.
        if "=== TOPIC:" in input_content or "=== INTRO ===" in input_content:
            return 0

        input_length = len(input_content)
        if input_length <= self.min_chunk_size:
            return input_length
        
        maximum_chunk_size = input_length // self.max_num_chunks
        if maximum_chunk_size >= self.min_chunk_size:
            return maximum_chunk_size
        
        # Calculate chunk size that maximizes size while maintaining minimum chunks
        return input_length // (input_length // self.min_chunk_size)

    def chunk_content(self, input_content: str, chunk_size: int) -> List[str]:
        """
        Split input content into chunks. Uses topic-aware splitting when
        === TOPIC: / === INTRO === / === RECAP === markers are present;
        otherwise falls back to sentence-aware character-count chunking.

        Args:
            input_content (str): The input text to chunk
            chunk_size (int): Maximum size of each chunk (0 = topic-aware mode)

        Returns:
            List[str]: List of content chunks
        """
        # Topic-aware mode: split on block markers.
        if chunk_size == 0:
            return self._chunk_by_topics(input_content)

        # Fallback: sentence-aware char-count chunking (legacy path).
        sentences = input_content.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length
            
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        return chunks

    @staticmethod
    def _chunk_by_topics(input_content: str) -> List[str]:
        """
        Split content on === INTRO ===, === TOPIC: ... ===, and === RECAP ===
        markers. Each block becomes one chunk. Blocks are returned in order.

        Also handles === PAPER: ... === markers for backward compatibility with
        load_markdown_and_pdfs (each === PAPER === block = one chunk).
        """
        import re as _re

        # Pattern matches any of the block start markers and captures until
        # the corresponding END marker (or the next start marker).
        # Supported: === INTRO === ... === END INTRO ===
        #            === TOPIC: <title> === ... === END TOPIC ===
        #            === RECAP === ... === END RECAP ===
        #            === PAPER: <name> === (no END marker — ends at next block)
        pattern = _re.compile(
            r"(=== (?:INTRO|TOPIC:|RECAP|PAPER:)[^\n]*===.*?)(?=(?:=== (?:INTRO|TOPIC:|RECAP|PAPER:)[^\n]*===)|$)",
            _re.DOTALL,
        )
        chunks = [m.group(1).strip() for m in pattern.finditer(input_content)]
        if chunks:
            return chunks
        # No markers found: fall back to single chunk.
        return [input_content] if input_content.strip() else []

    def _part_word_cap(self, part_idx: int, total_parts: int) -> int:
        """Word ceiling for a part: intro, then each source, then recap."""
        if part_idx == 0:
            return self.intro_words
        if part_idx == total_parts - 1:
            return self.recap_words
        return self.per_source_words

    @staticmethod
    def _turn_range_for_cap(cap: int) -> tuple[int, int, int]:
        """(lo, hi, min) turn counts for a part with a ``cap``-word budget.

        Conversational pace ≈ 24-34 words/turn; the min feeds the retry loop's
        under-generation floor. Not scaled by depth_factor — the cap already
        encodes how much the part should say.
        """
        lo = max(3, round(cap / 34.0))
        hi = max(4, round(cap / 24.0))
        mn = max(3, round(cap / 30.0))
        return lo, hi, mn

    def enhance_prompt_params(self, prompt_params: Dict,
                              part_idx: int,
                              total_parts: int,
                              chat_context: str,
                              chunk_len: int = 0) -> Dict:
        """
        Enhance prompt parameters for long-form content generation.

        Args:
            prompt_params (Dict): Original prompt parameters
            part_idx (int): Index of current conversation part
            total_parts (int): Total number of conversation parts
            chat_context (str): Chat context from previous parts
            chunk_len (int): Length of the current chunk in characters.
                Retained for signature compatibility (no longer drives length).

        Returns:
            Dict: Enhanced prompt parameters with part-specific instructions
        """
        enhanced_params = prompt_params.copy()
        # Initialize part_instructions with chat context
        enhanced_params["context"] = chat_context

        host1 = prompt_params.get("host1_name", "Person1")
        host2 = prompt_params.get("host2_name", "Person2")

        # Every part gets a word ceiling derived from the episode budget; the
        # prompt instructs it and _trim_to_words enforces it deterministically
        # afterward. We deliberately do NOT set a tight per-call token ceiling:
        # DeepSeek emits a reasoning preamble before the JSON, so a small
        # max_tokens would truncate that and cancel the whole part.
        cap_words = self._part_word_cap(part_idx, total_parts)
        turn_lo, turn_hi, _ = self._turn_range_for_cap(cap_words)

        COMMON_INSTRUCTIONS = """
            Podcast conversation so far is given in CONTEXT.
            Continue the natural flow of conversation. Follow-up on the very previous point/question without repeating topics already discussed in earlier PARTS. A topic previewed in the introduction is NOT already discussed. It gets its full treatment here.
            The first speaker must differ from the last speaker in CONTEXT. If the last tag in CONTEXT is <Person1>, the first to speak now should be <Person2>, and vice versa.
            This is a live conversation without any breaks. Avoid statements such as "we'll discuss after a short break" or "Okay, so, picking up where we left off".
            TRANSITION in two parts. First a thematic bridge: connect the previous topic to the new one (why Y follows X). Then a signpost: explicitly name the new topic so the listener hears the boundary. Vary the signpost phrasing every time. Never reuse the same signpost twice in the episode.
            Examples:
            - "[X] is about architecture. Turning to [Y], the training loop is a different bet entirely."
            - "So [X] solves the speed problem. Next up, [Y] asks the reliability question."
            - "That [X] sets up [Y], which takes the opposite approach."
            Vary with: "Turning to", "Next up", "Now onto", "Pivoting to", "That sets up", "Shifting to", "Moving to", "On the other hand", etc.
        """

        # Add part-specific instructions
        if part_idx == 0:
            enhanced_params["instruction"] = f"""
            You are generating the INTRODUCTION of the podcast.
            Do NOT explain any topic yet. Instead:
            1. BOTH hosts greet the audience by name. Person1 ({host1}) says one welcoming sentence introducing himself, then Person2 ({host2}) says one welcoming sentence introducing herself and greeting {host1}. Make it warm and conversational, not a rushed "Hey there!".
            2. Give a themed overview: group the topics into 2-3 themes. Weave them together conversationally. Do NOT label themes explicitly ("first theme," "second theme," "third theme"). Use natural connective phrases like "We'll also dive into," "Then we'll cover," "Finally, we'll discuss." For each topic, use a relative clause or flowing sentence that says what it does, not a standalone fragment. Integrate a brief "why it matters" into the theme, not as a separate label. Interleave genuine reactions between themes. One host reacts to the previous theme, the other continues to the next. Do NOT explain mechanisms, cite numbers, or describe how things work. Save those for the topic discussions.
               Example: "In today's episode, we'll cover three major model releases. A, which achieves unprecedented generation speeds. B, which brings multimodal capabilities to a compact architecture. And C, which uses an end-to-end self-improvement loop." [Reaction: "And those are pushing boundaries we didn't think possible a year ago."] "We'll also dive into agent frameworks, covering D's breakthrough in harness scaling and E's flexible navigation. Because raw intelligence doesn't mean much without a reliable environment to operate within." "Right. Finally, we'll discuss major industry news, including F's big acquisition and a shocking audit revealing benchmark cheating."
            3. End with "Let's begin!" or similar.
            Keep this part to {turn_lo} to {turn_hi} turns total and at most {cap_words} words. The overview should tease topics by name and significance, woven into flowing sentences, not listed as fragments. No analogy or numbers in the intro.
            """
        elif part_idx == total_parts - 1:
            enhanced_params["instruction"] = f"""
            You are generating the FINAL part (closing recap) of the podcast.
            {COMMON_INSTRUCTIONS}
            This part's INPUT contains a list of all topics covered in the episode.
            Do NOT re-explain any topic in depth. Instead:
            1. Synthesize the episode's arc: group the topics into 2-3 themes and summarize what each theme revealed. Show the through-line, not just a list.
            2. Connect the themes: show how they relate and build on each other (e.g. "We started with speed, then saw how reliability matters just as much, and finally learned that even our benchmarks can't be trusted").
            3. End with a provocative question for the audience, then Person1 ({host1}) says a brief goodbye addressing {host2} ("Until next time, keep digging...").
            Keep this part to {turn_lo} to {turn_hi} turns and at most {cap_words} words. This is a recap with synthesis, not a new discussion. No per-topic analogy needed here.
            """
        else:
            enhanced_params["instruction"] = f"""
            You are generating part {part_idx+1} of {total_parts} of the podcast.
            {COMMON_INSTRUCTIONS}
            Discuss the topic(s) in the INPUT. You MUST fully discuss every topic in this part's INPUT even if it was previewed in the introduction. The introduction only names topics. The full discussion happens here.

            For each topic, follow this structure:
            1. WHAT it does. One plain-language sentence. Skip basics the audience knows.
            2. HOW it works. The key mechanism or technique.
            3. ANALOGY. One per topic, mandatory. One concrete, vivid analogy that maps to the actual mechanism, not decorative. Do not skip it. Do not force a second.
               Quality bar (style reference only, do not reuse): "An autoregressive model paints a mural left-to-right, one pixel at a time. It physically cannot paint the right side until the left is done. DiffusionGemma sketches the whole mural in broad strokes, then sharpens every detail at once."
            4. WHY it matters. The practical impact for AI researchers and engineers.

            Cite at most 1-2 striking numbers per topic. Do NOT recite every metric. Lead with what the number means (the delta, ratio, or comparison), not the raw endpoints. "Jumps 9 points to 85.4" beats "from 76.7 to 85.4." "Doubles to 35.4" beats "from 17.2 to 35.4." Never stack more than two numbers in a single turn. Vivid cost pairs like "$15 vs $574" may stay as-is. The gap is the story.
            If the INPUT contains named case studies, specific models, or concrete behaviors, USE THEM BY NAME. Do not paraphrase them into a generality. For example, if the INPUT says a model ran `git clone` on the writeup repo to read the flag, say which model and say "git clone", not "a model cheated".
            Keep this part to {turn_lo} to {turn_hi} turns and at most {cap_words} words of dialogue.
            """

        return enhanced_params

    def generate_long_form(
        self, 
        input_content: str, 
        prompt_params: Dict,
        on_part=None,
    ) -> str:
        """
        Generate a complete long-form conversation using chunked content.
        
        Args:
            input_content (str): Input text for conversation
            prompt_params (Dict): Base prompt parameters
            on_part (callable): Optional callback ``on_part(idx, text)`` fired
                after each part is cleaned, enabling pipeline overlap (e.g.
                starting TTS for a part while later parts are still generating).
            
        Returns:
            str: Generated long-form conversation
        """
        # Get chunk size
        chunk_size = self.__calculate_chunk_size(input_content)

        chunks = self.chunk_content(input_content, chunk_size)
        conversation_parts = []
        chat_context = input_content
        num_parts = len(chunks)
        print(f"Generating {num_parts} parts")
        
        for i, chunk in enumerate(chunks):
            cap_words = self._part_word_cap(i, num_parts)
            _, _, min_turns = self._turn_range_for_cap(cap_words)
            enhanced_params = self.enhance_prompt_params(
                prompt_params,
                part_idx=i,
                total_parts=num_parts,
                chat_context=chat_context,
                chunk_len=len(chunk),
            )
            enhanced_params["input_text"] = chunk
            response = self._invoke_with_retry(
                enhanced_params,
                min_turns=min_turns,
            )
            response = ContentCleanerMixin._strip_preamble(response)
            # Enforce the part's word ceiling deterministically (prompt + token
            # cap are soft; this makes the total transcript conform to length).
            trimmed = ContentCleanerMixin._trim_to_words(response, cap_words)
            if trimmed != response:
                print(f"      [part {i+1}/{num_parts}] trimmed "
                      f"{len(response.split())} -> {len(trimmed.split())} words "
                      f"(cap {cap_words})")
                response = trimmed
            if on_part is not None:
                on_part(i, response)
            if i == 0:
                chat_context = response
            else:
                chat_context = chat_context + response
            # Cap the context window so very long podcasts do not exceed the LLM
            # context limit: keep at most the last ~60k chars of prior dialogue.
            if len(chat_context) > 60000:
                chat_context = chat_context[-60000:]
            print(f"Generated part {i+1}/{num_parts}: Size {len(chunk)} characters.")
            #print(f"[LLM-START] Step: {i+1} ##############################")
            #print(response)
            #print(f"[LLM-END] Step: {i+1} ##############################")
            conversation_parts.append(response)

        return self.stitch_conversations(conversation_parts)

    def _invoke_with_retry(self, enhanced_params: dict, max_attempts: int = 6,
                           min_turns: int = 0) -> str:
        """
        Invoke the chain, retrying on empty results, transient errors, or
        under-generation (too few turns) so a single flaky LLM call (truncated
        JSON, network blip, under-allocation, token-budget exhaustion) does
        not abort the whole episode or starve a topic.

        Args:
            enhanced_params: Prompt parameters for the LLM chain.
            max_attempts: Number of retry attempts on failure.
            min_turns: Minimum number of <PersonN> turns expected. If the
                response has fewer, retry. Applies to ALL parts (intro, mid,
                recap). Set to 0 to disable the turn-count check.
        """
        best_response = ""
        for attempt in range(max_attempts):
            try:
                response = self.llm_chain.invoke(enhanced_params)
            except Exception as e:
                logger.warning(f"LLM invoke error (attempt {attempt+1}): {e}")
                if attempt == max_attempts - 1:
                    raise
                continue
            response = ContentCleanerMixin._strip_preamble(response)
            if len(response.strip()) < 40:
                logger.warning(
                    f"LLM returned empty/too-short result (attempt {attempt+1}); retrying..."
                )
                continue
            # Turn-count check (safety net for under-generation).
            if min_turns > 0:
                turn_count = response.count("<Person1>") + response.count("<Person2>")
                if turn_count < min_turns:
                    logger.warning(
                        f"Generated {turn_count} turns, need {min_turns} "
                        f"(attempt {attempt+1}); retrying..."
                    )
                    if len(response) > len(best_response):
                        best_response = response
                    continue
            return response
        # All attempts failed; return the longest attempt (graceful degradation).
        return best_response if best_response else response

    def stitch_conversations(self, parts: List[str]) -> str:
        """
        Combine conversation parts with smooth transitions.
        
        Args:
            parts (List[str]): List of conversation parts
            
        Returns:
            str: Combined conversation
        """
        # Simply join the parts, preserving all markup
        return "\n".join(parts)
# Make BaseContentCleaner a mixin class
class ContentCleanerMixin:
    """
    Mixin class containing common transcript cleaning operations.
    
    Provides reusable cleaning methods that can be used by different content generation strategies.
    Methods use protected naming convention (_method_name) as they are intended for internal use
    by the strategies.
    """
    
    @staticmethod
    def _strip_preamble(text: str) -> str:
        """
        Remove any content that precedes the first Person tag.

        Some OpenAI-compatible backends (e.g. DeepSeek via SkaiNet) prepend a
        "thinking" / reasoning preamble to the response. This keeps only the
        actual dialogue starting from the first <Person1> or <Person2> tag.
        """
        try:
            match = re.search(r"<Person[12]>", text)
            if match:
                return text[match.start():].strip()
            return text.strip()
        except Exception:
            return text.strip()

    @staticmethod
    def _keep_tagged_dialogue(text: str) -> str:
        """
        Keep only well-formed <Person1>/<Person2> tagged dialogue, dropping any
        stray reasoning/drafting that leaks between or around the tags.

        Reasoning models (e.g. DeepSeek via SkaiNet) sometimes interleave
        unwrapped draft lines such as "Person1: ...", "[Person1]", "Let's
        draft:", "Need ...", or dump whole thinking blocks inside a tag. This
        extracts only the tagged turns and joins them.
        """
        if not text:
            return ""

        # First drop any leading reasoning up to the first tag (also strips
        # "response" markers that sometimes precede a tag).
        text = ContentCleanerMixin._strip_preamble(text)

        # Normalize tagged blocks that may span multiple lines / be missing
        # closers. Match from an opening <PersonN> to the next opening tag or
        # end, then build strict pairs.
        pattern = r"<Person([12])>(.*?)(?=<Person[12]>|$)"
        blocks = re.findall(pattern, text, flags=re.DOTALL)
        kept = []
        for person_num, content in blocks:
            # Drop content that looks like reasoning/drafting rather than speech:
            # it is only kept if it contains some letters beyond stray notes.
            content = content.strip()
            # Remove lines that are pure drafting markers.
            content = ContentCleanerMixin._drop_draft_lines(content)
            content = " ".join(content.split()).strip()
            if not content:
                continue
            kept.append(f"<Person{person_num}>{content}</Person{person_num}>")
        joined = "\n".join(kept)
        # Normalize accidental double closing tags (</Person1></Person1>)
        return re.sub(r"(</Person[12]>)\s*\1", r"\1", joined)

    @staticmethod
    def _drop_draft_lines(content: str) -> str:
        """
        Remove lines from a turn that look like the model's internal drafting
        rather than spoken dialogue (e.g. "Person1: ...", "Let's draft:",
        "Need ...", "Perhaps structure: ...", "So last tag is ...").

        A line is dropped if it starts with a recognized drafting marker or if
        it is an explicit "Speaker: ..." draft line.
        """
        markers = (
            "person1:", "person2:", "[person1]", "[person2]",
            "let's draft", "lets draft",
            "let's craft", "lets craft",
            "let's generate", "lets generate",
            "let's write", "lets write",
            "let's get into it", "lets get into it",
            "need ", "perhaps ", "maybe structure",
            "we need to", "we'll fit", "we can produce",
            "so last tag is", "thus first to speak",
            "also need", "not necessary", "life-craft", "let's do",
        )
        kept_lines = []
        for line in content.splitlines():
            low = line.strip().lower()
            if not low:
                continue
            # A line that is itself a "Speaker: <speech>" draft is dropped.
            if re.match(r"^\s*(?:<)?(?:person[12]|p1|p2)(?:>)?\s*:", low):
                continue
            if any(low.startswith(m) for m in markers):
                continue
            if low in ("...", "…", "wait", "ok", "ok.") or len(low) <= 2:
                continue
            kept_lines.append(line.strip())
        return " ".join(kept_lines).strip()

    @staticmethod
    def _strip_dialogue_tags(text: str) -> str:
        """Remove <PersonN> tags so word counting reflects spoken text only."""
        return re.sub(r"</?Person[12]>", "", text)

    @staticmethod
    def _word_count(text: str) -> int:
        if not text:
            return 0
        return len(ContentCleanerMixin._strip_dialogue_tags(text).split())

    @staticmethod
    def _trim_to_words(text: str, cap: int) -> str:
        """Deterministically bound ``text`` to ``cap`` spoken words.

        Keeps whole <PersonN>...</PersonN> turns while the running total is at
        or under the cap; the next (partially fitting) turn is cut at the last
        sentence boundary within the remaining budget. Tags stay balanced.
        Returns the original text unchanged if it is already within ``cap``.
        """
        if cap <= 0 or ContentCleanerMixin._word_count(text) <= cap:
            return text
        turn_re = re.compile(r"(<Person[12]>.*?</Person[12]>)", re.DOTALL)
        matches = list(turn_re.finditer(text))
        if not matches:
            # No turns to protect: bound the raw text by word count.
            return " ".join(text.split()[:cap]).strip()
        words = 0
        last_kept_end = None
        for m in matches:
            turn = m.group(0)
            n = ContentCleanerMixin._word_count(turn)
            if words + n <= cap:
                words += n
                last_kept_end = m.end()
                if words >= cap:
                    break
                continue
            # This turn does not fit whole: keep its leading part within budget.
            trimmed = ContentCleanerMixin._trim_turn_to_words(turn, cap - words)
            if last_kept_end is not None:
                return (text[:last_kept_end] + "\n" + trimmed).strip()
            return (text[:m.start()] + trimmed).strip()
        return text[:last_kept_end].strip() if last_kept_end is not None else ""

    @staticmethod
    def _trim_turn_to_words(turn: str, budget: int) -> str:
        """Cut a single ``<PersonN>...</PersonN>`` turn to ``budget`` words at the
        last sentence boundary. Returns an empty string for a zero budget."""
        if budget <= 0:
            return ""
        m = re.match(r"(<Person[12]>)(.*?)(</Person[12]>)", turn, re.DOTALL)
        if not m:
            return turn if ContentCleanerMixin._word_count(turn) <= budget else ""
        open_tag, body, close_tag = m.groups()
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        out: list[str] = []
        words = 0
        for sentence in sentences:
            sw = len(sentence.split())
            if words + sw <= budget:
                out.append(sentence)
                words += sw
                continue
            # Partially keep this sentence up to the budget.
            keep = []
            for tok in sentence.split():
                if words + 1 <= budget:
                    keep.append(tok)
                    words += 1
                else:
                    break
            if keep:
                out.append(" ".join(keep))
            break
        body2 = " ".join(out).strip()
        if not body2:
            return ""
        return f"{open_tag} {body2} {close_tag}"

    @staticmethod
    def _clean_scratchpad(text: str) -> str:
        """
        Remove scratchpad blocks, plaintext blocks, standalone triple backticks, any string enclosed in brackets, and underscores around words.
        """
        try:
            import re
            pattern = r'```scratchpad\n.*?```\n?|```plaintext\n.*?```\n?|```\n?|\[.*?\]'
            cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
            # Remove "xml" if followed by </Person1> or </Person2>
            cleaned_text = re.sub(r"xml(?=\s*</Person[12]>)", "", cleaned_text)
            # Remove underscores around words
            cleaned_text = re.sub(r'_(.*?)_', r'\1', cleaned_text)
            return cleaned_text.strip()
        except Exception as e:
            logger.error(f"Error cleaning scratchpad content: {str(e)}")
            return text

    @staticmethod
    def _clean_tss_markup(
        input_text: str, 
        additional_tags: List[str] = ["Person1", "Person2"]
    ) -> str:
        """
        Remove unsupported TSS markup tags while preserving supported ones.
        """
        try:
            input_text = ContentCleanerMixin._clean_scratchpad(input_text)
            supported_tags = ["speak", "lang", "p", "phoneme", "s", "sub"]
            supported_tags.extend(additional_tags)

            pattern = r"</?(?!(?:" + "|".join(supported_tags) + r")\b)[^>]+>"
            cleaned_text = re.sub(pattern, "", input_text)
            cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
            cleaned_text = re.sub(r"\*", "", cleaned_text)

            for tag in additional_tags:
                cleaned_text = re.sub(
                    f'<{tag}>(.*?)(?=<(?:{"|".join(additional_tags)})>|$)',
                    f"<{tag}>\\1</{tag}>",
                    cleaned_text,
                    flags=re.DOTALL,
                )
            


            return cleaned_text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning TSS markup: {str(e)}")
            return input_text

class ContentGenerationStrategy(ABC):
    """Abstract base class for content generation strategies."""

    @abstractmethod
    def generate(
        self,
        chain,
        input_texts: str,
        prompt_params: Dict[str, Any],
        **kwargs,
    ) -> str:
        """Generate content from input texts."""
        pass

    @abstractmethod
    def clean(self, response: str, config: Dict[str, Any]) -> str:
        """Clean the generated response."""
        pass

class LongFormContentStrategy(ContentGenerationStrategy, ContentCleanerMixin):
    """
    Strategy for generating long-form content.
    
    Implements advanced content generation using chunking and context maintenance.
    Includes additional cleaning operations specific to long-form content.
    
    Note:
        - Only works with text input (no images)
        - Requires non-empty input text
    """
    
    def __init__(self, llm, content_generator_config: Dict[str, Any], config_conversation: Dict[str, Any]):
        """
        Initialize LongFormContentStrategy.
        
        Args:
            content_generator_config (Dict[str, Any]): Configuration for content generation
            config_conversation (Dict[str, Any]): Conversation configuration
        """
        self.llm = llm
        self.content_generator_config = content_generator_config
        self.config_conversation = config_conversation
    
    def validate(self, input_texts: str, image_file_paths: List[str]) -> None:
        """Validate inputs for long-form generation."""
        if not input_texts.strip():
            raise ValueError("Long-form generation requires non-empty input text")
        if image_file_paths:
            raise ValueError("Long-form generation is not available with image inputs")
            
    def generate(self, 
                chain,
                input_texts: str,
                prompt_params: Dict[str, Any],
                **kwargs) -> str:
        """Generate long-form content."""
        generator = LongFormContentGenerator(chain, self.llm, self.config_conversation)
        return generator.generate_long_form(
            input_texts,
            prompt_params,
            on_part=kwargs.get("on_part"),
        )
        
    def clean(self, 
             response: str,
             config: Dict[str, Any]) -> str:
        """Apply enhanced cleaning for long-form content."""
        # Drop any unwrapped reasoning/drafting, keeping only tagged dialogue
        tagged = self._keep_tagged_dialogue(response)
        # First apply standard cleaning using common method
        standard_clean = self._clean_tss_markup(tagged)
        # Then apply additional long-form specific cleaning
        return self._clean_transcript_response(standard_clean, config)
    
    def _clean_transcript_response(self, transcript: str, config: Dict[str, Any]) -> str:
        """
        Clean transcript using a two-step process with LLM-based cleaning.
        
        First cleans the markup using a specialized prompt template, then rewrites
        for better flow and consistency using a second prompt template.
        
        Args:
            transcript (str): Raw transcript text that may contain scratchpad blocks
            config (Dict[str, Any]): Configuration dictionary containing LLM and prompt settings
            
        Returns:
            str: Cleaned and rewritten transcript with proper tags and improved flow
            
        Note:
            Falls back to original or partially cleaned transcript if any cleaning step fails
        """
        logger.debug("Starting transcript cleaning process")

        final_transcript = self._fix_alternating_tags(transcript)
        
        logger.debug("Completed transcript cleaning process")
        
        return final_transcript

        
    def _fix_alternating_tags(self, transcript: str) -> str:
        """
        Ensures transcript has properly alternating Person1 and Person2 tags.
        
        Merges consecutive same-person tags and ensures proper tag alternation
        throughout the transcript.
        
        Args:
            transcript (str): Input transcript text that may have consecutive same-person tags
            
        Returns:
            str: Transcript with properly alternating tags and merged content
            
        Example:
            Input:
                <Person1>Hello</Person1>
                <Person1>World</Person1>
                <Person2>Hi</Person2>
            Output:
                <Person1>Hello World</Person1>
                <Person2>Hi</Person2>
                
        Note:
            Returns original transcript if cleaning fails
        """
        try:
            # Split into individual tag blocks while preserving tags
            pattern = r'(<Person[12]>.*?</Person[12]>)'
            blocks = re.split(pattern, transcript, flags=re.DOTALL)
            
            # Filter out empty/whitespace blocks
            blocks = [b.strip() for b in blocks if b.strip()]
            
            merged_blocks = []
            current_content = []
            current_person = None
            
            for block in blocks:
                # Extract person number and content
                match = re.match(r'<Person([12])>(.*?)</Person\1>', block, re.DOTALL)
                if not match:
                    continue
                    
                person_num, content = match.groups()
                content = content.strip()
                
                if current_person == person_num:
                    # Same person - append content
                    current_content.append(content)
                else:
                    # Different person - flush current content if any
                    if current_content:
                        merged_text = " ".join(current_content)
                        merged_blocks.append(f"<Person{current_person}>{merged_text}</Person{current_person}>")
                    # Start new person
                    current_person = person_num
                    current_content = [content]
            
            # Flush final content
            if current_content:
                merged_text = " ".join(current_content)
                merged_blocks.append(f"<Person{current_person}>{merged_text}</Person{current_person}>")
                
            return "\n".join(merged_blocks)
            
        except Exception as e:
            logger.error(f"Error fixing alternating tags: {str(e)}")
            return transcript  # Return original if fixing fails

    def compose_prompt_params(self,
                            config_conversation: Dict[str, Any],
                            image_file_paths: List[str] = [],
                            image_path_keys: List[str] = [],
                            input_texts: str = "") -> Dict[str, Any]:
        """Compose prompt parameters for long-form content generation."""
        return {
            "roles_person1": config_conversation.get("roles_person1"),
            "roles_person2": config_conversation.get("roles_person2"),
            "podcast_name": config_conversation.get("podcast_name"),
            "podcast_tagline": config_conversation.get("podcast_tagline"),
            "output_language": config_conversation.get("output_language"),
        }
