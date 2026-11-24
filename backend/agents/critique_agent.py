"""Deterministic prompt and answer-quality metrics for the chat response."""
import re


_CITATION_RE = re.compile(
    r"\[([^\[\]]+?\.pdf)\s*,\s*p\.?\s*(\d+)(?:\s*[-–]\s*(\d+))?\]",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(
    r"^\s*(.+?\.pdf)(?:\s*\(\s*p\.?\s*(\d+)\s*\))?\s*$",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "are", "as",
    "at", "be", "because", "been", "before", "being", "but", "by", "can",
    "could", "describe", "do", "does", "for", "from", "give", "has", "have",
    "help", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please",
    "should", "so", "tell", "that", "the", "their", "them", "there", "these",
    "this", "to", "was", "what", "when", "where", "which", "who", "why",
    "with", "would", "you", "your",
    "answer", "bullet", "citations", "cite", "common", "document", "documents",
    "from", "include", "indexed", "list", "page", "pages", "short", "summarise",
    "summarize", "summarizze", "summary", "present",
}
_RELEVANCE_INSTRUCTION_WORDS = {
    "according", "each", "exact", "explain", "factual", "point", "points",
    "pdf", "statement", "statements", "two", "using",
}
_NON_DOCUMENT_EVIDENCE_RE = re.compile(
    r"\b(urgent medical alert|urgent medical warning|seek emergency care|"
    r"call emergency services|nearest emergency room|do not wait for an ai assessment|"
    r"medical emergency|immediate emergency medical evaluation)\b",
    re.IGNORECASE,
)
_RELEVANCE_WORD_FORMS = {
    "diagnosis": "diagnos",
    "diagnoses": "diagnos",
    "diagnosed": "diagnos",
    "diagnostic": "diagnos",
    "diagnostics": "diagnos",
    "diagnosing": "diagnos",
    "manage": "manag",
    "managed": "manag",
    "management": "manag",
    "manages": "manag",
    "managing": "manag",
    "guideline": "guideline",
    "guidelines": "guideline",
}


class CritiqueAgent:
    """Score prompt clarity and answer evidence with explainable heuristics."""

    AGENT_NAME = "Quality Metrics Agent"

    def critique(self, query: str, draft_answer: str, sources: list[str]) -> dict:
        prompt_quality = self._score_prompt(query)
        answer_quality = self._score_answer(query, draft_answer, sources)

        validated = draft_answer.strip()
        disclaimer = (
            "This is educational information from indexed documents, not a medical diagnosis. "
            "Consult a qualified healthcare professional for personal medical concerns."
        )
        if "medical disclaimer" not in validated.lower() and "not a medical diagnosis" not in validated.lower():
            validated = f"{validated}\n\n## Medical Disclaimer\n{disclaimer}"

        return {
            "validated_answer": validated,
            "issues_found": answer_quality["suggestions"],
            "quality_score": answer_quality["level"],
            "quality_metrics": {
                "prompt_quality": prompt_quality,
                "answer_quality": answer_quality,
            },
            "agent": self.AGENT_NAME,
        }

    @classmethod
    def _score_prompt(cls, query: str) -> dict:
        words = cls._tokens(query)
        content_words = [word for word in words if word not in _STOP_WORDS]
        normalized = query.lower()

        task_clear = bool(
            re.search(r"\b(what|which|when|where|why|who|how|explain|compare|"
                      r"summarize|summarise|list|describe|identify|analyze|analyse)\b",
                      normalized)
        )
        topic_specific = min(100, len(content_words) * 35)
        context_present = bool(
            re.search(
                r"\b(i have|i feel|my|patient|document|paper|study|age|"
                r"since|for \d+|after|before|during|history|diagnosed)\b",
                normalized,
            )
            or re.search(r"\b\d+\b", normalized)
        )
        output_guidance = bool(
            re.search(
                r"\b(step[- ]by[- ]step|bullet|table|brief|detailed|compare|"
                r"with sources|cite|in \d+ words|pros and cons)\b",
                normalized,
            )
        )
        dimensions = {
            "task_clarity": 100 if task_clear else 0,
            "topic_specificity": topic_specific,
            "context_detail": 100 if context_present else 0,
            "output_guidance": 100 if output_guidance else 0,
        }
        weights = {
            "task_clarity": 30,
            "topic_specificity": 30,
            "context_detail": 20,
            "output_guidance": 20,
        }
        score = round(sum(dimensions[key] * weights[key] for key in dimensions) / 100)
        suggestions = []
        if not task_clear:
            suggestions.append("State what you want to know or do.")
        if topic_specific < 70:
            suggestions.append("Name the condition, topic, or document.")
        if not context_present:
            suggestions.append("Add relevant context, such as symptoms, duration, or document context.")
        if not output_guidance:
            suggestions.append("Optionally request a format or depth, such as a short summary or bullet list.")

        return {
            "score": score,
            "level": cls._level(score, high=75, moderate=45),
            "metrics": dimensions,
            "weights": weights,
            "suggestions": suggestions,
            "note": "Prompt clarity is separate from answer quality and never blocks an answer.",
        }

    @classmethod
    def _score_answer(cls, query: str, answer: str, sources: list[str]) -> dict:
        citations = list(_CITATION_RE.finditer(answer))
        retrieved = []
        for source in sources:
            match = _SOURCE_RE.search(source)
            if match:
                page = int(match.group(2)) if match.group(2) else None
                retrieved.append((match.group(1).lower(), page))

        supported_count = 0
        for citation in citations:
            filename = citation.group(1).strip().lower()
            first_page = int(citation.group(2))
            last_page = int(citation.group(3) or first_page)
            if any(
                source_file == filename
                and (source_page is None or first_page <= source_page <= last_page)
                for source_file, source_page in retrieved
            ):
                supported_count += 1

        claim_units = cls._claim_units(answer)
        evidence_units = [
            unit for unit in claim_units
            if not cls._is_non_document_evidence(unit)
        ]
        cited_units = sum("CITATION" in unit for unit in evidence_units)
        citation_coverage = (
            round(100 * cited_units / len(evidence_units)) if evidence_units else 0
        )
        source_match = round(100 * supported_count / len(citations)) if citations else 0

        query_terms = {
            cls._relevance_key(word) for word in cls._tokens(query)
            if word not in _STOP_WORDS
            and word not in _RELEVANCE_INSTRUCTION_WORDS
            and len(word) > 2
        }
        answer_text = " ".join(claim_units)
        answer_text = _CITATION_RE.sub("", answer_text)
        answer_terms = {cls._relevance_key(word) for word in cls._tokens(answer_text)}
        relevance = (
            round(100 * len(query_terms & answer_terms) / len(query_terms))
            if query_terms else 100
        )
        detail = cls._detail_score(answer)

        metrics = {
            "citation_coverage": citation_coverage,
            "retrieved_source_match": source_match,
            "query_term_relevance": relevance,
            "answer_detail": detail,
        }
        score = round(
            citation_coverage * 0.30
            + source_match * 0.35
            + relevance * 0.20
            + detail * 0.15
        )
        suggestions = []
        if citation_coverage < 80:
            suggestions.append("Add inline file-and-page citations to factual statements.")
        if source_match < 100:
            suggestions.append("Check that cited files and page numbers appear in retrieved sources.")
        if relevance < 40:
            suggestions.append("Address the main topic terms in the question more directly.")
        if detail < 50:
            suggestions.append("Add a concise explanation or supporting detail.")

        return {
            "score": score,
            "level": cls._level(score, high=80, moderate=60),
            "metrics": metrics,
            "weights": {
                "citation_coverage": 30,
                "retrieved_source_match": 35,
                "query_term_relevance": 20,
                "answer_detail": 15,
            },
            "suggestions": suggestions,
            "note": (
                "Citation coverage counts answer sentences/bullets with at least 3 words, "
                "excluding user-reported symptom restatements and urgent-care safety notices; "
                "source match only checks whether a cited filename/page appeared in retrieved "
                "sources; it does not check whether the passage supports the claim. Relevance "
                "uses normalized word overlap, not semantic matching; detail combines response "
                "length and structure. "
                "Prompt clarity is scored independently. These signals are not a fact-check "
                "and do not verify answer or clinical correctness."
            ),
        }

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())

    @staticmethod
    def _relevance_key(word: str) -> str:
        return _RELEVANCE_WORD_FORMS.get(word, word)

    @staticmethod
    def _level(score: int, *, high: int, moderate: int) -> str:
        if score >= high:
            return "High"
        if score >= moderate:
            return "Moderate"
        return "Low"

    @staticmethod
    def _claim_units(answer: str) -> list[str]:
        units = []
        in_disclaimer = False
        in_sources = False
        for line in answer.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("# ").lower()
                in_disclaimer = "disclaimer" in heading
                in_sources = heading == "sources"
                continue
            if in_disclaimer or in_sources or not stripped:
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:]
            citation_normalized = _CITATION_RE.sub(" CITATION ", stripped)
            pieces = re.split(r"""(?<=[.!?])["'”’)]?\s+""", citation_normalized)
            units.extend(
                piece for piece in pieces
                if len(_WORD_RE.findall(piece)) >= 3
                and not piece.strip().lower().startswith(("source:", "sources:"))
            )
        return units

    @staticmethod
    def _is_non_document_evidence(unit: str) -> bool:
        normalized = unit.strip().lower()
        return (
            normalized.startswith("patient symptoms:")
            or bool(_NON_DOCUMENT_EVIDENCE_RE.search(normalized))
        )

    @staticmethod
    def _detail_score(answer: str) -> int:
        content = re.sub(r"(?m)^#{1,3}\s+.*$", "", answer)
        content = _CITATION_RE.sub("", content)
        content = re.sub(r"(?m)^[-*•]\s*", "", content)
        word_count = len(_WORD_RE.findall(content))
        if word_count >= 20:
            length_score = 100
        elif word_count >= 8:
            length_score = 75
        elif word_count >= 3:
            length_score = 55
        else:
            length_score = 0
        has_sections = len(re.findall(r"(?m)^#{1,3}\s+", answer)) >= 2
        has_paragraphs = len([line for line in answer.splitlines() if line.strip()]) >= 3
        structure_score = 100 if has_sections else 80 if has_paragraphs else 70
        return round(length_score * 0.65 + structure_score * 0.35)


critique_agent = CritiqueAgent()
