"""
resume_parser.py – Extracts structured information from a resume file
(PDF / DOCX / plain text) using IBM Granite for NLP extraction.

Extracted fields
----------------
  name, email, phone, skills, experience_years, job_titles, education,
  certifications, summary
"""

from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path
from dataclasses import dataclass, field

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from src.watsonx_client import get_llm

logger = logging.getLogger(__name__)


@dataclass
class ResumeProfile:
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: list[str] = field(default_factory=list)
    experience_years: int = 0
    job_titles: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    summary: str = ""
    raw_text: str = ""

    @property
    def experience_level(self) -> str:
        """Infer experience level from years."""
        y = self.experience_years
        if y == 0:
            return "fresher"
        if y <= 2:
            return "junior"
        if y <= 5:
            return "mid"
        if y <= 10:
            return "senior"
        return "lead"


class ResumeParser:
    """Parses a resume file and extracts structured profile data."""

    _EXTRACTION_PROMPT = textwrap.dedent("""
        Extract structured information from the resume text below.
        Return ONLY the following fields, one per line, exactly as shown:

        NAME: <full name>
        EMAIL: <email address or NONE>
        PHONE: <phone number or NONE>
        EXPERIENCE_YEARS: <integer number of years of professional experience>
        JOB_TITLES: <comma-separated list of job titles held>
        SKILLS: <comma-separated list of technical and soft skills>
        EDUCATION: <comma-separated degrees/institutions>
        CERTIFICATIONS: <comma-separated certifications or NONE>
        SUMMARY: <2-sentence professional summary>

        Resume text:
        {resume_text}
    """).strip()

    def __init__(self) -> None:
        self._llm = get_llm()

    # ── File loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_text(file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                pages = PyPDFLoader(str(file_path)).load()
                return "\n".join(p.page_content for p in pages)
            if suffix == ".docx":
                pages = Docx2txtLoader(str(file_path)).load()
                return "\n".join(p.page_content for p in pages)
            # plain text / markdown
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.error("Failed to load resume file %s: %s", file_path, exc)
            return ""

    # ── Regex-based pre-extraction ────────────────────────────────────────────

    @staticmethod
    def _regex_extract(text: str) -> dict:
        """Fast heuristic extraction as a fallback / cross-check."""
        result: dict = {}
        email_m = re.search(r"[\w.+-]+@[\w-]+\.\w+", text)
        if email_m:
            result["email"] = email_m.group()
        phone_m = re.search(r"[\+\(]?[\d\s\-\(\)]{7,15}", text)
        if phone_m:
            result["phone"] = phone_m.group().strip()
        years_m = re.search(r"(\d+)\s*\+?\s*years?", text, re.IGNORECASE)
        if years_m:
            result["experience_years"] = int(years_m.group(1))
        return result

    # ── LLM extraction ────────────────────────────────────────────────────────

    def _llm_extract(self, text: str) -> dict:
        """Use IBM Granite to extract structured fields from resume text."""
        # Limit text to avoid token overflow
        truncated = text[:4000]
        prompt = self._EXTRACTION_PROMPT.format(resume_text=truncated)
        try:
            raw = self._llm.invoke(prompt)
        except Exception as exc:
            logger.error("LLM extraction failed: %s", exc)
            return {}

        result: dict = {}
        for line in raw.strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().upper()
            value = value.strip()
            if not value or value.upper() == "NONE":
                continue
            if key == "NAME":
                result["name"] = value
            elif key == "EMAIL":
                result["email"] = value
            elif key == "PHONE":
                result["phone"] = value
            elif key == "EXPERIENCE_YEARS":
                try:
                    result["experience_years"] = int(re.search(r"\d+", value).group())
                except Exception:
                    pass
            elif key == "JOB_TITLES":
                result["job_titles"] = [t.strip() for t in value.split(",") if t.strip()]
            elif key == "SKILLS":
                result["skills"] = [s.strip() for s in value.split(",") if s.strip()]
            elif key == "EDUCATION":
                result["education"] = [e.strip() for e in value.split(",") if e.strip()]
            elif key == "CERTIFICATIONS":
                result["certifications"] = [c.strip() for c in value.split(",") if c.strip()]
            elif key == "SUMMARY":
                result["summary"] = value
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, file_path: str | Path) -> ResumeProfile:
        """Parse a resume file and return a ResumeProfile."""
        fp = Path(file_path)
        if not fp.exists():
            logger.error("Resume not found: %s", fp)
            return ResumeProfile()

        raw_text = self._load_text(fp)
        if not raw_text.strip():
            logger.warning("Empty resume text from %s", fp)
            return ResumeProfile(raw_text=raw_text)

        regex_data = self._regex_extract(raw_text)
        llm_data = self._llm_extract(raw_text)

        # LLM result takes priority; regex fills gaps
        merged = {**regex_data, **llm_data}

        profile = ResumeProfile(
            name=merged.get("name", fp.stem),
            email=merged.get("email", ""),
            phone=merged.get("phone", ""),
            skills=merged.get("skills", []),
            experience_years=merged.get("experience_years", 0),
            job_titles=merged.get("job_titles", []),
            education=merged.get("education", []),
            certifications=merged.get("certifications", []),
            summary=merged.get("summary", ""),
            raw_text=raw_text,
        )
        logger.info(
            "Parsed resume: %s | %d skills | %d yrs exp",
            profile.name,
            len(profile.skills),
            profile.experience_years,
        )
        return profile

    def parse_job_title(self, job_title: str) -> ResumeProfile:
        """Create a minimal profile from just a job title (no resume file)."""
        prompt = (
            f"List the top 10 technical and soft skills typically required "
            f"for the role: '{job_title}'. Return a comma-separated list only."
        )
        try:
            raw = self._llm.invoke(prompt)
            skills = [s.strip() for s in raw.split(",") if s.strip()]
        except Exception:
            skills = []
        return ResumeProfile(
            name="User",
            job_titles=[job_title],
            skills=skills,
        )
