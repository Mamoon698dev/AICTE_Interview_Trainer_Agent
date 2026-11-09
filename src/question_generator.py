"""
question_generator.py – Generates tailored interview question sets using
IBM Granite via the RAG pipeline and structured prompt templates.

Supports:
  * Technical questions (role-specific)
  * Behavioural / STAR-format questions
  * HR / situational questions
  * Model answers + improvement tips
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Literal

from src.config import config
from src.watsonx_client import get_llm
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

QuestionType = Literal["technical", "behavioral", "hr", "situational"]
ExperienceLevel = Literal["fresher", "junior", "mid", "senior", "lead"]


@dataclass
class InterviewQuestion:
    question: str
    category: QuestionType
    difficulty: str
    model_answer: str
    improvement_tips: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)


@dataclass
class InterviewPlan:
    profile_name: str
    job_role: str
    experience_level: ExperienceLevel
    preparation_strategy: str
    technical_questions: list[InterviewQuestion] = field(default_factory=list)
    behavioral_questions: list[InterviewQuestion] = field(default_factory=list)
    hr_questions: list[InterviewQuestion] = field(default_factory=list)
    skill_gaps: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)


class QuestionGenerator:
    """Generates a full interview preparation plan using IBM Granite + RAG."""

    # ── Prompt templates ──────────────────────────────────────────────────────

    _TECH_PROMPT = textwrap.dedent("""
        You are a senior technical interviewer for the role of {job_role}.
        The candidate's name is {name} and experience level is {level}.

        Context from knowledge base:
        {context}

        Generate exactly {n} technical interview questions for this role and level.
        For each question provide:
        1. The question itself
        2. A comprehensive model answer (3-5 sentences)
        3. Two improvement tips

        Format each question as:
        Q: <question>
        A: <model answer>
        TIPS: <tip1> | <tip2>
        ---
    """).strip()

    _BEHAVIORAL_PROMPT = textwrap.dedent("""
        You are an HR expert preparing behavioural interview questions.
        Candidate: {name}, Role: {job_role}, Level: {level}.

        Context:
        {context}

        Generate exactly {n} STAR-format behavioural interview questions.
        Cover teamwork, conflict resolution, leadership, failure, and achievement.

        Format each question as:
        Q: <question>
        A: <example STAR answer>
        TIPS: <tip1> | <tip2>
        ---
    """).strip()

    _HR_PROMPT = textwrap.dedent("""
        You are an experienced HR interviewer.
        Candidate: {name}, Role: {job_role}, Level: {level}.

        Generate exactly {n} HR and situational interview questions covering:
        - Motivation and career goals
        - Salary expectations
        - Company culture fit
        - Strengths and weaknesses
        - Work-life balance

        Format each question as:
        Q: <question>
        A: <model answer>
        TIPS: <tip1> | <tip2>
        ---
    """).strip()

    _STRATEGY_PROMPT = textwrap.dedent("""
        You are a career coach. Create a concise, actionable interview preparation
        strategy for:
        - Name: {name}
        - Role: {job_role}
        - Experience: {level}
        - Skills extracted from resume: {skills}

        Provide:
        1. A 7-day preparation plan (bullet points)
        2. Top 3 skill gaps to address
        3. 3 recommended learning resources
        4. Key confidence-building tips

        Keep the response practical and motivating.
    """).strip()

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(self, rag: RAGPipeline | None = None) -> None:
        self._llm = get_llm()
        self._rag = rag or RAGPipeline()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _invoke(self, prompt: str) -> str:
        """Call the LLM and return the raw text."""
        try:
            return self._llm.invoke(prompt).strip()
        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc)
            return ""

    @staticmethod
    def _parse_qa_blocks(raw: str, category: QuestionType) -> list[InterviewQuestion]:
        """Parse the structured Q/A/TIPS blocks returned by the LLM."""
        questions: list[InterviewQuestion] = []
        blocks = [b.strip() for b in raw.split("---") if b.strip()]
        for block in blocks:
            lines = block.split("\n")
            q_text = a_text = ""
            tips: list[str] = []
            for line in lines:
                line = line.strip()
                if line.startswith("Q:"):
                    q_text = line[2:].strip()
                elif line.startswith("A:"):
                    a_text = line[2:].strip()
                elif line.startswith("TIPS:"):
                    tips = [t.strip() for t in line[5:].split("|") if t.strip()]
            if q_text:
                questions.append(
                    InterviewQuestion(
                        question=q_text,
                        category=category,
                        difficulty="medium",
                        model_answer=a_text,
                        improvement_tips=tips,
                    )
                )
        return questions

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_technical_questions(
        self,
        name: str,
        job_role: str,
        level: ExperienceLevel,
        n: int = 5,
    ) -> list[InterviewQuestion]:
        context = self._rag.get_context_string(
            f"{job_role} technical interview questions {level}"
        )
        prompt = self._TECH_PROMPT.format(
            name=name, job_role=job_role, level=level, n=n, context=context or "N/A"
        )
        raw = self._invoke(prompt)
        return self._parse_qa_blocks(raw, "technical")

    def generate_behavioral_questions(
        self,
        name: str,
        job_role: str,
        level: ExperienceLevel,
        n: int = 4,
    ) -> list[InterviewQuestion]:
        context = self._rag.get_context_string(
            f"behavioural interview questions {job_role}"
        )
        prompt = self._BEHAVIORAL_PROMPT.format(
            name=name, job_role=job_role, level=level, n=n, context=context or "N/A"
        )
        raw = self._invoke(prompt)
        return self._parse_qa_blocks(raw, "behavioral")

    def generate_hr_questions(
        self,
        name: str,
        job_role: str,
        level: ExperienceLevel,
        n: int = 3,
    ) -> list[InterviewQuestion]:
        prompt = self._HR_PROMPT.format(
            name=name, job_role=job_role, level=level, n=n
        )
        raw = self._invoke(prompt)
        return self._parse_qa_blocks(raw, "hr")

    def generate_preparation_strategy(
        self,
        name: str,
        job_role: str,
        level: ExperienceLevel,
        skills: list[str],
    ) -> str:
        prompt = self._STRATEGY_PROMPT.format(
            name=name,
            job_role=job_role,
            level=level,
            skills=", ".join(skills) if skills else "not specified",
        )
        return self._invoke(prompt)

    def generate_full_plan(
        self,
        name: str,
        job_role: str,
        level: ExperienceLevel,
        skills: list[str] | None = None,
    ) -> InterviewPlan:
        """Generate a complete interview preparation plan."""
        logger.info(
            "Generating full interview plan for %s | %s | %s", name, job_role, level
        )
        skills = skills or []

        plan = InterviewPlan(
            profile_name=name,
            job_role=job_role,
            experience_level=level,
            preparation_strategy="",
        )

        plan.technical_questions = self.generate_technical_questions(
            name, job_role, level, n=5
        )
        plan.behavioral_questions = self.generate_behavioral_questions(
            name, job_role, level, n=4
        )
        plan.hr_questions = self.generate_hr_questions(name, job_role, level, n=3)
        plan.preparation_strategy = self.generate_preparation_strategy(
            name, job_role, level, skills
        )

        return plan
