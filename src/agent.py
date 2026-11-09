"""
agent.py – InterviewTrainerAgent: top-level orchestrator that ties together
the RAG pipeline, resume parser, question generator, and skill assessor.

Usage
-----
    from src.agent import InterviewTrainerAgent

    agent = InterviewTrainerAgent()
    session = agent.start_session(
        name="Alice",
        job_role="Data Scientist",
        experience_level="mid",
        resume_path="alice_resume.pdf",   # optional
    )
    plan = agent.generate_plan(session)
    evaluation = agent.evaluate_answer(
        session,
        question=plan.technical_questions[0].question,
        model_answer=plan.technical_questions[0].model_answer,
        candidate_answer="Your candidate's answer here",
    )
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import config
from src.knowledge_base import seed_knowledge_base
from src.rag_pipeline import RAGPipeline
from src.resume_parser import ResumeParser, ResumeProfile
from src.question_generator import (
    QuestionGenerator,
    InterviewPlan,
    ExperienceLevel,
)
from src.skill_assessor import SkillAssessor, AnswerEvaluation, SkillGapReport

logger = logging.getLogger(__name__)


@dataclass
class AgentSession:
    """Holds per-user session state."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    job_role: str = ""
    experience_level: ExperienceLevel = "mid"
    profile: Optional[ResumeProfile] = None
    plan: Optional[InterviewPlan] = None
    evaluations: list[AnswerEvaluation] = field(default_factory=list)
    gap_report: Optional[SkillGapReport] = None


class InterviewTrainerAgent:
    """
    Top-level orchestrator for the Interview Trainer Agent.

    Lifecycle
    ---------
    1. `__init__`           – Initialise components, seed knowledge base, ingest docs.
    2. `start_session`      – Create a session and optionally parse the user's resume.
    3. `generate_plan`      – Produce a full interview preparation plan (via RAG + LLM).
    4. `evaluate_answer`    – Score and give feedback on a candidate's answer.
    5. `get_skill_gap`      – Analyse skill gaps for the session's role.
    6. `chat`               – Free-form Q&A using RAG.
    """

    def __init__(self) -> None:
        config.ensure_dirs()

        logger.info("Seeding built-in knowledge base …")
        seeded = seed_knowledge_base()
        if seeded:
            logger.info("%d knowledge files written.", seeded)

        logger.info("Initialising RAG pipeline …")
        self._rag = RAGPipeline()
        self._rag.ingest_knowledge_base()

        self._parser = ResumeParser()
        self._generator = QuestionGenerator(rag=self._rag)
        self._assessor = SkillAssessor()

        logger.info("Interview Trainer Agent ready.")

    # ── Session management ────────────────────────────────────────────────────

    def start_session(
        self,
        name: str,
        job_role: str,
        experience_level: ExperienceLevel = "mid",
        resume_path: Optional[str | Path] = None,
    ) -> AgentSession:
        """
        Create a new agent session.

        Parameters
        ----------
        name              : Candidate's name.
        job_role          : Target job title (e.g., "Software Engineer").
        experience_level  : 'fresher' | 'junior' | 'mid' | 'senior' | 'lead'
        resume_path       : Path to resume file (PDF/DOCX/TXT). Optional.
        """
        session = AgentSession(
            name=name,
            job_role=job_role,
            experience_level=experience_level,
        )

        if resume_path:
            fp = Path(resume_path)
            if fp.exists():
                logger.info("Parsing resume: %s", fp)
                session.profile = self._parser.parse(fp)
                # Update session with parsed info
                if session.profile.name and session.profile.name != "User":
                    session.name = session.profile.name
                if session.profile.experience_level != "fresher" or session.profile.experience_years > 0:
                    session.experience_level = session.profile.experience_level  # type: ignore[assignment]
                # Ingest into resume-specific collection for personalised retrieval
                self._rag.ingest_resume(fp)
            else:
                logger.warning("Resume file not found: %s", fp)
                # Fall back to job-title based profile
                session.profile = self._parser.parse_job_title(job_role)
        else:
            session.profile = self._parser.parse_job_title(job_role)

        logger.info(
            "Session started: %s | %s | %s | skills=%d",
            session.session_id,
            session.name,
            session.job_role,
            len(session.profile.skills) if session.profile else 0,
        )
        return session

    # ── Plan generation ───────────────────────────────────────────────────────

    def generate_plan(self, session: AgentSession) -> InterviewPlan:
        """Generate a complete, tailored interview preparation plan."""
        skills = session.profile.skills if session.profile else []
        plan = self._generator.generate_full_plan(
            name=session.name,
            job_role=session.job_role,
            level=session.experience_level,
            skills=skills,
        )
        session.plan = plan

        # Also generate skill gap report
        if skills:
            session.gap_report = self._assessor.analyse_skill_gap(
                job_role=session.job_role,
                candidate_skills=skills,
                experience_level=session.experience_level,
            )
        return plan

    # ── Answer evaluation ─────────────────────────────────────────────────────

    def evaluate_answer(
        self,
        session: AgentSession,
        question: str,
        model_answer: str,
        candidate_answer: str,
    ) -> AnswerEvaluation:
        """Evaluate a candidate's answer and store it in the session."""
        evaluation = self._assessor.evaluate_answer(
            question=question,
            candidate_answer=candidate_answer,
            model_answer=model_answer,
        )
        session.evaluations.append(evaluation)
        return evaluation

    # ── Skill gap analysis ────────────────────────────────────────────────────

    def get_skill_gap(self, session: AgentSession) -> SkillGapReport:
        """Analyse and return the skill gap report for the session."""
        skills = session.profile.skills if session.profile else []
        report = self._assessor.analyse_skill_gap(
            job_role=session.job_role,
            candidate_skills=skills,
            experience_level=session.experience_level,
        )
        session.gap_report = report
        return report

    # ── Free-form chat ────────────────────────────────────────────────────────

    def chat(self, session: AgentSession, user_message: str) -> str:
        """
        Answer a free-form question using RAG over the knowledge base.

        Falls back to direct LLM generation if the knowledge base is empty.
        """
        try:
            chain = self._rag.build_qa_chain(collection="kb")
            result = chain.invoke(user_message)
            return result if isinstance(result, str) else "I could not find an answer. Please try rephrasing."
        except RuntimeError:
            # KB empty – use LLM directly
            from src.watsonx_client import get_llm
            llm = get_llm()
            prompt = (
                f"You are an expert interview coach helping {session.name} "
                f"prepare for a {session.job_role} interview "
                f"(experience level: {session.experience_level}).\n\n"
                f"User: {user_message}\n\nCoach:"
            )
            return llm.invoke(prompt)

    # ── Session summary ───────────────────────────────────────────────────────

    def session_summary(self, session: AgentSession) -> dict:
        """Return a dictionary summarising the session state."""
        evals = session.evaluations
        avg_score = (
            sum(e.score for e in evals) / len(evals) if evals else None
        )
        return {
            "session_id": session.session_id,
            "name": session.name,
            "job_role": session.job_role,
            "experience_level": session.experience_level,
            "skills_identified": len(session.profile.skills) if session.profile else 0,
            "questions_generated": (
                len(session.plan.technical_questions)
                + len(session.plan.behavioral_questions)
                + len(session.plan.hr_questions)
            ) if session.plan else 0,
            "answers_evaluated": len(evals),
            "average_score": round(avg_score, 2) if avg_score is not None else "N/A",
            "skill_gaps": session.gap_report.gap_skills if session.gap_report else [],
        }
