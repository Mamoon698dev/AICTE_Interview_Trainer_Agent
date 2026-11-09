"""
skill_assessor.py – Evaluates candidate responses and generates improvement
feedback using IBM Granite.

Provides:
  * Score a candidate's answer against a model answer (0-10)
  * Identify skill gaps based on job role vs resume skills
  * Generate personalised soft-skill improvement tips
  * Conduct a mock interview round with real-time feedback
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from src.watsonx_client import get_llm

logger = logging.getLogger(__name__)

# Skill taxonomy used for gap analysis
TECHNICAL_SKILL_DOMAINS = {
    "software_engineer": [
        "Python", "Java", "JavaScript", "TypeScript", "Data Structures",
        "Algorithms", "System Design", "REST APIs", "SQL", "Git",
        "Docker", "Kubernetes", "CI/CD", "Cloud (AWS/Azure/GCP/IBM Cloud)",
    ],
    "data_scientist": [
        "Python", "R", "Machine Learning", "Deep Learning", "NLP",
        "Statistics", "SQL", "Pandas", "NumPy", "TensorFlow", "PyTorch",
        "Data Visualisation", "Feature Engineering", "Model Deployment",
    ],
    "devops_engineer": [
        "Linux", "Bash/Shell", "Docker", "Kubernetes", "Terraform",
        "Ansible", "CI/CD (Jenkins/GitHub Actions)", "Monitoring (Prometheus/Grafana)",
        "Networking", "Security best practices",
    ],
    "product_manager": [
        "Roadmapping", "User Stories", "Agile/Scrum", "OKRs",
        "A/B Testing", "Market Research", "Stakeholder Management",
        "SQL (basic)", "Data-driven decision making", "Prioritisation frameworks",
    ],
    "data_engineer": [
        "Python", "SQL", "Spark", "Kafka", "Airflow", "ETL pipelines",
        "Data Warehousing", "dbt", "Cloud data platforms", "NoSQL",
    ],
}

SOFT_SKILLS = [
    "Communication", "Teamwork", "Leadership", "Problem-solving",
    "Adaptability", "Time management", "Critical thinking",
    "Conflict resolution", "Emotional intelligence", "Presentation skills",
]


@dataclass
class AnswerEvaluation:
    question: str
    candidate_answer: str
    model_answer: str
    score: float          # 0 – 10
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    improved_answer: str = ""
    confidence_tip: str = ""


@dataclass
class SkillGapReport:
    job_role: str
    candidate_skills: list[str]
    required_skills: list[str]
    gap_skills: list[str]
    strength_skills: list[str]
    soft_skill_tips: list[str] = field(default_factory=list)
    learning_resources: list[str] = field(default_factory=list)


class SkillAssessor:
    """Assesses candidate answers and skill gaps using IBM Granite."""

    # ── Prompt templates ──────────────────────────────────────────────────────

    _EVAL_PROMPT = textwrap.dedent("""
        You are a strict but fair technical interviewer.
        Evaluate the candidate's answer to the interview question.

        Question: {question}

        Model Answer: {model_answer}

        Candidate's Answer: {candidate_answer}

        Provide your evaluation in EXACTLY this format:
        SCORE: <number from 0 to 10>
        STRENGTHS: <comma-separated list of what was done well>
        WEAKNESSES: <comma-separated list of what was missing or wrong>
        IMPROVED_ANSWER: <a polished version of the candidate's answer>
        CONFIDENCE_TIP: <one actionable tip to improve delivery>
    """).strip()

    _SOFT_SKILL_PROMPT = textwrap.dedent("""
        You are an executive coach specialising in interview preparation.
        The candidate is applying for {job_role} at experience level {level}.

        Their identified soft-skill gaps: {gaps}

        Provide exactly 3 actionable improvement tips for each gap skill.
        Format:
        SKILL: <skill name>
        TIP1: <tip>
        TIP2: <tip>
        TIP3: <tip>
        ---
    """).strip()

    _RESOURCES_PROMPT = textwrap.dedent("""
        Suggest 5 specific learning resources (books, courses, or platforms)
        for a {level} candidate targeting the role of {job_role} who needs
        to improve in: {gaps}.

        Format each resource as:
        - <Resource Name> – <why it helps> (<URL or platform>)
    """).strip()

    def __init__(self) -> None:
        self._llm = get_llm()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _invoke(self, prompt: str) -> str:
        try:
            return self._llm.invoke(prompt).strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return ""

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_answer(
        self,
        question: str,
        candidate_answer: str,
        model_answer: str,
    ) -> AnswerEvaluation:
        """Score and provide feedback on a candidate's answer."""
        prompt = self._EVAL_PROMPT.format(
            question=question,
            model_answer=model_answer,
            candidate_answer=candidate_answer,
        )
        raw = self._invoke(prompt)

        eval_data: dict = {
            "score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "improved_answer": "",
            "confidence_tip": "",
        }
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    import re
                    m = re.search(r"[\d.]+", line)
                    eval_data["score"] = float(m.group()) if m else 0.0
                except Exception:
                    pass
            elif line.startswith("STRENGTHS:"):
                eval_data["strengths"] = [
                    s.strip() for s in line[10:].split(",") if s.strip()
                ]
            elif line.startswith("WEAKNESSES:"):
                eval_data["weaknesses"] = [
                    w.strip() for w in line[11:].split(",") if w.strip()
                ]
            elif line.startswith("IMPROVED_ANSWER:"):
                eval_data["improved_answer"] = line[16:].strip()
            elif line.startswith("CONFIDENCE_TIP:"):
                eval_data["confidence_tip"] = line[15:].strip()

        return AnswerEvaluation(
            question=question,
            candidate_answer=candidate_answer,
            model_answer=model_answer,
            **eval_data,
        )

    def analyse_skill_gap(
        self,
        job_role: str,
        candidate_skills: list[str],
        experience_level: str = "mid",
    ) -> SkillGapReport:
        """Compare candidate skills against role requirements."""
        # Normalise job role to known taxonomy key
        role_key = self._match_role_key(job_role)
        required = TECHNICAL_SKILL_DOMAINS.get(role_key, [])

        candidate_lower = {s.lower() for s in candidate_skills}
        gap = [r for r in required if r.lower() not in candidate_lower]
        strengths = [r for r in required if r.lower() in candidate_lower]

        # Get soft-skill tips from LLM
        soft_tips: list[str] = []
        resources: list[str] = []
        if gap:
            soft_raw = self._invoke(
                self._SOFT_SKILL_PROMPT.format(
                    job_role=job_role,
                    level=experience_level,
                    gaps=", ".join(gap[:5]),
                )
            )
            soft_tips = [
                l.strip()
                for l in soft_raw.splitlines()
                if l.strip() and not l.strip().startswith(("SKILL:", "---"))
            ]

            res_raw = self._invoke(
                self._RESOURCES_PROMPT.format(
                    level=experience_level,
                    job_role=job_role,
                    gaps=", ".join(gap[:5]),
                )
            )
            resources = [
                l.strip().lstrip("- ")
                for l in res_raw.splitlines()
                if l.strip().startswith("-")
            ]

        return SkillGapReport(
            job_role=job_role,
            candidate_skills=candidate_skills,
            required_skills=required,
            gap_skills=gap,
            strength_skills=strengths,
            soft_skill_tips=soft_tips,
            learning_resources=resources,
        )

    def assess_soft_skills(
        self,
        answers: dict[str, str],
        job_role: str,
    ) -> dict[str, str]:
        """
        Assess soft skills from a dictionary of {soft_skill_question: answer}.
        Returns {skill: feedback} mapping.
        """
        feedback: dict[str, str] = {}
        for skill, answer in answers.items():
            prompt = (
                f"Assess this interview answer for the soft skill '{skill}' "
                f"for the role of {job_role}.\n\n"
                f"Answer: {answer}\n\n"
                f"Provide concise feedback (2-3 sentences) and a score out of 10."
            )
            feedback[skill] = self._invoke(prompt)
        return feedback

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _match_role_key(job_role: str) -> str:
        """Map a free-form job title to a taxonomy key."""
        role_lower = job_role.lower()
        for key in TECHNICAL_SKILL_DOMAINS:
            if any(part in role_lower for part in key.split("_")):
                return key
        return "software_engineer"  # default
