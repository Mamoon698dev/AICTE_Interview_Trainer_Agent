"""
knowledge_base.py – Seeds the RAG vector store with curated interview
knowledge: question banks, industry expectations, behavioural scenarios,
and HR guidelines.

This module ships with an in-code knowledge base so the agent works
out-of-the-box without external files.  Additional documents can be
added by placing PDFs / DOCX files in data/knowledge_base/.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import config

logger = logging.getLogger(__name__)

# ── Built-in interview knowledge ──────────────────────────────────────────────
# Each entry is (filename, content).  Written to KB_DIR on first run so that
# the RAG pipeline can ingest them.

BUILTIN_KNOWLEDGE: list[tuple[str, str]] = [
    # ── Technical Q&A ────────────────────────────────────────────────────────
    (
        "tech_qa_software_engineering.txt",
        """
TOPIC: Software Engineering Interview Questions

Q: What is the difference between a stack and a queue?
A: A stack is LIFO (Last In, First Out) – items are pushed and popped from the top.
   A queue is FIFO (First In, First Out) – items are enqueued at the rear and dequeued from the front.
   Use a stack for undo operations, call stacks, and DFS. Use a queue for BFS, scheduling, and buffers.

Q: Explain the SOLID principles.
A: Single Responsibility – a class should have only one reason to change.
   Open/Closed – open for extension, closed for modification.
   Liskov Substitution – subclasses should be substitutable for their base class.
   Interface Segregation – prefer many specific interfaces over one general-purpose interface.
   Dependency Inversion – depend on abstractions, not concretions.

Q: What is the time complexity of binary search?
A: O(log n). Binary search halves the search space at each step, making it very efficient for sorted arrays.

Q: What is a RESTful API?
A: REST (Representational State Transfer) is an architectural style for distributed hypermedia systems.
   Key principles: stateless, uniform interface, client-server separation, cacheable, layered system.
   HTTP methods map to CRUD: GET (read), POST (create), PUT/PATCH (update), DELETE (remove).

Q: What is the difference between SQL JOIN types?
A: INNER JOIN returns rows where both tables have matching values.
   LEFT JOIN returns all rows from the left table plus matched rows from the right.
   RIGHT JOIN returns all rows from the right table plus matched rows from the left.
   FULL OUTER JOIN returns all rows from both tables, with NULLs where there is no match.

Q: Explain microservices architecture.
A: Microservices decompose an application into small, independently deployable services.
   Benefits: independent scaling, technology diversity, fault isolation, faster deployments.
   Challenges: distributed systems complexity, inter-service communication, data consistency.

Q: What is Docker and why is it used?
A: Docker is a containerisation platform that packages applications with all their dependencies.
   Benefits: consistent environments, faster deployment, efficient resource utilisation.
   Containers are lightweight compared to VMs as they share the host OS kernel.
""",
    ),
    # ── Data Science Q&A ─────────────────────────────────────────────────────
    (
        "tech_qa_data_science.txt",
        """
TOPIC: Data Science Interview Questions

Q: What is the bias-variance trade-off?
A: Bias is error from overly simplistic assumptions (underfitting).
   Variance is error from sensitivity to fluctuations in training data (overfitting).
   The goal is to find the right model complexity that minimises total error.

Q: Explain precision vs recall.
A: Precision = TP / (TP + FP) – of all predicted positives, how many are truly positive?
   Recall = TP / (TP + FN) – of all actual positives, how many did we capture?
   F1 score is the harmonic mean of both. Choose precision when false positives are costly,
   recall when false negatives are costly.

Q: What is cross-validation?
A: A technique to evaluate model generalisation by splitting data into k folds.
   The model is trained on k-1 folds and tested on the remaining fold, repeated k times.
   k-fold CV provides a robust performance estimate and reduces overfitting.

Q: How do you handle missing data?
A: Options: deletion (listwise/pairwise), mean/median/mode imputation,
   model-based imputation (KNN, regression), multiple imputation, or using algorithms
   that handle missing values natively (e.g., XGBoost).

Q: What is regularisation and why is it important?
A: Regularisation adds a penalty term to the loss function to prevent overfitting.
   L1 (Lasso) encourages sparsity; L2 (Ridge) shrinks coefficients towards zero.
   ElasticNet combines both. Critical when the model has many features relative to samples.
""",
    ),
    # ── Behavioural Q&A ──────────────────────────────────────────────────────
    (
        "behavioral_questions.txt",
        """
TOPIC: Behavioural Interview Questions & STAR Answers

Q: Tell me about a time you handled a conflict with a team member.
A (STAR): Situation – During a sprint, a colleague and I disagreed on the API design approach.
   Task – We needed to resolve the conflict quickly to avoid blocking the team.
   Action – I initiated a one-on-one, listened to their concerns, and we jointly reviewed
   both designs against the requirements. We agreed on a hybrid approach.
   Result – The feature was delivered on time and the relationship improved.

Q: Describe a challenging project and how you overcame obstacles.
A (STAR): Situation – I was tasked with migrating a legacy monolith to microservices in 3 months.
   Task – The codebase had no tests and limited documentation.
   Action – I created a strangler fig migration plan, added a test harness, and held weekly
   knowledge-sharing sessions with the team.
   Result – Delivered 80% migration on schedule; remaining 20% completed the following sprint.

Q: Give an example of when you showed leadership.
A (STAR): Situation – Our tech lead left mid-project and the team lacked direction.
   Task – I stepped up to coordinate without formal authority.
   Action – Organised daily stand-ups, maintained the backlog, and escalated blockers to management.
   Result – The project shipped successfully and I was offered a team lead role.

Q: Tell me about a failure and what you learned.
A (STAR): Situation – I deployed a database migration without proper rollback planning.
   Task – When the migration failed in production, I needed to recover quickly.
   Action – Used backups to restore the database and worked with the DBA to fix the script.
   Result – Service was restored within 2 hours. I now enforce mandatory rollback plans
   and staging deployments for all migrations.

Q: How do you prioritise when you have multiple deadlines?
A: I use a combination of urgency/importance matrix (Eisenhower Matrix) and direct communication
   with stakeholders. I break tasks into milestones, identify dependencies, and flag risks early.
   I believe in transparency – I'd rather renegotiate a deadline proactively than miss it silently.
""",
    ),
    # ── HR Guidelines ────────────────────────────────────────────────────────
    (
        "hr_guidelines.txt",
        """
TOPIC: HR Interview Guidelines & Best Practices

SALARY NEGOTIATION:
- Research market rates using Glassdoor, LinkedIn Salary, and Levels.fyi before the interview.
- Express enthusiasm for the role before discussing compensation.
- Give a range based on market data, with your target at the lower end of the range.
- Example: "Based on my research and experience, I'm targeting £65,000–£75,000."

STRENGTHS AND WEAKNESSES:
- Strengths: Choose 3 that are directly relevant to the role with concrete examples.
- Weaknesses: Choose a genuine weakness that you are actively working to improve.
- Avoid clichés like "I'm a perfectionist" – it comes across as disingenuous.

WHY DO YOU WANT THIS ROLE?
- Research the company thoroughly (mission, culture, recent news, products).
- Connect the role to your career goals and values.
- Be specific about what excites you about this particular opportunity.

WHERE DO YOU SEE YOURSELF IN 5 YEARS?
- Align your answer with the career path available at the company.
- Show ambition balanced with realism and loyalty.
- Focus on growth in expertise, not just titles.

CULTURE FIT QUESTIONS:
- Be authentic – companies value genuine alignment over rehearsed answers.
- Relate your values to the company's stated values and mission.
- Use specific examples from past experience to illustrate cultural alignment.

CLOSING THE INTERVIEW:
- Always have 2-3 thoughtful questions ready for the interviewer.
- Good questions: team structure, success metrics, biggest challenges, next steps.
- End by reaffirming your interest and thanking the interviewer.
""",
    ),
    # ── Industry Expectations ────────────────────────────────────────────────
    (
        "industry_expectations.txt",
        """
TOPIC: Industry Expectations by Experience Level

FRESHER (0–1 year):
- Strong foundations in computer science fundamentals.
- Proficiency in at least one programming language.
- Familiarity with version control (Git).
- Good communication skills and eagerness to learn.
- Basic understanding of software development lifecycle.

JUNIOR (1–3 years):
- Ability to work independently on well-defined tasks.
- Code review participation and understanding of code quality.
- Experience with testing (unit and integration tests).
- Understanding of CI/CD concepts.
- Collaboration skills within Agile teams.

MID-LEVEL (3–6 years):
- System design capabilities for medium-scale systems.
- Mentoring junior developers.
- Strong debugging and performance optimisation skills.
- Cross-functional collaboration (Product, Design, QA).
- Ownership of features end-to-end.

SENIOR (6–10 years):
- Technical leadership and architecture decisions.
- Driving engineering best practices.
- Deep expertise in multiple areas.
- Stakeholder management and business impact awareness.
- Hiring and growing the engineering team.

LEAD / STAFF (10+ years):
- Org-wide technical strategy.
- Influencing without authority across teams.
- Long-horizon planning and roadmap alignment.
- Deep domain expertise and thought leadership.
- Mentoring senior engineers.
""",
    ),
]


def seed_knowledge_base() -> int:
    """
    Write built-in knowledge files to the KB directory.
    Returns the number of files written.
    """
    kb_dir = Path(config.KB_DIR)
    kb_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for filename, content in BUILTIN_KNOWLEDGE:
        dest = kb_dir / filename
        if not dest.exists():
            dest.write_text(content.strip(), encoding="utf-8")
            logger.info("Seeded knowledge file: %s", filename)
            written += 1
        else:
            logger.debug("Knowledge file already exists: %s", filename)
    return written
