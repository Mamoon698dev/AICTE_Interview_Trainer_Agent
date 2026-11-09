"""
app.py – Streamlit web UI for the Interview Trainer Agent.

Run: streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="IBM Interview Trainer Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.WARNING)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2328;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #57606a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ibm-badge {
        background: #0f62fe;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .score-high { color: #2da44e; font-weight: 700; }
    .score-mid  { color: #d29922; font-weight: 700; }
    .score-low  { color: #cf222e; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "agent": None,
        "session": None,
        "plan": None,
        "gap_report": None,
        "chat_history": [],
        "evaluations": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_resource(show_spinner="Initialising IBM Interview Trainer Agent…")
def _load_agent():
    from src.agent import InterviewTrainerAgent
    return InterviewTrainerAgent()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict | None:
    st.sidebar.markdown(
        '<div style="text-align:center">'
        '<span class="ibm-badge">IBM Granite · Watsonx.ai</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.title("🎯 Interview Trainer")
    st.sidebar.markdown("---")

    from src.config import config
    missing = config.validate()
    if missing:
        st.sidebar.warning(
            f"⚠️ Missing env vars: `{', '.join(missing)}`\n\n"
            "Edit **env.example** → rename to **.env** with your IBM Cloud credentials."
        )

    st.sidebar.subheader("Candidate Profile")
    name = st.sidebar.text_input("Your Name", value="Alex Johnson")
    job_role = st.sidebar.text_input("Target Role", value="Software Engineer")
    level = st.sidebar.selectbox(
        "Experience Level",
        ["fresher", "junior", "mid", "senior", "lead"],
        index=2,
    )
    resume_file = st.sidebar.file_uploader(
        "Upload Resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"]
    )

    if st.sidebar.button("🚀 Start Session", type="primary", use_container_width=True):
        return {
            "name": name,
            "job_role": job_role,
            "level": level,
            "resume_file": resume_file,
        }
    return None


# ── Tab: Generate Plan ────────────────────────────────────────────────────────

def tab_generate_plan() -> None:
    st.subheader("📋 Personalised Interview Preparation Plan")
    if st.session_state.session is None:
        st.info("👈 Complete the profile in the sidebar and click **Start Session** first.")
        return

    if st.button("⚡ Generate My Interview Plan", type="primary"):
        with st.spinner("IBM Granite is crafting your plan…"):
            plan = st.session_state.agent.generate_plan(st.session_state.session)
            st.session_state.plan = plan

    plan = st.session_state.plan
    if plan is None:
        return

    # Preparation strategy
    if plan.preparation_strategy:
        with st.expander("🗓️ Preparation Strategy & 7-Day Plan", expanded=True):
            st.markdown(plan.preparation_strategy)

    # Technical questions
    if plan.technical_questions:
        st.markdown("### 🔧 Technical Questions")
        for i, q in enumerate(plan.technical_questions, 1):
            with st.expander(f"Q{i}: {q.question[:80]}…" if len(q.question) > 80 else f"Q{i}: {q.question}"):
                st.markdown(f"**Question:** {q.question}")
                st.success(f"**Model Answer:** {q.model_answer}")
                if q.improvement_tips:
                    st.info("**Tips:** " + " | ".join(q.improvement_tips))

    # Behavioural
    if plan.behavioral_questions:
        st.markdown("### 🧠 Behavioural Questions (STAR Format)")
        for i, q in enumerate(plan.behavioral_questions, 1):
            with st.expander(f"B{i}: {q.question[:80]}…" if len(q.question) > 80 else f"B{i}: {q.question}"):
                st.markdown(f"**Question:** {q.question}")
                st.success(f"**Model STAR Answer:** {q.model_answer}")
                if q.improvement_tips:
                    st.info("**Tips:** " + " | ".join(q.improvement_tips))

    # HR
    if plan.hr_questions:
        st.markdown("### 👔 HR / Situational Questions")
        for i, q in enumerate(plan.hr_questions, 1):
            with st.expander(f"HR{i}: {q.question[:80]}…" if len(q.question) > 80 else f"HR{i}: {q.question}"):
                st.markdown(f"**Question:** {q.question}")
                st.success(f"**Model Answer:** {q.model_answer}")
                if q.improvement_tips:
                    st.info("**Tips:** " + " | ".join(q.improvement_tips))


# ── Tab: Mock Interview ───────────────────────────────────────────────────────

def tab_mock_interview() -> None:
    st.subheader("🎤 Mock Interview Practice")
    if st.session_state.session is None:
        st.info("👈 Start a session first.")
        return
    if st.session_state.plan is None:
        st.info("Generate your interview plan first.")
        return

    plan = st.session_state.plan
    all_questions = (
        plan.technical_questions
        + plan.behavioral_questions
        + plan.hr_questions
    )
    if not all_questions:
        st.warning("No questions available. Generate a plan first.")
        return

    q_index = st.selectbox(
        "Select a question to practice:",
        range(len(all_questions)),
        format_func=lambda i: f"Q{i+1}: {all_questions[i].question[:70]}…"
        if len(all_questions[i].question) > 70
        else f"Q{i+1}: {all_questions[i].question}",
    )
    selected_q = all_questions[q_index]
    st.markdown(f"**Question:** {selected_q.question}")
    candidate_answer = st.text_area(
        "Your Answer",
        height=150,
        placeholder="Type your answer here as you would in a real interview…",
    )

    if st.button("📊 Evaluate My Answer", type="primary") and candidate_answer.strip():
        with st.spinner("IBM Granite is evaluating your answer…"):
            evaluation = st.session_state.agent.evaluate_answer(
                session=st.session_state.session,
                question=selected_q.question,
                model_answer=selected_q.model_answer,
                candidate_answer=candidate_answer,
            )
            st.session_state.evaluations.append(evaluation)

        score = evaluation.score
        css_cls = (
            "score-high" if score >= 7
            else "score-mid" if score >= 5
            else "score-low"
        )
        st.markdown(
            f'<p>Score: <span class="{css_cls}">{score}/10</span></p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.success(
                "**✅ Strengths:**\n"
                + ("\n".join(f"- {s}" for s in evaluation.strengths) or "N/A")
            )
        with col2:
            st.error(
                "**❌ Areas to Improve:**\n"
                + ("\n".join(f"- {w}" for w in evaluation.weaknesses) or "N/A")
            )
        st.info(f"**💡 Improved Answer:** {evaluation.improved_answer}")
        st.warning(f"**🎯 Confidence Tip:** {evaluation.confidence_tip}")


# ── Tab: Skill Gap ────────────────────────────────────────────────────────────

def tab_skill_gap() -> None:
    st.subheader("📊 Skill Gap Analysis")
    if st.session_state.session is None:
        st.info("👈 Start a session first.")
        return

    if st.button("🔍 Analyse My Skill Gaps", type="primary"):
        with st.spinner("Analysing skill gaps…"):
            st.session_state.gap_report = st.session_state.agent.get_skill_gap(
                st.session_state.session
            )

    report = st.session_state.gap_report
    if report is None:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Your Strengths")
        if report.strength_skills:
            for s in report.strength_skills:
                st.success(s)
        else:
            st.info("No matching skills found in taxonomy.")
    with col2:
        st.markdown("### ⚠️ Skills to Develop")
        if report.gap_skills:
            for g in report.gap_skills:
                st.error(g)
        else:
            st.success("You cover all required skills! 🎉")

    if report.soft_skill_tips:
        st.markdown("### 💡 Improvement Tips")
        for tip in report.soft_skill_tips:
            st.info(tip)

    if report.learning_resources:
        st.markdown("### 📚 Recommended Learning Resources")
        for r in report.learning_resources:
            st.markdown(f"- {r}")


# ── Tab: Chat ─────────────────────────────────────────────────────────────────

def tab_chat() -> None:
    st.subheader("💬 Chat with Your Interview Coach")
    if st.session_state.session is None:
        st.info("👈 Start a session first.")
        return

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask your interview coach anything…"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = st.session_state.agent.chat(
                    st.session_state.session, user_input
                )
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# ── Tab: Session Summary ──────────────────────────────────────────────────────

def tab_summary() -> None:
    st.subheader("📈 Session Summary")
    if st.session_state.session is None:
        st.info("👈 Start a session first.")
        return

    summary = st.session_state.agent.session_summary(st.session_state.session)
    col1, col2, col3 = st.columns(3)
    col1.metric("Questions Generated", summary["questions_generated"])
    col2.metric("Answers Evaluated", summary["answers_evaluated"])
    col3.metric(
        "Average Score",
        f"{summary['average_score']}/10"
        if isinstance(summary["average_score"], float)
        else summary["average_score"],
    )

    st.markdown("---")
    st.markdown(f"**Candidate:** {summary['name']}")
    st.markdown(f"**Role:** {summary['job_role']} ({summary['experience_level']})")
    st.markdown(f"**Skills Identified:** {summary['skills_identified']}")

    if summary["skill_gaps"]:
        st.markdown(
            "**Skill Gaps:** "
            + ", ".join(f"`{g}`" for g in summary["skill_gaps"])
        )

    if st.session_state.evaluations:
        st.markdown("### Recent Evaluations")
        for ev in st.session_state.evaluations[-5:]:
            score = ev.score
            colour = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
            st.markdown(
                f"{colour} **{ev.score}/10** – {ev.question[:80]}…"
                if len(ev.question) > 80
                else f"{colour} **{ev.score}/10** – {ev.question}"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()

    # Header
    st.markdown(
        '<h1 class="main-header">🎯 IBM Interview Trainer Agent</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Powered by <span class="ibm-badge">IBM Granite</span> '
        '&nbsp; <span class="ibm-badge">Watsonx.ai</span> &nbsp; '
        '<span class="ibm-badge">RAG</span></p>',
        unsafe_allow_html=True,
    )

    # Sidebar – collect profile & start session
    profile = render_sidebar()
    if profile is not None:
        agent = _load_agent()
        st.session_state.agent = agent

        resume_path = None
        if profile["resume_file"] is not None:
            # Save uploaded file to a temp path
            suffix = Path(profile["resume_file"].name).suffix
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=str(Path("data/resumes"))
            ) as tmp:
                tmp.write(profile["resume_file"].read())
                resume_path = tmp.name

        with st.spinner("Starting your session…"):
            st.session_state.session = agent.start_session(
                name=profile["name"],
                job_role=profile["job_role"],
                experience_level=profile["level"],
                resume_path=resume_path,
            )
            # Reset plan etc. for new session
            st.session_state.plan = None
            st.session_state.gap_report = None
            st.session_state.chat_history = []
            st.session_state.evaluations = []

        st.sidebar.success(
            f"✅ Session started!\n"
            f"Skills: **{len(st.session_state.session.profile.skills) if st.session_state.session.profile else 0}** identified"
        )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Interview Plan", "🎤 Mock Interview", "📊 Skill Gap", "💬 Chat", "📈 Summary"]
    )
    with tab1:
        tab_generate_plan()
    with tab2:
        tab_mock_interview()
    with tab3:
        tab_skill_gap()
    with tab4:
        tab_chat()
    with tab5:
        tab_summary()


if __name__ == "__main__":
    main()
