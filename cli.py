"""
cli.py – Rich interactive command-line interface for the Interview Trainer Agent.

Run: python cli.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint

console = Console()

BANNER = """
[bold blue]
╔══════════════════════════════════════════════════════════╗
║        🎯  IBM Interview Trainer Agent                   ║
║        Powered by IBM Granite & RAG (Watsonx.ai)         ║
╚══════════════════════════════════════════════════════════╝
[/bold blue]
"""

EXPERIENCE_LEVELS = ["fresher", "junior", "mid", "senior", "lead"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,  # suppress verbose library logs in CLI
        format="%(levelname)s | %(name)s | %(message)s",
    )


def print_banner() -> None:
    console.print(BANNER)


def collect_user_profile() -> dict:
    """Interactively collect candidate profile information."""
    console.rule("[bold cyan]Candidate Profile Setup")

    name = Prompt.ask("[cyan]Your name")
    job_role = Prompt.ask("[cyan]Target job role", default="Software Engineer")

    console.print("\n[dim]Experience levels: fresher, junior, mid, senior, lead[/dim]")
    level = Prompt.ask(
        "[cyan]Experience level",
        default="mid",
        choices=EXPERIENCE_LEVELS,
    )

    resume_path = None
    if Confirm.ask("\n[cyan]Do you have a resume to upload? (PDF/DOCX/TXT)"):
        while True:
            path_str = Prompt.ask("[cyan]Resume file path")
            p = Path(path_str)
            if p.exists():
                resume_path = str(p)
                break
            console.print(f"[red]File not found: {path_str}. Please try again.[/red]")

    return {"name": name, "job_role": job_role, "level": level, "resume_path": resume_path}


def display_plan(plan) -> None:
    """Render the interview plan to the terminal."""
    console.rule("[bold green]Your Personalised Interview Preparation Plan")

    # ── Preparation strategy ──────────────────────────────────────────────────
    if plan.preparation_strategy:
        console.print(
            Panel(
                Markdown(plan.preparation_strategy),
                title="[bold yellow]📋 Preparation Strategy",
                border_style="yellow",
            )
        )

    # ── Technical questions ───────────────────────────────────────────────────
    if plan.technical_questions:
        console.print("\n[bold blue]🔧 Technical Questions[/bold blue]")
        for i, q in enumerate(plan.technical_questions, 1):
            console.print(
                Panel(
                    f"[bold]Q{i}:[/bold] {q.question}\n\n"
                    f"[green]Model Answer:[/green] {q.model_answer}\n\n"
                    f"[yellow]Tips:[/yellow] {' | '.join(q.improvement_tips)}",
                    border_style="blue",
                )
            )

    # ── Behavioural questions ─────────────────────────────────────────────────
    if plan.behavioral_questions:
        console.print("\n[bold magenta]🧠 Behavioural Questions (STAR)[/bold magenta]")
        for i, q in enumerate(plan.behavioral_questions, 1):
            console.print(
                Panel(
                    f"[bold]Q{i}:[/bold] {q.question}\n\n"
                    f"[green]Model Answer:[/green] {q.model_answer}\n\n"
                    f"[yellow]Tips:[/yellow] {' | '.join(q.improvement_tips)}",
                    border_style="magenta",
                )
            )

    # ── HR questions ─────────────────────────────────────────────────────────
    if plan.hr_questions:
        console.print("\n[bold cyan]👔 HR / Situational Questions[/bold cyan]")
        for i, q in enumerate(plan.hr_questions, 1):
            console.print(
                Panel(
                    f"[bold]Q{i}:[/bold] {q.question}\n\n"
                    f"[green]Model Answer:[/green] {q.model_answer}\n\n"
                    f"[yellow]Tips:[/yellow] {' | '.join(q.improvement_tips)}",
                    border_style="cyan",
                )
            )


def display_evaluation(evaluation) -> None:
    """Render an answer evaluation to the terminal."""
    score = evaluation.score
    colour = "green" if score >= 7 else "yellow" if score >= 5 else "red"

    console.print(
        Panel(
            f"[bold {colour}]Score: {score}/10[/bold {colour}]\n\n"
            f"[green]✅ Strengths:[/green] {', '.join(evaluation.strengths) or 'None noted'}\n\n"
            f"[red]❌ Weaknesses:[/red] {', '.join(evaluation.weaknesses) or 'None noted'}\n\n"
            f"[blue]💡 Improved Answer:[/blue] {evaluation.improved_answer}\n\n"
            f"[yellow]🎯 Confidence Tip:[/yellow] {evaluation.confidence_tip}",
            title="[bold]Answer Evaluation",
            border_style=colour,
        )
    )


def display_skill_gap(gap_report) -> None:
    """Render skill gap analysis."""
    console.rule("[bold red]Skill Gap Analysis")

    # Strengths table
    if gap_report.strength_skills:
        t = Table(title="✅ Your Strengths", style="green")
        t.add_column("Skill", style="bold green")
        for s in gap_report.strength_skills:
            t.add_row(s)
        console.print(t)

    # Gaps table
    if gap_report.gap_skills:
        t = Table(title="⚠️ Skills to Develop", style="red")
        t.add_column("Skill", style="bold red")
        for g in gap_report.gap_skills:
            t.add_row(g)
        console.print(t)

    # Tips
    if gap_report.soft_skill_tips:
        console.print("\n[bold yellow]Improvement Tips:[/bold yellow]")
        for tip in gap_report.soft_skill_tips:
            console.print(f"  • {tip}")

    # Resources
    if gap_report.learning_resources:
        console.print("\n[bold blue]📚 Learning Resources:[/bold blue]")
        for r in gap_report.learning_resources:
            console.print(f"  • {r}")


def mock_interview_loop(agent, session, plan) -> None:
    """Interactive mock interview loop."""
    all_questions = (
        plan.technical_questions
        + plan.behavioral_questions
        + plan.hr_questions
    )
    if not all_questions:
        console.print("[red]No questions available for mock interview.[/red]")
        return

    console.rule("[bold green]🎤 Mock Interview Session")
    console.print("[dim]Type your answers as you would in a real interview. Press Enter twice to submit.[/dim]\n")

    for i, q in enumerate(all_questions, 1):
        console.print(f"\n[bold cyan]Question {i}/{len(all_questions)}:[/bold cyan]")
        console.print(Panel(q.question, border_style="cyan"))

        lines: list[str] = []
        console.print("[dim]Your answer (press Enter on an empty line to submit):[/dim]")
        while True:
            line = input()
            if not line:
                break
            lines.append(line)

        if not lines:
            console.print("[yellow]Skipped.[/yellow]")
            continue

        candidate_answer = " ".join(lines)
        with console.status("[bold green]Evaluating your answer with IBM Granite…"):
            evaluation = agent.evaluate_answer(
                session=session,
                question=q.question,
                model_answer=q.model_answer,
                candidate_answer=candidate_answer,
            )

        display_evaluation(evaluation)

        if i < len(all_questions):
            if not Confirm.ask("\n[cyan]Continue to next question?"):
                break


def chat_mode(agent, session) -> None:
    """Free-form Q&A with the agent."""
    console.rule("[bold blue]💬 Chat with your Interview Coach")
    console.print("[dim]Ask any interview-related question. Type 'exit' to quit.[/dim]\n")
    while True:
        user_input = Prompt.ask("[bold cyan]You")
        if user_input.lower() in ("exit", "quit", "q"):
            break
        with console.status("[bold green]Thinking…"):
            response = agent.chat(session, user_input)
        console.print(Panel(response, title="[green]Coach", border_style="green"))


def main() -> None:
    setup_logging()
    print_banner()

    # Check credentials
    from src.config import config
    missing = config.validate()
    if missing:
        console.print(
            Panel(
                f"[yellow]⚠️  Missing environment variables: {', '.join(missing)}\n\n"
                "Copy [bold]env.example[/bold] to [bold].env[/bold] and fill in your "
                "IBM Cloud API key and Watsonx project ID.\n\n"
                "[dim]The agent will use local HuggingFace embeddings as a fallback.[/dim]",
                title="Configuration Warning",
                border_style="yellow",
            )
        )
        if not Confirm.ask("Continue anyway (limited functionality)?"):
            sys.exit(0)

    # Initialise agent
    with console.status("[bold blue]Initialising IBM Interview Trainer Agent…"):
        from src.agent import InterviewTrainerAgent
        agent = InterviewTrainerAgent()

    # Collect profile
    profile_data = collect_user_profile()

    # Start session
    with console.status("[bold blue]Starting session and parsing profile…"):
        session = agent.start_session(
            name=profile_data["name"],
            job_role=profile_data["job_role"],
            experience_level=profile_data["level"],
            resume_path=profile_data.get("resume_path"),
        )

    console.print(
        f"\n[bold green]✅ Session started![/bold green] "
        f"ID: [dim]{session.session_id}[/dim] | "
        f"Skills identified: [bold]{len(session.profile.skills) if session.profile else 0}[/bold]"
    )

    # Main menu loop
    while True:
        console.rule("[bold]Main Menu")
        choice = Prompt.ask(
            "\n[cyan]What would you like to do?[/cyan]",
            choices=["1", "2", "3", "4", "5"],
            default="1",
        )
        console.print(
            "\n  [bold]1[/bold] Generate interview plan\n"
            "  [bold]2[/bold] Mock interview (practice answering)\n"
            "  [bold]3[/bold] Skill gap analysis\n"
            "  [bold]4[/bold] Chat with your coach\n"
            "  [bold]5[/bold] Exit\n"
        )
        choice = Prompt.ask(
            "[cyan]Enter choice",
            choices=["1", "2", "3", "4", "5"],
            default="1",
        )

        if choice == "1":
            with console.status("[bold blue]Generating your personalised interview plan via IBM Granite…"):
                plan = agent.generate_plan(session)
            display_plan(plan)

        elif choice == "2":
            if not session.plan:
                console.print("[yellow]Please generate an interview plan first (option 1).[/yellow]")
                continue
            mock_interview_loop(agent, session, session.plan)

        elif choice == "3":
            with console.status("[bold blue]Analysing skill gaps…"):
                gap_report = agent.get_skill_gap(session)
            display_skill_gap(gap_report)

        elif choice == "4":
            chat_mode(agent, session)

        elif choice == "5":
            summary = agent.session_summary(session)
            console.print(
                Panel(
                    f"Session: [bold]{summary['session_id']}[/bold]\n"
                    f"Candidate: [bold]{summary['name']}[/bold]\n"
                    f"Role: [bold]{summary['job_role']}[/bold]\n"
                    f"Questions generated: [bold]{summary['questions_generated']}[/bold]\n"
                    f"Answers evaluated: [bold]{summary['answers_evaluated']}[/bold]\n"
                    f"Average score: [bold]{summary['average_score']}[/bold]\n"
                    f"Skill gaps: [bold]{', '.join(summary['skill_gaps']) or 'None'}[/bold]",
                    title="[bold green]Session Summary",
                    border_style="green",
                )
            )
            console.print("\n[bold blue]Good luck with your interview! 🚀[/bold blue]\n")
            break


if __name__ == "__main__":
    main()
