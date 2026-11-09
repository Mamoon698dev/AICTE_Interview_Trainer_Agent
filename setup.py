"""
setup.py -- One-click setup script for the Interview Trainer Agent.

Run: python setup.py
"""

import os
import sys
import shutil
from pathlib import Path


def main() -> None:
    # Ensure UTF-8 output on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  IBM Interview Trainer Agent - Setup")
    print("=" * 60)

    root = Path(__file__).parent

    # -- 1. Create required directories
    dirs = [
        root / "data",
        root / "data" / "chroma_db",
        root / "data" / "resumes",
        root / "data" / "knowledge_base",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Directory ready: {d.relative_to(root)}")

    # -- 2. Check .env
    env_file = root / ".env"
    env_example = root / "env.example"
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("\n  [OK] Created .env from env.example")
            print("  [!!] Edit .env and add your IBM Cloud API key and Project ID!")
        else:
            print("\n  [!!] env.example not found. Please create .env manually.")
    else:
        print(f"\n  [OK] .env file already exists.")

    # -- 3. Validate credentials
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        api_key = os.getenv("WATSONX_API_KEY", "")
        project_id = os.getenv("WATSONX_PROJECT_ID", "")
        missing = []
        if not api_key or "your_ibm" in api_key:
            missing.append("WATSONX_API_KEY")
        if not project_id or "your_watsonx" in project_id:
            missing.append("WATSONX_PROJECT_ID")
        if missing:
            print(f"\n  [!!] Missing credentials: {', '.join(missing)}")
            print("       Open .env and fill them in.")
        else:
            print("\n  [OK] IBM Cloud credentials found.")
    except ImportError:
        print("\n  [i] python-dotenv not installed yet.")

    # -- 4. Seed knowledge base
    try:
        sys.path.insert(0, str(root))
        from src.knowledge_base import seed_knowledge_base
        written = seed_knowledge_base()
        if written:
            print(f"\n  [OK] Seeded {written} knowledge-base files.")
        else:
            print("\n  [OK] Knowledge base already populated.")
    except Exception as exc:
        print(f"\n  [!!] Could not seed knowledge base: {exc}")

    # -- 5. Print run instructions
    print("\n" + "=" * 60)
    print("  Setup complete! Next steps:")
    print()
    print("  1. Edit .env with your IBM Cloud credentials:")
    print("       WATSONX_API_KEY=<your_api_key>")
    print("       WATSONX_PROJECT_ID=<your_project_id>")
    print()
    print("  2. Install dependencies:")
    print("       pip install -r requirements.txt")
    print()
    print("  3. Launch the Streamlit web UI:")
    print("       streamlit run app.py")
    print()
    print("  4. OR launch the terminal CLI:")
    print("       python cli.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
