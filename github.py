# publish_github.py
# pip install requests

import configparser
import os
import subprocess
import sys
from pathlib import Path

import requests


def run(cmd, cwd=None, check=True):
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout.strip()


def load_config(cfg_path: Path):
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.ini not found: {cfg_path}")

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")

    gh = cfg["github"]
    pr = cfg["project"]

    username = gh.get("username", "").strip()
    token = gh.get("token", "").strip()
    repo_name = gh.get("repo_name", "").strip()
    private = gh.get("private", "true").strip().lower() in ("1", "true", "yes", "y")

    project_path = Path(pr.get("path", "").strip())

    if not username or not token or not repo_name or not str(project_path):
        raise ValueError("Missing config values in config.ini ([github] username/token/repo_name, [project] path)")

    return username, token, repo_name, private, project_path


def ensure_git_repo(project_path: Path):
    if not project_path.exists():
        raise FileNotFoundError(f"Project path not found: {project_path}")

    git_dir = project_path / ".git"
    if not git_dir.exists():
        run(["git", "init"], cwd=project_path)
        run(["git", "checkout", "-b", "main"], cwd=project_path, check=False)

    # Ensure .gitignore contains config.ini
    gi = project_path / ".gitignore"
    if gi.exists():
        content = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
    else:
        content = []

    if "config.ini" not in content:
        content.append("config.ini")
        gi.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8")

    # Commit if needed
    run(["git", "add", "."], cwd=project_path)
    # Check if there is anything to commit
    status = run(["git", "status", "--porcelain"], cwd=project_path, check=False)
    if status.strip():
        run(["git", "commit", "-m", "init"], cwd=project_path, check=False)

    # Ensure branch is main
    run(["git", "branch", "-M", "main"], cwd=project_path, check=False)


def create_repo_if_needed(username: str, token: str, repo_name: str, private: bool):
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"name": repo_name, "private": private}

    r = requests.post(url, headers=headers, json=payload, timeout=30)

    if r.status_code == 201:
        print(f"Repo created: {username}/{repo_name}")
        return
    if r.status_code == 422:
        # Usually "name already exists on this account"
        print(f"Repo already exists (skip create): {username}/{repo_name}")
        return

    raise RuntimeError(f"Create repo failed: {r.status_code}\n{r.text}")


def set_remote_and_push(project_path: Path, username: str, token: str, repo_name: str):
    authed = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
    clean = f"https://github.com/{username}/{repo_name}.git"

    # Set origin (with token) for push
    run(["git", "remote", "remove", "origin"], cwd=project_path, check=False)
    run(["git", "remote", "add", "origin", authed], cwd=project_path)

    # Push
    run(["git", "push", "-u", "origin", "main"], cwd=project_path)

    # Replace remote to clean URL (avoid leaving token in remote)
    run(["git", "remote", "set-url", "origin", clean], cwd=project_path)
    print("Push done. Remote reset to clean URL.")


def main():
    cfg_path = Path(__file__).with_name("config.ini")
    username, token, repo_name, private, project_path = load_config(cfg_path)

    ensure_git_repo(project_path)
    create_repo_if_needed(username, token, repo_name, private)
    set_remote_and_push(project_path, username, token, repo_name)

    print("\nDone.")
    print(f"Repo: https://github.com/{username}/{repo_name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
