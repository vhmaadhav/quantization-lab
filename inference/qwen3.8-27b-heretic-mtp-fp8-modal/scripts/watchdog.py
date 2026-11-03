#!/usr/bin/env python3
"""Cost and safety watchdog for the qwen-abliterated Modal deployment.

Stdlib only. It talks to Modal only through the `modal` CLI: read-only `list`
commands, plus `app stop` / `container stop` behind --fix. It never calls the
served endpoint, and it never reads or prints the value of VLLM_API_KEY.
Run it through ./wd from the repo root.
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = "qwen-abliterated"
EXPECTED_VOLUMES = {"qwen-weights", "qwen-vllm-cache"}
EXPECTED_SECRETS = {"qwen-api-key"}
# Optional spend thresholds for COST.md. WD_BUDGET (dollars) sets WARN at 40% and ALERT at 75%.
# Unset or 0 reports the estimate without thresholds.
try:
    BUDGET = float(os.environ.get("WD_BUDGET", "0") or 0)
except ValueError:
    BUDGET = 0.0
GPU_RATE_PER_S = 0.000542  # L40S list price (modal.com/pricing)


def _max_age_from_env():
    raw = os.environ.get("WD_MAX_AGE_MIN", "35")
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        sys.stderr.write(f"WD_MAX_AGE_MIN must be a positive whole number of minutes, got {raw!r}\n")
        sys.exit(64)
    return value


# Runaway threshold in minutes. Default 35 = 1800 s QWEN_TIMEOUT + 5 min margin.
# With QWEN_TIMEOUT=10800 run with WD_MAX_AGE_MIN=185.
MAX_AGE_MIN = _max_age_from_env()
ACTIVE_STATES = {"running", "ephemeral", "initializing"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}
RANK = {"OK": 0, "WARN": 1, "ALERT": 2}
LABEL = {0: "OK", 1: "WARN", 2: "ALERT"}


class Parser(argparse.ArgumentParser):
    def error(self, message):  # exit 64 for usage errors, so 2 always means ALERT
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


def parse_args(argv):
    p = Parser(
        prog="wd",
        description="Watchdog for the qwen-abliterated Modal app: spend, containers, "
                    "stray apps, resources, secrets hygiene and config drift.",
        epilog="exit codes: 0 all OK, 1 WARN, 2 ALERT, 64 usage error.\n"
               "Read-only unless --fix. Nothing here calls the served endpoint.\n"
               "--fix asks y/N before each stop and needs a terminal; --dry-run only lists them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--watch", nargs="?", const=300, type=int, metavar="N",
                   help="re-check every N seconds (default 300, minimum 60); print changes only; "
                        "exit 2 on the first ALERT")
    p.add_argument("--fix", action="store_true",
                   help="after the checks, offer to stop unexpected apps and runaway/extra containers")
    p.add_argument("--dry-run", action="store_true", help="list what --fix would stop, stop nothing")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--notify", action="store_true", help="run notify-send on ALERT if it is installed")
    args = p.parse_args(argv)
    if args.watch is not None and args.watch < 60:
        p.error("--watch N must be at least 60")
    if args.watch is not None and (args.fix or args.dry_run):
        p.error("--fix and --dry-run run once; drop --watch")
    if args.fix and args.json:
        p.error("--fix is interactive; drop --json")
    return args


def check(name, status="OK", reason="", actions=None, manual=None, hints=None):
    return {"name": name, "status": status, "reason": reason, "actions": actions or [],
            "manual": manual or [], "hints": hints or []}


def modal_json(*args):
    exe = shutil.which("modal")
    if exe is None:
        raise RuntimeError("modal CLI not on PATH")
    cmd = " ".join(args)
    try:
        proc = subprocess.run([exe, *args, "--json"], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"`modal {cmd}` timed out")
    if proc.returncode != 0:
        last = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["unknown error"]
        raise RuntimeError(f"`modal {cmd}` failed: {last[0][:160]}")
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        raise RuntimeError(f"`modal {cmd}` did not return JSON")


def parse_time(text):
    when = dt.datetime.fromisoformat(text)
    return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)


def age_min(container, now):
    try:  # Modal shows 'Pending' until the container has started
        return (now - parse_time(container["start_time"])).total_seconds() / 60
    except (KeyError, TypeError, ValueError):
        return None


def guarded(name, fn):
    try:
        return fn()
    except Exception as exc:  # a failed check is a WARN, never a crash
        return check(name, "WARN", f"could not check: {exc}")


def check_apps(apps):
    deployed = [a for a in apps if a.get("description") == APP and a.get("state") == "deployed"]
    ids = {a["app_id"] for a in deployed}
    problems, actions, manual = [], [], []
    for a in apps:
        name, state, aid = a.get("description"), a.get("state"), a.get("app_id")
        if state not in ACTIVE_STATES and state != "deployed":
            continue
        if aid in ids and len(deployed) == 1:
            continue
        desc = f"{name} ({aid}) is {state}"
        problems.append(desc)
        if name == APP:  # never auto-stop the qwen-abliterated app itself
            manual.append(f"{desc}: stop by hand with `modal app stop {aid}`")
        else:
            actions.append({"label": f"stop app {name} ({aid})",
                            "cmd": ["modal", "app", "stop", aid, "-y"]})
    if len(deployed) > 1:
        problems.append(f"{len(deployed)} deployed {APP} apps (expected 1)")
    if problems:
        status, reason = "ALERT", "; ".join(problems)
    elif not deployed:
        status, reason = "WARN", f"no deployed {APP} (nothing bills; fine if intentional)"
    else:
        status, reason = "OK", f"deployed {APP} is the only active app"
    return check("apps", status, reason, actions, manual), ids


def check_containers(conts, ids, now):
    problems, actions, hints, seen = [], [], [], set()

    def propose(c, why):
        cid = c["container_id"]
        if cid not in seen:
            seen.add(cid)
            actions.append({"label": f"stop container {cid} ({why})",
                            "cmd": ["modal", "container", "stop", cid, "-y"]})

    mine = sorted((c for c in conts if c.get("app_id") in ids), key=lambda c: c.get("start_time") or "")
    for c in conts:
        cid = c["container_id"]
        if c.get("app_id") not in ids:
            problems.append(f"container {cid} belongs to {c.get('app_name')}, not the deployed {APP}")
            propose(c, "unexpected app")
        age = age_min(c, now)
        if age is not None and age > MAX_AGE_MIN:
            problems.append(f"container {cid} up {age:.0f} min: possible runaway "
                            f"(over WD_MAX_AGE_MIN={MAX_AGE_MIN})")
            propose(c, f"older than {MAX_AGE_MIN} min")
        elif age is not None and c.get("app_id") in ids:
            hints.append(f"container {cid} up {age:.0f} min; it should scale down 5 min after the "
                         f"last request, and the function timeout (QWEN_TIMEOUT) stops it regardless")
        elif age is None:
            hints.append(f"container {cid} start_time is {c.get('start_time')!r}: age unknown, re-run shortly")
        if age is not None:  # spend estimate for every live container (GPU only, no CPU/memory)
            uptime_s = age * 60
            hints.append(f"container {cid} est. cost so far ${uptime_s * GPU_RATE_PER_S:.2f} "
                         f"(uptime {uptime_s:.0f} s x {GPU_RATE_PER_S} $/s, GPU only)")
    if len(conts) > 1:
        problems.append(f"{len(conts)} containers live (max_containers=1)")
    for c in mine[:-1]:
        propose(c, "extra container")
    if problems:
        status, reason = "ALERT", "; ".join(problems)
    elif not conts:
        status, reason = "OK", "no live containers (no GPU billing)"
    else:
        status, reason = "OK", (f"{len(conts)} container live: normal while active, "
                                "scales down 5 min after the last request")
    return check("containers", status, reason, actions, hints=hints)


def check_spend():
    cost_path = os.path.join(REPO, "COST.md")
    if not os.path.isfile(cost_path):  # optional spend log: skip cleanly when it is absent
        return check("spend", "OK", "skipped: no COST.md next to this folder (optional spend log; "
                     "the Modal dashboard is the source of truth)")
    try:
        with open(cost_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        return check("spend", "WARN", f"cannot read COST.md ({exc.strerror})")
    idx, last = None, None
    for line in lines:  # last numeric cell of the 'Running total' column wins
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if idx is None:
            if any(c.startswith("Running total") for c in cells):
                idx = next(i for i, c in enumerate(cells) if c.startswith("Running total"))
            continue
        if set(line) <= set("|-: "):
            continue
        m = re.search(r"\d+(?:\.\d+)?", cells[idx]) if len(cells) > idx else None
        if m:
            last = float(m.group(0))
    if last is None:
        return check("spend", "WARN", "could not parse the Running total in COST.md")
    warn_at, alert_at = (0.4 * BUDGET, 0.75 * BUDGET) if BUDGET > 0 else (float("inf"), float("inf"))
    if last >= alert_at:
        status = "ALERT"
    elif last >= warn_at:
        status = "WARN"
    else:
        status = "OK"
    budget_note = (f" of ${BUDGET:.2f} budget (about ${BUDGET - last:.2f} left)" if BUDGET > 0
                   else " (set WD_BUDGET to enable thresholds)")
    return check("spend", status, f"est. ${last:.2f} spent{budget_note}; "
                 "from COST.md, the Modal dashboard is the source of truth")


def check_resources():
    vols = {v["name"] for v in modal_json("volume", "list")}
    secs = {s["name"] for s in modal_json("secret", "list")}
    issues = []
    if vols - EXPECTED_VOLUMES:
        issues.append("unexpected volumes: " + ", ".join(sorted(vols - EXPECTED_VOLUMES)))
    if secs - EXPECTED_SECRETS:
        issues.append("unexpected secrets (names only): " + ", ".join(sorted(secs - EXPECTED_SECRETS)))
    if EXPECTED_VOLUMES - vols:
        issues.append("missing volumes: " + ", ".join(sorted(EXPECTED_VOLUMES - vols)))
    if EXPECTED_SECRETS - secs:
        issues.append("missing secrets: " + ", ".join(sorted(EXPECTED_SECRETS - secs)))
    if issues:
        return check("resources", "WARN", "; ".join(issues))
    return check("resources", "OK", "volumes and secrets have the expected names")


def read_env_key(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def check_secrets():
    env_path = os.path.join(REPO, ".env")
    if not os.path.isfile(env_path):
        return check("secrets", "WARN", ".env not found")
    problems, warns = [], []
    mode = stat.S_IMODE(os.stat(env_path).st_mode)
    if mode != 0o600:
        warns.append(f".env mode is {mode:o}, want 600")
    ignored = subprocess.run(["git", "-C", REPO, "check-ignore", "-q", ".env"]).returncode == 0
    if not ignored:
        problems.append(".env is NOT git-ignored")
    tracked = subprocess.run(["git", "-C", REPO, "ls-files", "--", ".env"],
                             capture_output=True, text=True).stdout.strip()
    if tracked:
        problems.append(".env is tracked by git")
    key = read_env_key(env_path)
    if not key:
        warns.append("VLLM_API_KEY not found in .env")
    scanned, leaks = 0, []
    if key:  # the value is only compared, never printed
        needle = key.encode()
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                path = os.path.join(root, name)
                if os.path.abspath(path) == os.path.abspath(env_path):
                    continue
                try:
                    if os.path.getsize(path) > 50_000_000:
                        continue
                    with open(path, "rb") as fh:
                        data = fh.read()
                except OSError:
                    continue
                scanned += 1
                if needle in data:
                    leaks.append(os.path.relpath(path, REPO))
    if leaks:
        problems.append("API key value appears in: " + ", ".join(leaks))
    if problems:
        return check("secrets", "ALERT", "; ".join(problems + warns))
    if warns:
        return check("secrets", "WARN", "; ".join(warns))
    return check("secrets", "OK", f"mode 600, git-ignored, untracked, key not found in {scanned} files")


def check_config():
    with open(os.path.join(REPO, "modal_app.py"), encoding="utf-8") as fh:
        src = fh.read()
    problems, warns = [], []
    m = re.search(r"min_containers\s*=\s*(\d+)", src)
    if m is None:
        warns.append("min_containers not set (Modal default is 0)")
    elif m.group(1) != "0":
        problems.append(f"min_containers={m.group(1)}: always-on GPU burn")
    m = re.search(r"max_containers\s*=\s*(\d+)", src)
    if m is None or m.group(1) != "1":
        warns.append("max_containers is not 1")
    for key in ("scaledown_window", "timeout"):
        if not re.search(rf"\b{key}\s*=", src):
            warns.append(f"{key} not set")
    pat = re.compile(r"""api[_-]?key\s*[=:]\s*["'][^"'\s]{8,}["']""", re.IGNORECASE)
    for no, line in enumerate(src.splitlines(), 1):
        if pat.search(line):  # report the line number only, never the literal
            problems.append(f"line {no}: hardcoded API key literal")
    if problems:
        return check("config", "ALERT", "; ".join(problems + warns))
    if warns:
        return check("config", "WARN", "; ".join(warns))
    return check("config", "OK", "min_containers=0, max_containers=1, timeout and scaledown set, no hardcoded key")


def gather():
    now = dt.datetime.now(dt.timezone.utc)
    try:
        apps_check, ids = check_apps(modal_json("app", "list"))
    except Exception as exc:
        apps_check, ids = check("apps", "WARN", f"could not check: {exc}"), None
    if ids is None:
        conts_check = check("containers", "WARN", "app list unavailable, containers not judged")
    else:
        conts_check = guarded("containers", lambda: check_containers(modal_json("container", "list"), ids, now))
    checks = [apps_check, conts_check, guarded("spend", check_spend),
              guarded("resources", check_resources), guarded("secrets", check_secrets),
              guarded("config", check_config)]
    hints = [h for c in checks for h in c["hints"]]
    return {"time": now.isoformat(timespec="seconds"), "checks": checks, "hints": hints}


def worst(report):
    return max(RANK[c["status"]] for c in report["checks"])


def line(c):
    return f"{c['status']:<5}  {c['name']:<10} {c['reason']}"


def print_report(report):
    for c in report["checks"]:
        print(line(c))
        for note in c["manual"]:
            print(f"       manual: {note}")
    for h in report["hints"]:
        print(f"INFO   containers  {h}")
    print(f"overall: {LABEL[worst(report)]} (exit {worst(report)})")


def apply_actions(report, dry):
    actions = [a for c in report["checks"] for a in c["actions"]]
    if not actions:
        print("--fix: nothing to stop")
        return
    if not dry and not sys.stdin.isatty():
        print("--fix needs an interactive terminal; nothing was stopped")
        return
    for a in actions:
        cmd = " ".join(a["cmd"])
        if dry:
            print(f"DRY RUN would {a['label']}: {cmd}")
            continue
        try:
            answer = input(f"{a['label']}? run `{cmd}` [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "y":
            print("  skipped")
            continue
        try:
            proc = subprocess.run(a["cmd"], capture_output=True, text=True, timeout=180)
            print(f"  exit {proc.returncode}: {(proc.stdout or proc.stderr).strip()[-200:]}")
        except subprocess.TimeoutExpired:
            print("  timed out; check `modal app list` by hand")


def notify(report):
    exe = shutil.which("notify-send")
    if exe is None:
        return
    alerts = "\n".join(f"{c['name']}: {c['reason']}" for c in report["checks"] if c["status"] == "ALERT")
    try:
        subprocess.run([exe, "qwen-abliterated watchdog: ALERT", alerts[:500]], timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def watch(args):
    last = None
    while True:
        report = gather()
        code = worst(report)
        current = {c["name"]: (c["status"], c["reason"]) for c in report["checks"]}
        if current != last:
            changed = [c for c in report["checks"] if last is None or last.get(c["name"]) != current[c["name"]]]
            stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if args.json:
                print(json.dumps({"time": report["time"], "overall": LABEL[code], "changed": changed}))
            else:
                print(f"[{stamp}] overall {LABEL[code]}")
                for c in changed:
                    print(line(c))
            sys.stdout.flush()
            last = current
        if code == 2:
            if args.notify:
                notify(report)
            return 2
        time.sleep(args.watch)


def main(argv=None):
    args = parse_args(argv)
    if args.watch is not None:
        try:
            return watch(args)
        except KeyboardInterrupt:
            return 0
    report = gather()
    code = worst(report)
    if args.json:
        planned = [{"label": a["label"], "cmd": a["cmd"]} for c in report["checks"] for a in c["actions"]]
        print(json.dumps({"time": report["time"], "overall": LABEL[code], "exit": code,
                          "checks": report["checks"], "hints": report["hints"], "planned": planned},
                         indent=2))
    else:
        print_report(report)
    if args.fix or args.dry_run:
        if not args.json:
            apply_actions(report, dry=args.dry_run)
        if args.fix and not args.dry_run:
            report = gather()
            code = worst(report)
            print("after --fix:")
            print_report(report)
    if code == 2 and args.notify:
        notify(report)
    return code


if __name__ == "__main__":
    sys.exit(main())
