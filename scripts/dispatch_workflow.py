#!/usr/bin/env python3
"""Dispatch a GitHub Actions workflow through the REST API.

``github.com`` can be unreachable while ``api.github.com`` still works, and a
workflow file must exist on the target ref before it can be dispatched. This
uses the same credential git stores, without printing the token.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request


def _token() -> str:
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("git credential 中没有 GitHub token")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="lzo-noth/DAIE")
    parser.add_argument("--workflow", required=True, help="workflow 文件名，如 release_annotate.yml")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--inputs", default="{}", help="workflow inputs 的 JSON")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    inputs = json.loads(args.inputs)
    payload = {"ref": args.ref, "inputs": {str(k): str(v) for k, v in inputs.items()}}
    if args.dry_run:
        print(json.dumps({"workflow": args.workflow, "payload": payload}, ensure_ascii=False))
        return 0

    request = urllib.request.Request(
        f"https://api.github.com/repos/{args.repo}/actions/workflows/{args.workflow}/dispatches",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {_token()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            print(f"dispatched {args.workflow} status={response.status}")
    except urllib.error.HTTPError as exc:
        print(exc.read().decode()[:300], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
