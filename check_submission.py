"""Check a ranked day produced by your batch ranking image.

Usage:
    python check_submission.py data/ranked.csv
    python check_submission.py data/ranked.csv --day data/day_sample.csv

By default the day file is day_sample.csv in the same folder as ranked.csv.
Exit code 0 means every check passed.
"""
import argparse
import csv
import math
import os
import sys

REQUIRED = ["cve_id", "rank", "score", "model_version"]
NEVER_WAIT = {"CWE-78", "CWE-94", "CWE-287", "CWE-306", "CWE-502"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames or [], list(reader)


def is_never_wait(cwe_ids):
    return bool({c.strip() for c in (cwe_ids or "").split(";")} & NEVER_WAIT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ranked")
    ap.add_argument("--day", help="the day file that was ranked (default: day_sample.csv next to ranked)")
    args = ap.parse_args()
    day_path = args.day or os.path.join(os.path.dirname(os.path.abspath(args.ranked)), "day_sample.csv")

    failures = []

    def check(ok, message):
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)
        return ok

    if not check(os.path.exists(args.ranked), f"ranked file exists: {args.ranked}"):
        return 1
    if not check(os.path.exists(day_path), f"day file exists: {day_path}"):
        return 1

    cols, rows = read_csv(args.ranked)
    _, day_rows = read_csv(day_path)

    if not check(sorted(cols) == sorted(REQUIRED),
                 f"columns are exactly {REQUIRED} (found {cols})"):
        return 1

    ids = [r["cve_id"] for r in rows]
    day_ids = [r["cve_id"] for r in day_rows]
    check(len(ids) == len(set(ids)), "every cve_id appears once")
    check(set(ids) == set(day_ids),
          f"same CVEs as the day file ({len(set(day_ids))} expected, {len(set(ids))} found, "
          f"{len(set(day_ids) - set(ids))} missing, {len(set(ids) - set(day_ids))} unexpected)")

    try:
        ranks = [int(r["rank"]) for r in rows]
        rank_ok = sorted(ranks) == list(range(1, len(rows) + 1))
    except ValueError:
        ranks, rank_ok = [], False
    check(rank_ok, "ranks are whole numbers from 1 to the number of CVEs, no gaps or repeats")

    try:
        scores = [float(r["score"]) for r in rows]
        score_ok = all(not math.isnan(s) and 0.0 <= s <= 1.0 for s in scores)
    except ValueError:
        scores, score_ok = [], False
    check(score_ok, "every score is a number between 0 and 1")

    versions = {r["model_version"].strip() for r in rows}
    check(len(versions) == 1 and "" not in versions, f"one non-empty model_version (found {sorted(versions)})")

    if rank_ok and score_ok and set(ids) == set(day_ids):
        cwe_of = {r["cve_id"]: r.get("cwe_ids", "") for r in day_rows}
        ordered = sorted(zip(ranks, ids, scores))
        flags = [is_never_wait(cwe_of[cid]) for _, cid, _ in ordered]
        n_nw = sum(flags)
        check(all(flags[:n_nw]) and not any(flags[n_nw:]),
              f"all {n_nw} never-wait CVEs come first")
        for group, members in (("never-wait", ordered[:n_nw]), ("other", ordered[n_nw:])):
            s = [score for _, _, score in members]
            check(all(a >= b for a, b in zip(s, s[1:])),
                  f"inside the {group} group, scores never increase as rank grows")

    print()
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
