"""V2-PRE-C — run v1 architectural assertions on the v2 substrate (instr §7.2).

STOP triggers:
  1. Any of the 11 architectural assertions fails on any arm.
  2. v2 synthetic-stream input produces out-of-contract output (non-finite /
     wrong shape / log_var unclamped) on any arm.
  3. Parameter counts diverge from v1's committed parameter_counts.json.
"""

from __future__ import annotations

import torch

from v2.src.preflight.pre_c_arch_assertions_v2_substrate import run_pre_c, write_report


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pre_c] device={device}")
    report = run_pre_c(device=device)
    path = write_report(report)

    stops = []
    if not report["all_assertions_passed"]:
        stops.append("TRIGGER 1: an architectural assertion failed on the v2 substrate")
    if not report["v2_substrate_forward_smoke_ok"]:
        stops.append("TRIGGER 2: v2 synthetic input produced out-of-contract output")
    if not report["parameter_counts_match_v1"]:
        stops.append("TRIGGER 3: parameter counts diverge from v1 parameter_counts.json")

    print(f"[pre_c] total assertions: {report['total_assertions']} "
          f"(all_passed={report['all_assertions_passed']})")
    for arm in report["arms"]:
        passed = sum(a["passed"] for a in arm["assertions"])
        print(f"  {arm['arm']:<10} {passed}/{len(arm['assertions'])} assertions PASS  "
              f"| params={arm['parameter_count']:,}  "
              f"| v2-smoke finite={arm['v2_synthetic_window_smoke']['finite_ok']}")
    print(f"[pre_c] parameter_counts_match_v1: {report['parameter_counts_match_v1']}")
    print(f"[pre_c] v2 forward smoke ok: {report['v2_substrate_forward_smoke_ok']}")
    print(f"[pre_c] report: {path}")

    if stops:
        print("\n[pre_c] STOP TRIGGERS FIRED:")
        for s in stops:
            print("   -", s)
        return 1
    print("\n[pre_c] PRE-C complete: no STOP triggers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
