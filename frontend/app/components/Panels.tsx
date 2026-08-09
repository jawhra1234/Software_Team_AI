// Read-only panels: streaming timeline, plan, diff viewer, review verdict.
import type { FinalState, RunEvent } from "@/app/lib/api";

export function Timeline({ events }: { events: RunEvent[] }) {
  return (
    <div className="card">
      <h2>Timeline · {events.length} events</h2>
      <div className="timeline">
        {events.length === 0 && <div className="empty">no events yet</div>}
        {events.map((e) => (
          <div className="ev" key={e.seq}>
            <span className="k">{e.type}</span>
            <span className="n">{e.node ?? (e.request?.kind ?? "")}</span>
            <span>{describe(e)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function describe(e: RunEvent): string {
  if (e.type === "interrupt") return e.request?.context ?? "";
  if (e.type === "done") return `status=${e.status}`;
  if (e.type === "error") return e.error ?? "";
  if (e.type === "tool") {
    const d = e.data ?? {};
    return `${d.tool}${d.ok === false ? " ✗" : ""}`;
  }
  if (e.type === "node_end") {
    const d = e.data ?? {};
    if ("verify_result" in d && d.verify_result) {
      const vr = d.verify_result as { passed: boolean };
      return vr.passed ? "verify PASS" : "verify FAIL";
    }
    if ("review" in d && d.review) return `review: ${(d.review as { verdict: string }).verdict}`;
    if ("plan" in d && d.plan) return `plan v${(d.plan as { version?: number }).version ?? 1}`;
  }
  return "";
}

export function PlanPanel({ state }: { state: FinalState | null }) {
  const plan = state?.plan;
  return (
    <div className="card">
      <h2>Plan</h2>
      {!plan ? (
        <div className="empty">not planned yet</div>
      ) : (
        <>
          <div style={{ marginBottom: 8 }}>{plan.summary}</div>
          <ol style={{ margin: 0, paddingLeft: 18 }}>
            {(plan.tasks ?? []).map((t) => (
              <li key={t.id}>
                <span className="muted">[{t.kind}]</span> {t.title}
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}

export function DiffPanel({ state }: { state: FinalState | null }) {
  const diff = state?.diff_summary;
  return (
    <div className="card">
      <h2>Diff · {state?.changed_files?.length ?? 0} files</h2>
      {!diff || diff === "(no changes)" ? (
        <div className="empty">no changes yet</div>
      ) : (
        <pre className="code diff">
          {diff.split("\n").map((line, i) => (
            <div key={i} className={diffClass(line)}>
              {line}
            </div>
          ))}
        </pre>
      )}
    </div>
  );
}

function diffClass(line: string): string {
  if (line.startsWith("+")) return "add";
  if (line.startsWith("-")) return "del";
  if (line.startsWith("@@") || line.startsWith("diff ")) return "hunk";
  return "";
}

export function ReviewPanel({ state }: { state: FinalState | null }) {
  const review = state?.review;
  return (
    <div className="card">
      <h2>Review</h2>
      {!review ? (
        <div className="empty">not reviewed yet</div>
      ) : (
        <>
          <div>
            verdict:{" "}
            <span className={`pill ${review.verdict === "approved" ? "done" : "failed"}`}>
              {review.verdict}
            </span>
          </div>
          {review.summary && <div className="muted" style={{ marginTop: 6 }}>{review.summary}</div>}
          {(review.issues ?? []).map((iss, i) => (
            <div className="issue" key={i}>
              <span className={`sev ${iss.severity}`}>{iss.severity}</span> {iss.description}
              {iss.file ? <span className="muted"> ({iss.file})</span> : null}
            </div>
          ))}
        </>
      )}
    </div>
  );
}
