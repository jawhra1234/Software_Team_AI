// The one place a human is asked anything — surfaced when the run pauses.
import type { HitlRequest } from "@/app/lib/api";

export function HitlCard({
  request,
  busy,
  onDecision,
}: {
  request: HitlRequest;
  busy: boolean;
  onDecision: (decision: string) => void;
}) {
  // Fall back to sensible defaults if the backend didn't enumerate options.
  const options = request.options.length ? request.options : ["approve", "abort"];
  return (
    <div className="card hitl">
      <h2>⏸ Human input needed · {request.kind}</h2>
      <div className="ctx">{request.context || "(no context)"}</div>
      <div className="actions">
        {options.map((opt) => (
          <button
            key={opt}
            className={opt === "approve" || opt === "accept" ? "primary" : ""}
            disabled={busy}
            onClick={() => onDecision(opt)}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}
