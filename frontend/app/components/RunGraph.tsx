// The 6-node pipeline, lighting up as the run progresses.
import { Fragment } from "react";

const NODES: { id: string; label: string }[] = [
  { id: "plan", label: "Plan" },
  { id: "human_gate", label: "Human Gate" },
  { id: "coder", label: "Coder" },
  { id: "verify", label: "Verify" },
  { id: "review", label: "Review" },
  { id: "finalize", label: "Finalize" },
];

export type NodeState = "pending" | "active" | "done" | "waiting";

export function RunGraph({
  states,
  hints,
}: {
  states: Record<string, NodeState>;
  hints: Record<string, string>;
}) {
  return (
    <div className="pipeline">
      {NODES.map((n, i) => (
        <Fragment key={n.id}>
          <div className={`node ${states[n.id] ?? "pending"}`}>
            <div className="name">{n.label}</div>
            <div className="hint">{hints[n.id] ?? ""}</div>
          </div>
          {i < NODES.length - 1 && <div className="arrow">→</div>}
        </Fragment>
      ))}
    </div>
  );
}
