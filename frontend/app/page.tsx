"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { HitlCard } from "@/app/components/HitlCard";
import { DiffPanel, PlanPanel, ReviewPanel, Timeline } from "@/app/components/Panels";
import { NodeState, RunGraph } from "@/app/components/RunGraph";
import {
  Autonomy,
  FinalState,
  HitlRequest,
  RunEvent,
  respond,
  startRun,
  subscribe,
} from "@/app/lib/api";

const GATE_KINDS = new Set(["plan_approval", "escalation", "final_accept"]);

interface View {
  status: string;
  pendingInterrupt: HitlRequest | null;
  nodeStates: Record<string, NodeState>;
  hints: Record<string, string>;
  liveState: FinalState;
}

const EMPTY_STATE: FinalState = {
  status: null, plan: null, review: null, verify_result: null,
  diff_summary: "", changed_files: [], node_history: [],
};

// Everything the UI shows is derived from the ordered event stream — one source of truth.
function computeView(events: RunEvent[]): View {
  const nodeStates: Record<string, NodeState> = {};
  const hints: Record<string, string> = {};
  let liveState: FinalState = { ...EMPTY_STATE };
  let pendingInterrupt: HitlRequest | null = null;
  let status = events.length ? "running" : "idle";

  for (const e of events) {
    if (e.type === "queued") status = "queued";
    if (e.type === "node_start" && e.node) {
      nodeStates[e.node] = "active";
      status = "running";
    }
    if (e.type === "tool" && e.node) {
      // live activity: surface the current tool on the (still-active) node
      hints[e.node] = String((e.data ?? {}).tool ?? "");
    }
    if (e.type === "node_end" && e.node) {
      nodeStates[e.node] = "done";
      const d = e.data ?? {};
      if (d.plan) liveState = { ...liveState, plan: d.plan as FinalState["plan"] };
      if (d.review) {
        liveState = { ...liveState, review: d.review as FinalState["review"] };
        hints[e.node] = (d.review as { verdict: string }).verdict;
      }
      if (d.verify_result) {
        liveState = { ...liveState, verify_result: d.verify_result as FinalState["verify_result"] };
        hints[e.node] = (d.verify_result as { passed: boolean }).passed ? "PASS" : "FAIL";
      }
      if (typeof d.diff_summary === "string") liveState = { ...liveState, diff_summary: d.diff_summary };
      if (Array.isArray(d.changed_files)) {
        liveState = { ...liveState, changed_files: d.changed_files as FinalState["changed_files"] };
      }
    }
    if (e.type === "interrupt" && e.request) {
      pendingInterrupt = e.request;
      status = "waiting_human";
    }
    if (e.type === "resumed") {
      pendingInterrupt = null;
      status = "running";
    }
    if (e.type === "done") {
      pendingInterrupt = null;
      status = e.status ?? "done";
      if (e.state) liveState = e.state;
    }
    if (e.type === "error") {
      status = "error";
    }
  }

  if (pendingInterrupt) {
    const target = GATE_KINDS.has(pendingInterrupt.kind) ? "human_gate" : null;
    if (target) nodeStates[target] = "waiting";
  }
  return { status, pendingInterrupt, nodeStates, hints, liveState };
}

export default function Page() {
  const [request, setRequest] = useState("Create calc.py with add(a, b) and a passing test_calc.py.");
  const [autonomy, setAutonomy] = useState<Autonomy>("semi");
  const [runId, setRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [starting, setStarting] = useState(false);
  const [responding, setResponding] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const seen = useRef<Set<number>>(new Set());

  const view = useMemo(() => computeView(events), [events]);

  useEffect(() => {
    if (!runId) return;
    seen.current = new Set();
    setEvents([]);
    const stop = subscribe(runId, (e) => {
      if (seen.current.has(e.seq)) return; // backend replays from 0 on reconnect
      seen.current.add(e.seq);
      setEvents((prev) => [...prev, e].sort((a, b) => a.seq - b.seq));
    });
    return stop;
  }, [runId]);

  const onStart = useCallback(async () => {
    setErr(null);
    setStarting(true);
    try {
      const snap = await startRun(request, autonomy);
      setRunId(snap.run_id);
    } catch (e) {
      setErr(String(e));
    } finally {
      setStarting(false);
    }
  }, [request, autonomy]);

  const onDecision = useCallback(
    async (decision: string) => {
      if (!runId) return;
      setResponding(true);
      try {
        await respond(runId, decision);
      } catch (e) {
        setErr(String(e));
      } finally {
        setResponding(false);
      }
    },
    [runId],
  );

  return (
    <div className="wrap">
      <div className="topbar">
        <h1>AI SWE · Mission Control</h1>
        <span className="sub">live view of the supervised coding-agent pipeline</span>
        {runId && <span className={`pill ${view.status}`} style={{ marginLeft: "auto" }}>{view.status}</span>}
      </div>

      <div className="form">
        <textarea
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          placeholder="Describe a coding task…"
        />
        <select value={autonomy} onChange={(e) => setAutonomy(e.target.value as Autonomy)}>
          <option value="auto">auto</option>
          <option value="semi">semi</option>
          <option value="manual">manual</option>
        </select>
        <button className="primary" onClick={onStart} disabled={starting || !request.trim()}>
          {starting ? "starting…" : "Start run"}
        </button>
      </div>

      {err && <div className="card" style={{ borderColor: "var(--bad)", marginBottom: 14 }}>{err}</div>}

      {runId && (
        <>
          <div className="card" style={{ marginBottom: 14 }}>
            <h2>Pipeline · {runId}</h2>
            <RunGraph states={view.nodeStates} hints={view.hints} />
          </div>

          {view.pendingInterrupt && (
            <div style={{ marginBottom: 14 }}>
              <HitlCard request={view.pendingInterrupt} busy={responding} onDecision={onDecision} />
            </div>
          )}

          <div className="grid" style={{ marginBottom: 14 }}>
            <PlanPanel state={view.liveState} />
            <ReviewPanel state={view.liveState} />
          </div>
          <div className="grid" style={{ marginBottom: 14 }}>
            <DiffPanel state={view.liveState} />
            <Timeline events={events} />
          </div>
        </>
      )}

      {!runId && (
        <div className="card empty">
          Start a run to watch it flow through plan → human_gate → coder → verify → review → finalize.
        </div>
      )}
    </div>
  );
}
