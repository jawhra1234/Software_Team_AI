// API + SSE client for the Mission-Control backend (Phase 6).
// Types mirror the FastAPI surface in backend/app/api/server.py.

export type Autonomy = "auto" | "semi" | "manual";
export type RunStatus = "running" | "waiting_human" | "done" | "error";

export interface HitlRequest {
  kind: string; // plan_approval | escalation | final_accept | clarification | command_approval
  context: string;
  options: string[];
  payload: Record<string, unknown>;
}

export interface ReviewIssue {
  severity: "blocker" | "major" | "minor" | "nit";
  file?: string | null;
  line?: number | null;
  description: string;
  suggestion?: string | null;
}

export interface FinalState {
  status: string | null;
  plan: { summary?: string; tasks?: { id: string; title: string; kind: string }[] } | null;
  review: { verdict: string; summary: string; issues: ReviewIssue[] } | null;
  verify_result: { passed: boolean; summary: string } | null;
  diff_summary: string;
  changed_files: { path: string; status: string }[];
  node_history: string[];
}

export interface RunSnapshot {
  run_id: string;
  request: string;
  autonomy: Autonomy;
  status: RunStatus;
  pending_interrupt: HitlRequest | null;
  event_count: number;
  final_state: FinalState | null;
}

// One streamed event. `type` is the discriminator; other fields depend on it.
export interface RunEvent {
  seq: number;
  ts: string;
  type: string; // node_start | node_end | interrupt | resumed | done | error
  node?: string;
  data?: Record<string, unknown>;
  request?: HitlRequest; // on `interrupt`
  status?: string; // on `done`
  state?: FinalState; // on `done`
  error?: string; // on `error`
  decision?: unknown; // on `resumed`
}

// Talk to the FastAPI backend DIRECTLY (CORS is open) rather than via Next's dev
// rewrite — the rewrite proxy buffers Server-Sent Events, so the browser would
// otherwise see no live events until the stream closed. Override with
// NEXT_PUBLIC_API_BASE if the backend runs elsewhere.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function startRun(request: string, autonomy: Autonomy): Promise<RunSnapshot> {
  const res = await fetch(`${API_BASE}/api/runs`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ request, autonomy }),
  });
  if (!res.ok) throw new Error(`startRun failed: ${res.status}`);
  return res.json();
}

export async function getRun(runId: string): Promise<RunSnapshot> {
  const res = await fetch(`${API_BASE}/api/runs/${runId}`);
  if (!res.ok) throw new Error(`getRun failed: ${res.status}`);
  return res.json();
}

export async function respond(runId: string, decision: string, note?: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/runs/${runId}/respond`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ decision, note: note ?? null }),
  });
  // 409 = the run is no longer awaiting input (already resumed by a prior click).
  // That's benign — the answer was accepted; a duplicate arrived late. Don't surface it.
  if (res.status === 409) return;
  if (!res.ok) throw new Error(`respond failed: ${res.status}`);
}

// Subscribe to a run's SSE stream. Returns a cleanup fn. Dedups by `seq` (the
// backend replays from the start on reconnect, so callers may see repeats).
export function subscribe(runId: string, onEvent: (e: RunEvent) => void): () => void {
  const src = new EventSource(`${API_BASE}/api/runs/${runId}/events`);
  src.onmessage = (msg) => {
    if (!msg.data || msg.data === "{}") return;
    try {
      onEvent(JSON.parse(msg.data) as RunEvent);
    } catch {
      /* ignore malformed frame */
    }
  };
  src.addEventListener("end", () => src.close());
  src.onerror = () => src.close();
  return () => src.close();
}
