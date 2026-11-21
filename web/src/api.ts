// Client for the FastAPI backend (api/server.py).

export interface Hit {
  id: string
  citation: string // "TS 24.501 §A.1"
  spec: string
  section: string
  title: string
  path: string
  part: number
  n_parts: number
  score: number
  ranks: { vector?: number; bm25?: number; final?: number }
  text: string
}

export interface AnswerEvent {
  text: string
  refused: boolean
  refusal_reason: string | null
  citations: string[]
  invalid_citations: string[]
  model: string
  timings: { retrieval?: number; llm?: number }
}

export interface Settings {
  model: string
  k: number
  hybrid: boolean
  rerank: boolean
  route: boolean
  threshold: number
}

export interface Config {
  models: { id: string; label: string }[]
  default_model: string
  threshold: number
  corpus: { spec: string; version: string; title: string }[]
  chunks: number
}

export type StreamEvent =
  | { type: 'route'; specs: string[]; reason: string }
  | { type: 'retrieval'; hits: Hit[]; timings: Record<string, number>; reranked: boolean }
  | ({ type: 'answer' } & AnswerEvent)
  | { type: 'error'; message: string }

export async function getConfig(): Promise<Config> {
  const res = await fetch('/api/config')
  if (!res.ok) throw new Error(`Backend unavailable (${res.status})`)
  return res.json()
}

/** POST /api/ask and call onEvent for each NDJSON line as it arrives. */
export async function ask(
  question: string,
  settings: Settings,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, ...settings }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let nl: number
    while ((nl = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, nl).trim()
      buffer = buffer.slice(nl + 1)
      if (line) onEvent(JSON.parse(line) as StreamEvent)
    }
  }
}
