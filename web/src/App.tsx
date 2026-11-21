import { useCallback, useEffect, useRef, useState } from 'react'
import { ask, getConfig, type Config, type Settings } from './api'
import { Sources } from './components/Sources'
import { SettingsPanel } from './components/SettingsPanel'
import { TurnView } from './components/TurnView'
import type { Turn } from './types'

const EXAMPLES: { kind: string; tone: string; q: string }[] = [
  { kind: 'NAS · cause', tone: 'teal', q: 'What does cause #9 mean in 5GMM?' },
  { kind: 'NAS · timer', tone: 'teal', q: 'What is timer T3510 used for?' },
  { kind: 'Architecture', tone: 'blue', q: 'What is the role of the AMF?' },
  { kind: 'Cross-spec', tone: 'yellow', q: 'Describe the registration procedure and the NAS messages exchanged between the UE and the AMF.' },
  { kind: 'Cross-spec', tone: 'yellow', q: 'How is a PDU session established and which NAS messages are used?' },
  { kind: 'Out of corpus', tone: 'coral', q: 'What is the capital of France?' },
]

const DEFAULTS: Settings = {
  model: 'openai/gpt-oss-120b', k: 5, hybrid: true, rerank: true, route: true, threshold: 0.02,
}

// Conversation survives a page reload (per browser tab). Unfinished turns are marked interrupted.
function loadTurns(): Turn[] {
  try {
    const ts: Turn[] = JSON.parse(sessionStorage.getItem('turns') ?? '[]')
    return ts.map((t) => (t.stage === 'done' || t.stage === 'error'
      ? t : { ...t, stage: 'error', error: 'Interrupted by a page reload.' }))
  } catch {
    return []
  }
}
const isMobile = () => window.matchMedia('(max-width: 800px)').matches

function loadSettings(): Settings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('settings') ?? '{}') }
  } catch {
    return DEFAULTS
  }
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [backendError, setBackendError] = useState<string | null>(null)
  const [settings, setSettings] = useState<Settings>(loadSettings)
  const [showSettings, setShowSettings] = useState(false)
  const [turns, setTurns] = useState<Turn[]>(loadTurns)
  const [activeTurn, setActiveTurn] = useState<number | null>(null)
  const [activeCitation, setActiveCitation] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [sourcesOpen, setSourcesOpen] = useState(false) // mobile sheet
  const busy = turns.some((t) => t.stage !== 'done' && t.stage !== 'error')
  const nextId = useRef(Math.max(0, ...turns.map((t) => t.id)) + 1)
  const scrollTo = useRef<number | null>(null)
  const settingsRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    getConfig().then(setConfig).catch((e) => setBackendError(String(e.message ?? e)))
  }, [])
  useEffect(() => localStorage.setItem('settings', JSON.stringify(settings)), [settings])
  useEffect(() => sessionStorage.setItem('turns', JSON.stringify(turns)), [turns])

  // After a question is added, bring it to the top of the thread so its answer is read from the start.
  useEffect(() => {
    if (scrollTo.current === null) return
    document.querySelector(`[data-turn="${scrollTo.current}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    scrollTo.current = null
  }, [turns])

  // Settings popover: close on Escape or on a click outside it.
  useEffect(() => {
    if (!showSettings) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setShowSettings(false)
    const onDown = (e: MouseEvent) => {
      const t = e.target as HTMLElement
      if (!settingsRef.current?.contains(t) && !t.closest('[data-settings-toggle]')) setShowSettings(false)
    }
    document.addEventListener('keydown', onKey)
    document.addEventListener('mousedown', onDown)
    return () => { document.removeEventListener('keydown', onKey); document.removeEventListener('mousedown', onDown) }
  }, [showSettings])

  const update = useCallback((id: number, patch: Partial<Turn>) => {
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch } : t)))
  }, [])

  const submit = async (question: string) => {
    const q = question.trim()
    if (!q || busy) return
    const id = nextId.current++
    const turn: Turn = { id, question: q, settings, stage: 'routing', startedAt: performance.now() }
    setTurns((ts) => [...ts, turn])
    setActiveTurn(id)
    setActiveCitation(null)
    setInput('')
    scrollTo.current = id

    try {
      await ask(q, settings, (e) => {
        if (e.type === 'route') update(id, { route: { specs: e.specs, reason: e.reason }, stage: 'retrieving' })
        else if (e.type === 'retrieval')
          update(id, { hits: e.hits, reranked: e.reranked, retrievalTime: e.timings.total, stage: 'generating' })
        else if (e.type === 'answer') update(id, { answer: e, stage: 'done' })
        else if (e.type === 'error') update(id, { error: e.message, stage: 'error' })
      })
    } catch (err) {
      update(id, { error: err instanceof Error ? err.message : String(err), stage: 'error' })
    }
    inputRef.current?.focus()
  }

  const modelLabel = (id: string) => config?.models.find((m) => m.id === id)?.label ?? id
  const current = turns.find((t) => t.id === activeTurn) ?? turns[turns.length - 1] ?? null

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark mono">5G</span>
          <span className="brand-name">Specification Assistant</span>
        </div>
        <div className="top-actions">
          {turns.length > 0 && (
            <button type="button" className="link" onClick={() => { setTurns([]); setActiveTurn(null); inputRef.current?.focus() }} disabled={busy}>
              New chat
            </button>
          )}
          <button type="button" className="btn" data-settings-toggle aria-expanded={showSettings}
            onClick={() => setShowSettings(!showSettings)}>Settings</button>
        </div>
        {showSettings && (
          <div ref={settingsRef}>
            <SettingsPanel config={config} settings={settings} onChange={setSettings} onClose={() => setShowSettings(false)} />
          </div>
        )}
      </header>

      <main className="layout">
        <section className="conversation">
          <div className="thread">
            {backendError && (
              <div className="notice error">
                Cannot reach the backend: {backendError}. Start it with{' '}
                <code>.venv/bin/uvicorn api.server:app --port 8000</code>
              </div>
            )}

            {turns.length === 0 && (
              <div className="intro">
                <span className="eyebrow">3GPP Release 18 · TS 23.501 v18.12.0 · TS 24.501 v18.14.0</span>
                <h1>Ask the 5G specs</h1>
                <p>
                  Answers come only from <b>TS 23.501</b> (system architecture) and <b>TS 24.501</b> (NAS
                  protocol). Every statement cites its section. When the documents don't cover a question,
                  the assistant says so.
                </p>
                <div className="rule-label"><span>Try a question</span></div>
                <div className="examples">
                  {EXAMPLES.map((e) => (
                    <button key={e.q} type="button" className={`example tone-${e.tone}`} onClick={() => submit(e.q)}>
                      <span className="example-kind">{e.kind}</span>
                      <span className="example-q">{e.q}</span>
                      <span className="example-go" aria-hidden>→</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {turns.map((t) => (
              <TurnView key={t.id} turn={t} active={t.id === current?.id} activeCitation={activeCitation}
                modelLabel={modelLabel}
                onSelect={(id) => setActiveTurn(id)}
                onCite={(id, c) => { setActiveTurn(id); setActiveCitation(c); if (isMobile()) setSourcesOpen(true) }}
                onShowSources={(id) => { setActiveTurn(id); setActiveCitation(null); setSourcesOpen(true) }} />
            ))}
          </div>

          <form className="composer" onSubmit={(e) => { e.preventDefault(); submit(input) }}>
            <textarea
              ref={inputRef}
              autoFocus
              aria-label="Your question"
              value={input}
              rows={1}
              placeholder="Ask about 5G architecture or NAS signalling…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(input) }
              }}
            />
            <button type="submit" className="btn primary" disabled={busy || !input.trim()}>
              {busy ? 'Working…' : 'Ask'}
            </button>
          </form>
        </section>

        <Sources turn={current} activeCitation={current?.id === activeTurn ? activeCitation : null}
          onCite={(c) => setActiveCitation(c)} open={sourcesOpen} onClose={() => setSourcesOpen(false)} />
      </main>
    </div>
  )
}
