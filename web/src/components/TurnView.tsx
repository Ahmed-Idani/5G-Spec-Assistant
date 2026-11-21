import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { citationFromHref, linkCitations } from '../citations'
import type { Turn } from '../types'

interface Props {
  turn: Turn
  active: boolean
  activeCitation: string | null
  onCite: (turnId: number, citation: string) => void
  onSelect: (turnId: number) => void
  onShowSources: (turnId: number) => void
  modelLabel: (id: string) => string
}

const STEPS: { key: 'routing' | 'retrieving' | 'generating'; label: string }[] = [
  { key: 'routing', label: 'route' },
  { key: 'retrieving', label: 'retrieve' },
  { key: 'generating', label: 'generate' },
]
const ORDER = ['routing', 'retrieving', 'generating', 'done']

function Progress({ turn }: { turn: Turn }) {
  const at = ORDER.indexOf(turn.stage)
  return (
    <div className="progress mono">
      {STEPS.map((s, i) => {
        const state = turn.stage === 'error' ? 'idle' : i < at ? 'done' : i === at ? 'run' : 'idle'
        let detail = ''
        if (s.key === 'routing' && turn.route) detail = turn.route.specs.map((x) => x.replace('TS ', '')).join(' + ')
        if (s.key === 'retrieving' && turn.hits) detail = `${turn.hits.length} passages`
        return (
          <span key={s.key} className={`step ${state}`}>
            {s.label}
            {detail && <span className="step-detail"> {detail}</span>}
          </span>
        )
      })}
    </div>
  )
}

export function TurnView({ turn, active, activeCitation, onCite, onSelect, onShowSources, modelLabel }: Props) {
  const a = turn.answer
  const retrieved = new Set(turn.hits?.map((h) => h.citation) ?? [])

  return (
    <article className={`turn ${active ? 'active' : ''}`} data-turn={turn.id} onClick={() => onSelect(turn.id)}>
      <h2 className="question">{turn.question}</h2>

      {turn.stage !== 'done' && turn.stage !== 'error' && <Progress turn={turn} />}

      {turn.stage === 'error' && <div className="notice error">Error: {turn.error}</div>}

      {a && a.refused && (
        <div className="notice refusal">
          <p>{a.text}</p>
          <span className="mono small muted">
            {a.refusal_reason?.startsWith('best rerank')
              ? 'No passage in TS 23.501 or TS 24.501 is relevant to this question.'
              : 'The retrieved passages do not contain this information.'}
          </span>
        </div>
      )}

      {a && !a.refused && (
        <div className="answer">
          <Markdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) => {
                const cite = citationFromHref(href)
                if (!cite) return <a href={href} target="_blank" rel="noreferrer">{children}</a>
                const ok = retrieved.has(cite)
                return (
                  <button
                    type="button"
                    className={`cite mono ${ok ? '' : 'invalid'} ${cite.includes('23.501') ? 's23' : 's24'} ${
                      active && activeCitation === cite ? 'on' : ''
                    }`}
                    title={ok ? `Show ${cite}` : `${cite} was not among the retrieved passages`}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (ok) onCite(turn.id, cite)
                    }}
                  >
                    {children}
                  </button>
                )
              },
            }}
          >
            {linkCitations(a.text)}
          </Markdown>
        </div>
      )}

      {a && (
        <footer className="meta mono">
          <span>{modelLabel(a.model)}</span>
          {turn.route && <span>{turn.route.specs.map((s) => s.replace('TS ', '')).join(' + ')}</span>}
          {a.timings.retrieval !== undefined && <span>retrieval {a.timings.retrieval.toFixed(1)} s</span>}
          {a.timings.llm !== undefined && <span>generation {a.timings.llm.toFixed(1)} s</span>}
          {a.citations.length > 0 && (
            <span className={a.invalid_citations.length ? 'warn' : ''}>
              {a.citations.length - a.invalid_citations.length}/{a.citations.length} citations verified
            </span>
          )}
          {turn.hits && turn.hits.length > 0 && (
            <button type="button" className="link sources-btn"
              onClick={(e) => { e.stopPropagation(); onShowSources(turn.id) }}>
              {turn.hits.length} sources →
            </button>
          )}
        </footer>
      )}
    </article>
  )
}
