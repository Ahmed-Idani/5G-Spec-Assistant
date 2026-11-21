import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Hit } from '../api'
import type { Turn } from '../types'

interface Props {
  turn: Turn | null
  activeCitation: string | null
  onCite: (citation: string) => void
  open: boolean          // mobile: shown as a full-screen sheet
  onClose: () => void
}

/** Chunks separate paragraphs with single newlines, which Markdown would merge into one paragraph.
 *  Give each line its own paragraph, but keep consecutive table rows together. Table cells use
 *  ' <br> ' as an internal line separator (see ingest/parse_docx.py). */
function passageMarkdown(text: string): string {
  const lines = text.replace(/\s*<br>\s*/g, ' · ').split('\n')
  return lines.reduce((out, line, i) => {
    if (i === 0) return line
    const tableRow = line.startsWith('|') && lines[i - 1].startsWith('|')
    return out + (tableRow ? '\n' : '\n\n') + line
  }, '')
}

function SourceItem({ hit, index, cited, active, reranked, top, onCite }: {
  hit: Hit; index: number; cited: boolean; active: boolean; reranked: boolean; top: number
  onCite: (c: string) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLLIElement>(null)

  useEffect(() => {
    if (active) {
      setOpen(true)
      ref.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [active])

  // Reranker scores are relevance in [0, 1]; otherwise show relative to the best hit.
  const rel = Math.max(0, Math.min(1, reranked ? hit.score : hit.score / (top || 1)))
  const ranks = [
    hit.ranks.vector && `vec ${hit.ranks.vector}`,
    hit.ranks.bm25 && `bm25 ${hit.ranks.bm25}`,
  ].filter(Boolean).join(' · ')

  return (
    <li ref={ref} className={`source ${active ? 'active' : ''}`}>
      <button type="button" className="source-head" aria-expanded={open}
        onClick={() => { setOpen(!open); onCite(hit.citation) }}>
        <span className="source-index mono">{index}</span>
        <span className="source-main">
          <span className={`source-cite mono ${hit.spec.includes('23.501') ? 's23' : 's24'}`}>
            {hit.citation.replace('TS ', '')}
            {hit.n_parts > 1 && <span className="muted"> · {hit.part + 1}/{hit.n_parts}</span>}
          </span>
          <span className="source-title">{hit.title}</span>
          <span className="source-path">{hit.path}</span>
        </span>
        <span className="source-score mono">
          {cited && <span className="cited">cited</span>}
          <span className="score-value">{hit.score.toFixed(3)}</span>
          <span className="score-bar"><span style={{ width: `${rel * 100}%` }} /></span>
        </span>
      </button>
      {open && (
        <div className="source-body">
          {ranks && <div className="mono small muted">{ranks} → final {hit.ranks.final}</div>}
          <div className="passage">
            <Markdown remarkPlugins={[remarkGfm]}>{passageMarkdown(hit.text)}</Markdown>
          </div>
        </div>
      )}
    </li>
  )
}

export function Sources({ turn, activeCitation, onCite, open, onClose }: Props) {
  const head = (extra?: React.ReactNode) => (
    <header className="pane-head">
      <span className="label">Sources</span>
      <span className="pane-head-right">
        {extra}
        <button type="button" className="link sheet-close" onClick={onClose}>Close</button>
      </span>
    </header>
  )

  if (!turn || !turn.hits) {
    return (
      <aside className={`sources ${open ? 'open' : ''}`} aria-label="Sources">
        {head()}
        <p className="muted small pad">
          Retrieved passages appear here. Click a citation in an answer to open the exact section.
        </p>
      </aside>
    )
  }
  const cited = new Set(turn.answer?.citations ?? [])
  const top = Math.max(...turn.hits.map((h) => h.score))
  const firstActive = turn.hits.findIndex((h) => h.citation === activeCitation)
  const irrelevant = turn.answer?.refused && turn.answer.refusal_reason?.startsWith('best rerank')

  return (
    <aside className={`sources ${open ? 'open' : ''}`} aria-label="Sources">
      {head(
        <span className="mono small muted">
          {turn.hits.length} passages{turn.retrievalTime !== undefined && ` · ${turn.retrievalTime.toFixed(1)} s`}
        </span>,
      )}
      <p className="pane-question small muted">{turn.question}</p>
      {irrelevant && (
        <p className="pane-note small">
          None of these passages is relevant: the best reranker score ({top.toFixed(3)}) is below the refusal
          threshold ({turn.settings.threshold.toFixed(2)}), so the model was not called. They are shown only for
          transparency.
        </p>
      )}
      <ol className={`source-list ${irrelevant ? 'dimmed' : ''}`}>
        {turn.hits.map((h, i) => (
          <SourceItem key={h.id} hit={h} index={i + 1} cited={cited.has(h.citation)}
            active={i === firstActive} reranked={turn.reranked ?? true} top={top} onCite={onCite} />
        ))}
      </ol>
    </aside>
  )
}
