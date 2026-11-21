import type { Config, Settings } from '../api'

interface Props {
  config: Config | null
  settings: Settings
  onChange: (s: Settings) => void
  onClose: () => void
}

export function SettingsPanel({ config, settings, onChange, onClose }: Props) {
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) => onChange({ ...settings, [key]: value })

  return (
    <div className="settings" role="dialog" aria-label="Settings">
      <div className="settings-head">
        <span className="label">Settings</span>
        <button type="button" className="link" onClick={onClose}>Close</button>
      </div>

      <label className="field">
        <span>Model</span>
        <select value={settings.model} onChange={(e) => set('model', e.target.value)}>
          {(config?.models ?? []).map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </label>

      <label className="field">
        <span>Passages for the model <b className="mono">{settings.k}</b></span>
        <input type="range" min={1} max={10} value={settings.k} onChange={(e) => set('k', +e.target.value)} />
      </label>

      <fieldset className="field">
        <legend>Retrieval</legend>
        <label className="check">
          <input type="checkbox" checked={settings.hybrid} onChange={(e) => set('hybrid', e.target.checked)} />
          <span>Hybrid search <em>vector + BM25, fused with RRF</em></span>
        </label>
        <label className="check">
          <input type="checkbox" checked={settings.rerank} onChange={(e) => set('rerank', e.target.checked)} />
          <span>Reranker <em>bge-reranker-base cross-encoder</em></span>
        </label>
        <label className="check">
          <input type="checkbox" checked={settings.route} onChange={(e) => set('route', e.target.checked)} />
          <span>Routing <em>restrict search to the relevant specification</em></span>
        </label>
      </fieldset>

      <label className={`field ${settings.rerank ? '' : 'disabled'}`}>
        <span>Refusal threshold <b className="mono">{settings.threshold.toFixed(2)}</b></span>
        <input type="range" min={0} max={0.2} step={0.01} value={settings.threshold} disabled={!settings.rerank}
          onChange={(e) => set('threshold', +e.target.value)} />
        <em>Below this reranker score, the assistant declines without calling the model.</em>
      </label>
    </div>
  )
}
