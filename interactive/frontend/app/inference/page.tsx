'use client'

import { useState, useEffect, useRef, DragEvent, KeyboardEvent } from 'react'
import Link from 'next/link'
import styles from './inference.module.css'
import ThemeToggle from '../../components/ThemeToggle'

type Model = {
  model_id: string
  run_id: string
  strategy: string
  display_name: string
  description: string
  target: string
  task: string
  classes: string[]
  n_models: number
  calibrated: boolean
  submission_filename?: string
}

type Prediction = {
  sample_id: string
  prediction: number
  label: string
  probability: number
  confidence: number
}

type FeatureExplanation = {
  feature: string
  importance: number
  std: number
  n_models: number
  abs_importance?: number
  support?: number
  direction?: string
  method?: string
  value?: number
  baseline?: number
}

type InteractionEdge = {
  source: string
  target: string
  strength: number
  abs_strength?: number
  support?: number
  direction?: string
  method?: string
}

type SampleExplanation = {
  sample_id: string
  prediction: number
  label: string
  probability: number
  top_features: FeatureExplanation[]
  shap_features?: FeatureExplanation[]
  lime_features?: FeatureExplanation[]
  interactions?: InteractionEdge[]
  interaction_method?: string
  n_models_explained: number
  explanation_method?: string
  explanation_basis?: string
}

type ChatMessage = { role: 'user' | 'assistant'; content: string }
type PredictionRow = Record<string, string | number | null>
type Run = { id: string; target: string; task: string; classes: string[]; strategies: string[] }

type RawExplanation = {
  sample_id: string
  class_label: string
  probability: number
  method: string
  scope: string
  features: { name: string; value: number; abundance: number }[]
}

const safeToken = (label: string) => label.replace(/[^0-9A-Za-z]+/g, '_').replace(/^_+|_+$/g, '') || 'class'
const detectFormat = (content: string) => /^sample_id(?:,|\t)/i.test(content.trimStart()) ? 'wide' : 'matrix'
const prefixOf = (strategy: string) => strategy.toLowerCase().replace(/-/g, '_')

const responseData = async <T,>(response: Response): Promise<T> => {
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`)
  return data as T
}

const toPredictions = (rows: PredictionRow[], model: Model): Prediction[] => {
  const prefix = prefixOf(model.strategy)
  if (rows.length && !(`${prefix}_predicted_class` in rows[0]) && !(`${prefix}_prediction` in rows[0])) return []
  return rows.map(row => {
    const sample_id = String(row.sample_id)
    if (model.task !== 'classification') {
      const value = Number(row[`${prefix}_prediction`])
      return { sample_id, prediction: value, label: Number.isFinite(value) ? value.toPrecision(4) : '—', probability: 0, confidence: 0 }
    }
    const label = String(row[`${prefix}_predicted_class`] ?? '')
    const probability = Number(row[`${prefix}_probability_${safeToken(label)}`] ?? 0)
    return {
      sample_id,
      prediction: model.classes.indexOf(label),
      label,
      probability,
      confidence: model.classes.length === 2 ? Math.abs(probability - 0.5) * 2 : probability,
    }
  })
}

export default function InferencePage() {
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [dataFile, setDataFile] = useState<File | null>(null)
  const [results, setResults] = useState<Prediction[] | null>(null)
  const [explanations, setExplanations] = useState<SampleExplanation[] | null>(null)
  const [selectedSample, setSelectedSample] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [explaining, setExplaining] = useState(false)
  const [showShap, setShowShap] = useState(true)
  const [showLime, setShowLime] = useState(true)
  const [downloadingSubmission, setDownloadingSubmission] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [chatModel, setChatModel] = useState('')
  const [chatModelOptions, setChatModelOptions] = useState<string[]>([])
  const [chatConfigStatus, setChatConfigStatus] = useState('unknown')
  const [chatConfigDetail, setChatConfigDetail] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)

  const currentSelection = () => models.find(model => model.model_id === selectedModel)

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('/api/runs')
        const runs = await responseData<Run[]>(response)
        const found = runs.flatMap(run => run.strategies.map(strategy => ({
          model_id: `${run.id}:${strategy}`,
          run_id: run.id,
          strategy,
          display_name: `${run.target} · ${strategy}`,
          description: `Fitted ${strategy} model from the current experiment`,
          target: run.target,
          task: run.task,
          classes: run.classes.map(String),
          n_models: 1,
          calibrated: false,
        })))
        setModels(found)
        if (found.length) setSelectedModel(found[0].model_id)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load inference models')
      }
    }
    fetchModels()
    fetchChatConfig()
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  useEffect(() => {
    const model = currentSelection()
    setExplanations(null)
    setSelectedSample(null)
    if (!model || dataFile) {
      setResults(null)
      return
    }
    let active = true
    fetch(`/api/runs/${model.run_id}/predictions`)
      .then(responseData<PredictionRow[]>)
      .then(rows => {
        if (!active) return
        const converted = toPredictions(rows, model)
        setResults(converted.length ? converted : null)
        setSelectedSample(converted[0]?.sample_id || null)
      })
      .catch(() => { if (active) setResults(null) })
    return () => { active = false }
  }, [selectedModel, models, dataFile])

  const fetchChatConfig = async () => {
    try {
      const response = await fetch('/api/chat/models')
      const data = await responseData<{ models: string[] }>(response)
      const available = data.models.filter(Boolean)
      setChatModelOptions(available)
      setChatModel(previous => available.includes(previous) ? previous : (available[0] || ''))
      setChatConfigStatus(available.length ? 'ok' : 'unavailable')
      setChatConfigDetail(available.length ? null : 'No local Ollama models found. Install a local model to enable chat.')
    } catch (err) {
      setChatConfigStatus('unavailable')
      setChatConfigDetail('The local Ollama service is unavailable.')
      setChatModelOptions([])
      setChatModel('')
    }
  }

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (!file) return
    if (!/\.(tsv|csv|txt)$/i.test(file.name)) {
      setError('Please upload a TSV, CSV, or TXT file')
      return
    }
    setDataFile(file)
    setError(null)
  }

  const handleRunInference = async () => {
    const model = currentSelection()
    if (!dataFile || !model) return
    setLoading(true)
    setError(null)
    try {
      const content = await dataFile.text()
      const response = await fetch(`/api/runs/${model.run_id}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'uploaded', file: content, format: detectFormat(content), strategy: model.strategy }),
      })
      const rows = await responseData<PredictionRow[]>(response)
      const converted = toPredictions(rows, model)
      setResults(converted)
      setExplanations(null)
      setSelectedSample(converted[0]?.sample_id || null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed')
    } finally {
      setLoading(false)
    }
  }

  const handleDownloadSubmission = async () => {
    if (!results) return
    setDownloadingSubmission(true)
    const lines = [
      ['sample_id', 'prediction', 'probability'].join(','),
      ...results.map(row => [row.sample_id, row.label, String(row.probability)]
        .map(value => '"' + value.replace(/"/g, '""') + '"').join(',')),
    ]
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
    const href = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = href
    anchor.download = 'inference_predictions.csv'
    anchor.click()
    URL.revokeObjectURL(href)
    setDownloadingSubmission(false)
  }

  const upsertExplanation = (incoming: SampleExplanation[]) => {
    setExplanations(previous => {
      const byId = new Map((previous || []).map(row => [row.sample_id, row]))
      for (const row of incoming) byId.set(row.sample_id, { ...byId.get(row.sample_id), ...row })
      return [...byId.values()]
    })
  }

  const fetchExplanationForSample = async (sampleId?: string | null) => {
    const model = currentSelection()
    if (!model || !sampleId) return
    if (model.task !== 'classification') {
      setError('SHAP/LIME inference explanations are available for classification runs only.')
      return
    }
    setExplaining(true)
    setError(null)
    try {
      const methods = [showShap && 'shap', showLime && 'lime'].filter((value): value is string => Boolean(value))
      const content = dataFile ? await dataFile.text() : ''
      const requests = methods.map(async method => {
        const response = await fetch(`/api/runs/${model.run_id}/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source: dataFile ? 'uploaded' : 'saved',
            file: content,
            format: dataFile ? detectFormat(content) : 'wide',
            strategy: model.strategy,
            sample_id: sampleId,
            method,
          }),
        })
        return responseData<RawExplanation>(response)
      })
      const resolved = await Promise.allSettled(requests)
      const good = resolved.filter((r): r is PromiseFulfilledResult<RawExplanation> => r.status === 'fulfilled').map(r => r.value)
      if (!good.length) {
        const failed = resolved.find((r): r is PromiseRejectedResult => r.status === 'rejected')
        throw failed?.reason || new Error('Explanation failed')
      }
      const prediction = results?.find(row => row.sample_id === sampleId)
      const mapped: SampleExplanation = {
        sample_id: sampleId,
        prediction: prediction?.prediction ?? 0,
        label: good[0].class_label,
        probability: good[0].probability,
        n_models_explained: 1,
        explanation_basis: good[0].scope,
        top_features: [],
      }
      for (const result of good) {
        const features: FeatureExplanation[] = result.features.map(feature => ({
          feature: feature.name, importance: feature.value, std: 0, n_models: 1,
        }))
        if (result.method === 'shap') mapped.shap_features = features
        if (result.method === 'lime') mapped.lime_features = features
        mapped.top_features = mapped.shap_features || mapped.lime_features || []
      }
      upsertExplanation([mapped])
      setSelectedSample(sampleId)
      const failed = resolved.find((r): r is PromiseRejectedResult => r.status === 'rejected')
      if (failed) setError(`One explanation method failed: ${failed.reason instanceof Error ? failed.reason.message : 'unknown error'}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate explanations')
    } finally {
      setExplaining(false)
    }
  }

  const handleExplain = () => fetchExplanationForSample(selectedSample || results?.[0]?.sample_id || null)

  const handleSampleChange = (sampleId: string) => {
    setSelectedSample(sampleId)
    if (!explanations?.some(item => item.sample_id === sampleId)) fetchExplanationForSample(sampleId)
  }

  const toggleShap = () => { if (!showShap || showLime) setShowShap(!showShap) }
  const toggleLime = () => { if (!showLime || showShap) setShowLime(!showLime) }

  const buildContext = () => {
    const model = currentSelection()
    const summary = [model ? `Model: ${model.display_name}` : '']
    for (const row of results || []) summary.push(`${row.sample_id}: ${row.label}`)
    for (const item of explanations || []) {
      summary.push(`${item.sample_id} · ${item.explanation_basis || 'attribution'}`)
      for (const feature of item.top_features.slice(0, 8)) summary.push(`${feature.feature}: ${feature.importance.toFixed(5)}`)
    }
    return summary.join('\n').slice(0, 6000)
  }

  const handleSendChat = async () => {
    if (!chatInput.trim() || chatLoading || !chatModel) return
    const message = chatInput.trim()
    setChatInput('')
    setChatMessages(previous => [...previous, { role: 'user', content: message }])
    setChatLoading(true)
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: message, context: buildContext(), model: chatModel }),
      })
      const result = await responseData<{ answer: string }>(response)
      setChatMessages(previous => [...previous, { role: 'assistant', content: result.answer }])
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Local Ollama is unavailable'
      setChatMessages(previous => [...previous, { role: 'assistant', content: message }])
    } finally {
      setChatLoading(false)
    }
  }

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendChat()
    }
  }

  const currentModel = models.find(m => m.model_id === selectedModel)
  const currentExplanation = explanations?.find(e => e.sample_id === selectedSample)
  const taxonLabel = (feature: string, maxLen = 38) => {
    const raw = String(feature || '')
    const last = raw.includes('___') ? raw.split('___').pop() || raw : raw.includes('|') ? raw.split('|').pop() || raw : raw
    const cleaned = last.replace(/^[dkpcofgst]__/, '').replace(/_/g, ' ').trim() || raw
    const rankMatch = last.match(/^([dkpcofgst])__/)
    const rank = rankMatch ? `${rankMatch[1]}. ` : ''
    const label = `${rank}${cleaned}`
    return label.length > maxLen ? `${label.slice(0, maxLen - 1).trim()}…` : label
  }

  const signedValue = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(4)}`

  const renderExplanationList = (features: FeatureExplanation[] = []) => {
    const visible = features.slice(0, 12)
    const maxAbs = Math.max(...visible.map(f => Math.abs(f.importance || 0)), 1e-12)
    return (
      <div className={styles.featureImportanceList}>
        {visible.map((feat, idx) => {
          const magnitude = Math.min(1, Math.abs(feat.importance || 0) / maxAbs)
          const positive = (feat.importance || 0) >= 0
          const width = `${Math.max(1.5, magnitude * 47)}%`
          const left = positive ? '50%' : `${50 - Math.max(1.5, magnitude * 47)}%`
          return (
            <div key={`${feat.feature}-${idx}`} className={styles.featureImportanceItem}>
              <div className={styles.featureImportanceRank}>{idx + 1}</div>
              <div className={styles.featureImportanceName} title={feat.feature}>{taxonLabel(feat.feature)}</div>
              <div className={styles.featureImportanceBarContainer} aria-label={`${taxonLabel(feat.feature)} contribution ${signedValue(feat.importance || 0)}`}>
                <div className={styles.zeroLine} />
                <div
                  className={`${styles.featureImportanceBar} ${positive ? styles.supportToward : styles.supportAgainst}`}
                  style={{ width, left }}
                />
              </div>
              <div className={`${styles.featureImportanceValue} ${positive ? styles.valuePositive : styles.valueNegative}`}>
                {signedValue(feat.importance || 0)}
              </div>
            </div>
          )
        })}
        {visible.length > 0 && (
          <div className={styles.explainAxisRow} aria-hidden="true">
            <div />
            <div />
            <div className={styles.explainAxis}>
              <span className={styles.explainAxisMinus}>−</span>
              <span className={styles.explainAxisZero}>0</span>
              <span className={styles.explainAxisPlus}>+</span>
            </div>
            <div />
          </div>
        )}
      </div>
    )
  }


  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerInner}>
          <Link href="/" className={styles.logo}>
            <h1>mllabiome</h1>
            <span className={styles.subtitle}>interactive inference</span>
          </Link>
          <div className={styles.headerRight}>
            <ThemeToggle />
          </div>
        </div>
      </div>

      <div className={styles.mainLayout}>
        <aside className={styles.sidebar}>
          <div className={styles.sidebarSection}>
            <h3 className={styles.sidebarTitle}>Model</h3>
            <div className={styles.modelSelector}>
              {models.map(model => (
                <button
                  key={model.model_id}
                  className={`${styles.modelOption} ${selectedModel === model.model_id ? styles.modelOptionActive : ''}`}
                  onClick={() => setSelectedModel(model.model_id)}
                >
                  <span className={styles.modelOptionName}>{model.display_name}</span>
                  <span className={styles.modelOptionMeta}>{model.target || 'Prediction model'}</span>
                </button>
              ))}
              {models.length === 0 && (
                <div className={styles.modelInfo}>
                  <p className={styles.modelDescription}>No fitted inference models found in this experiment. Complete --stage inference first.</p>
                </div>
              )}
            </div>
            {currentModel && (
              <div className={styles.modelInfo}>
                <p className={styles.modelDescription}>Selected model: {currentModel.display_name}</p>
                {currentModel.calibrated && (
                  <span className={styles.calibratedBadge}>Calibrated</span>
                )}
              </div>
            )}
          </div>

          <div className={styles.sidebarSection}>
            <h3 className={styles.sidebarTitle}>Sample Data</h3>
            <div 
              className={`${styles.uploadCardSmall} ${isDragging ? styles.uploadCardDragging : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <label className={styles.dataUploadSmall}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".tsv,.csv,.txt"
                  onChange={(e) => {
                    setDataFile(e.target.files?.[0] || null)
                    setError(null)
                  }}
                />
                <div className={styles.uploadContentSmall}>
                  <span className={styles.uploadTitleSmall}>
                    {dataFile ? dataFile.name : 'Drop file here'}
                  </span>
                  <span className={styles.uploadHintSmall}>
                    TSV or CSV
                  </span>
                </div>
              </label>
            </div>
            
            {error && (
              <div className={styles.errorMessageSmall}>{error}</div>
            )}

            <div className={styles.buttonGroup}>
              <button
                className={styles.runButtonSmall}
                onClick={handleRunInference}
                disabled={!dataFile || !selectedModel || loading}
              >
                {loading ? 'Running...' : 'Run Inference'}
              </button>
              
              <button
                className={`${styles.runButtonSmall} ${styles.explainButtonSmall}`}
                onClick={handleExplain}
                disabled={!selectedModel || explaining || (!dataFile && !results)}
              >
                {explaining ? 'Explaining...' : 'Explain'}
              </button>
            </div>
          </div>

          <div className={styles.sidebarSection}>
            <h3 className={styles.sidebarTitle}>Status</h3>
            <div className={styles.statusList}>
              <div className={styles.statusItem}>
                <span className={styles.statusDot} style={{ backgroundColor: models.length > 0 ? '#10b981' : '#ef4444' }} />
                <span>{models.length > 0 ? 'Ready' : 'No inference model'}</span>
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusDot} style={{ backgroundColor: dataFile ? '#10b981' : '#6b7280' }} />
                <span>{dataFile ? 'Data loaded' : 'No data'}</span>
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusDot} style={{ backgroundColor: results ? '#10b981' : '#6b7280' }} />
                <span>{results ? `${results.length} predictions` : 'No results'}</span>
              </div>
            </div>
          </div>
        </aside>

        <main className={styles.mainContent}>
          <div className={styles.contentGrid}>
            <div className={styles.resultsPanel}>
              {results && results.length > 0 && (
                <section className={styles.section}>
                  <div className={styles.sectionHeader}>
                    <h2>Predictions</h2>
                    <div className={styles.sectionHeaderActions}>
                      <span className={styles.resultCount}>{results.length} samples</span>
                      <button
                        type="button"
                        className={styles.downloadCsvButton}
                        onClick={handleDownloadSubmission}
                        disabled={!results || downloadingSubmission}
                      >
                        {downloadingSubmission ? 'Preparing...' : 'Download CSV'}
                      </button>
                    </div>
                  </div>
                  <div className={styles.resultsTable}>
                    <div className={styles.tableHeader}>
                      <div className={styles.tableCell}>Sample</div>
                      <div className={styles.tableCell}>Prediction</div>
                      <div className={styles.tableCell}>Probability</div>
                      <div className={styles.tableCell}>Confidence</div>
                    </div>
                    {results.map((pred, idx) => (
                      <div key={idx} className={styles.tableRow}>
                        <div className={`${styles.tableCell} ${styles.sampleIdCell}`} title={pred.sample_id}>
                          <span className={styles.sampleIdText}>{pred.sample_id}</span>
                        </div>
                        <div className={styles.tableCell}>
                          <span className={`${styles.predictionLabel} ${pred.prediction === 1 ? styles.positive : styles.negative}`}>
                            {pred.label}
                          </span>
                        </div>
                        <div className={styles.tableCell}>
                          {currentModel?.task === 'classification' ? `${(pred.probability * 100).toFixed(1)}%` : '—'}
                        </div>
                        <div className={styles.tableCell}>
                          <span className={styles.confidence}>{currentModel?.task === 'classification' ? `${(pred.confidence * 100).toFixed(0)}%` : '—'}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {explanations && explanations.length > 0 && (
                <section className={`${styles.section} ${styles.explainabilitySection}`}>
                  <div className={styles.sectionHeader}>
                    <h2>Explainability</h2>
                    <span className={styles.resultCount}>Screened local attribution</span>
                    <div className={styles.explainabilityControls}>
                      <div className={styles.methodToggleGroup} aria-label="Explainability methods">
                        <button
                          type="button"
                          className={`${styles.methodToggle} ${showShap ? styles.methodToggleActive : ''}`}
                          onClick={toggleShap}
                          aria-pressed={showShap}
                        >
                          SHAP
                        </button>
                        <button
                          type="button"
                          className={`${styles.methodToggle} ${showLime ? styles.methodToggleActive : ''}`}
                          onClick={toggleLime}
                          aria-pressed={showLime}
                        >
                          LIME
                        </button>
                      </div>
                      <div className={styles.sampleSelectWrap}>
                        <label className={styles.sampleSelectLabel} htmlFor="explain-sample-select">Sample</label>
                        <select
                          id="explain-sample-select"
                          className={styles.sampleSelect}
                          value={selectedSample || currentExplanation?.sample_id || ''}
                          onChange={(e) => handleSampleChange(e.target.value)}
                          title={selectedSample || undefined}
                        >
                          {(results && results.length > 0 ? results : explanations).map((row) => (
                            <option key={row.sample_id} value={row.sample_id}>
                              {row.sample_id}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  {currentExplanation && (
                    <div className={styles.explanationCard}>
                      <div className={`${styles.explainabilityGrid} ${showShap && showLime ? styles.explainabilityGridTwo : styles.explainabilityGridOne}`}>
                        {showShap && (
                          <div className={styles.supportPanel}>
                            <div className={styles.panelHeadingRow}>
                              <h4 className={styles.subSectionTitle}>SHAP</h4>
                            </div>
                            {renderExplanationList(currentExplanation.shap_features || currentExplanation.top_features || [])}
                          </div>
                        )}

                        {showLime && (
                          <div className={styles.supportPanel}>
                            <div className={styles.panelHeadingRow}>
                              <h4 className={styles.subSectionTitle}>LIME</h4>
                            </div>
                            {renderExplanationList(currentExplanation.lime_features || [])}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </section>
              )}

              {!results && (
                <section className={styles.section}>
                  <div className={styles.emptyState}>
                    <h3>No Results</h3>
                    <p>Upload a microbiome profile or inspect existing inference predictions. Complete <code>--stage inference</code> first if no fitted model is available.</p>
                  </div>
                </section>
              )}
            </div>

            <div className={styles.chatPanel}>
              <div className={styles.chatHeader}>
                <h3>Inference Assistant</h3>
                <span className={styles.chatMeta}>Ollama · {chatModel || 'not configured'}</span>
              </div>
              <div className={styles.chatConfigBar}>
                <label className={styles.chatModelLabel} htmlFor="ollama-model">Model</label>
                <select
                  id="ollama-model"
                  value={chatModel}
                  onChange={(e) => setChatModel(e.target.value)}
                  className={styles.chatModelInput}
                  disabled={chatLoading || chatModelOptions.length === 0}
                >
                  {chatModelOptions.length === 0 && <option value="">No local models</option>}
                  {chatModelOptions.map(name => <option key={name} value={name}>{name}</option>)}
                </select>
                <button
                  type="button"
                  className={styles.chatRefreshButton}
                  onClick={fetchChatConfig}
                  disabled={chatLoading}
                  title="Refresh Ollama model list"
                >
                  Refresh
                </button>
              </div>
              {chatConfigStatus !== 'ok' && chatConfigDetail && (
                <div className={styles.chatConfigWarning}>{chatConfigDetail}</div>
              )}
              
              <div className={styles.chatMessages}>
                {chatMessages.length === 0 && (
                  <div className={styles.chatWelcome}>
                    {explanations ? (
                      <>
                        <p>Results loaded. Ask about the feature attributions.</p>
                        <p className={styles.chatHint}>Example: Why is Bacteroides important for this prediction?</p>
                      </>
                    ) : results ? (
                      <>
                        <p>Predictions ready. Click Explain for feature attribution, or ask about the prediction summary.</p>
                        <p className={styles.chatHint}>After Explain completes, ask about specific bacterial genera or contribution direction.</p>
                      </>
                    ) : (
                      <>
                        <p>Run inference to get started.</p>
                        <p className={styles.chatHint}>Upload data and click Run Inference.</p>
                      </>
                    )}
                  </div>
                )}
                {chatMessages.map((msg, idx) => (
                  <div key={idx} className={`${styles.chatMessage} ${styles[msg.role]}`}>
                    <div className={styles.messageContent}>{msg.content}</div>
                  </div>
                ))}
                {chatLoading && (
                  <div className={`${styles.chatMessage} ${styles.assistant}`}>
                    <div className={styles.messageContent}>
                      <span className={styles.typingIndicator}>...</span>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              
              <div className={styles.chatInputContainer}>
                <textarea
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about your results..."
                  className={styles.chatInput}
                  rows={2}
                  disabled={chatLoading || !chatModel}
                />
                <button 
                  onClick={handleSendChat} 
                  className={styles.chatSendButton}
                  disabled={!chatInput.trim() || chatLoading || !chatModel}
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
