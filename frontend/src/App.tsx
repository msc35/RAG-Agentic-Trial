import { useState, useRef, KeyboardEvent } from 'react'
import { ask, askStream } from './api'
import type { AskResponse, Source } from './types'

type Mode = 'agent' | 'stream'

interface Result {
  answer: string
  sources: Source[]
  meta?: Pick<AskResponse, 'latency_ms' | 'iterations' | 'tool_calls_made'>
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState<Mode>('agent')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)
  const startRef = useRef<number>(0)

  const handleSubmit = async () => {
    const q = question.trim()
    if (!q || loading) return

    setLoading(true)
    setError(null)
    setResult(null)
    startRef.current = Date.now()

    try {
      if (mode === 'agent') {
        const data = await ask(q)
        setResult({
          answer: data.answer,
          sources: data.sources,
          meta: {
            latency_ms: data.latency_ms,
            iterations: data.iterations,
            tool_calls_made: data.tool_calls_made,
          },
        })
      } else {
        // Streaming: build answer incrementally as tokens arrive.
        let accumulated = ''
        setResult({ answer: '', sources: [] })

        for await (const event of askStream(q)) {
          if (event.token !== undefined) {
            accumulated += event.token
            setResult(prev => ({ ...(prev ?? { sources: [] }), answer: accumulated }))
          }
          if (event.sources) {
            setResult(prev => ({ ...(prev ?? { answer: accumulated }), sources: event.sources! }))
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit()
  }

  const elapsedMs = result?.meta?.latency_ms ?? (loading ? Date.now() - startRef.current : null)

  return (
    <div className="layout">
      <header className="header">
        <h1>Agentic RAG</h1>
        <p>Ask questions about the ingested technical documents</p>
      </header>

      <main className="main">
        {/* Mode toggle */}
        <div className="mode-toggle">
          <button
            className={mode === 'agent' ? 'active' : ''}
            onClick={() => setMode('agent')}
            disabled={loading}
          >
            Agent loop
          </button>
          <button
            className={mode === 'stream' ? 'active' : ''}
            onClick={() => setMode('stream')}
            disabled={loading}
          >
            Streaming
          </button>
        </div>

        {/* Input */}
        <div className="input-group">
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g. How does MentorNet select training samples?&#10;&#10;⌘ + Enter to submit"
            rows={4}
            disabled={loading}
            autoFocus
          />
          <button
            className="submit"
            onClick={handleSubmit}
            disabled={loading || !question.trim()}
          >
            {loading ? <span className="spinner" /> : 'Ask'}
          </button>
        </div>

        {/* Error */}
        {error && <div className="error-box">{error}</div>}

        {/* Result */}
        {result && (
          <div className="result-card">
            {/* Answer */}
            <section className="answer">
              {result.answer
                ? result.answer.split('\n').map((line, i) => (
                    <p key={i}>{line}</p>
                  ))
                : loading && <span className="cursor-blink">▌</span>}
            </section>

            {/* Sources */}
            {result.sources.length > 0 && (
              <section className="sources">
                <h3>Sources</h3>
                <ul>
                  {result.sources.map((s, i) => (
                    <li key={i}>
                      <span className="source-file">{s.filename}</span>
                      <span className="source-page">p.{s.page}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {/* Meta */}
            {result.meta && (
              <section className="meta">
                <span>⏱ {result.meta.latency_ms.toFixed(0)} ms</span>
                <span>· {result.meta.iterations} iteration{result.meta.iterations !== 1 ? 's' : ''}</span>
                <span>· {result.meta.tool_calls_made.length} tool call{result.meta.tool_calls_made.length !== 1 ? 's' : ''}</span>
                {result.meta.tool_calls_made.length > 0 && (
                  <details className="tool-calls">
                    <summary>show tool calls</summary>
                    <ol>
                      {result.meta.tool_calls_made.map((tc, i) => (
                        <li key={i}><code>{tc}</code></li>
                      ))}
                    </ol>
                  </details>
                )}
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
