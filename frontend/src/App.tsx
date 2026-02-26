import { useState, useCallback } from 'react'
import { Routes, Route, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import TrialDetail from './pages/TrialDetail'

interface Trial {
  nct_id: string
  title: string
  conditions?: string
  contacts?: string
  locations?: string
  gainesville_active_recruiting?: string
  phase?: string
  criteria_overview?: string
  distance_miles?: number
  distance_label?: string
  sex?: string
  minimum_age?: string
  maximum_age?: string
  [key: string]: unknown
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  trials?: Trial[]
  patientSummary?: string
}

async function chat(
  messages: { role: string; content: string }[],
  patientZip?: string | null
): Promise<{ reply: string; trials: Trial[] }> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, patient_zip: patientZip || undefined }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || res.statusText || 'Chat failed')
  }
  return res.json()
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [patientZip, setPatientZip] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const getPatientSummary = useCallback((msgs: Message[]) => {
    return msgs.filter((m) => m.role === 'user').map((m) => m.content).join('\n')
  }, [])

  const sendMessage = useCallback(async () => {
    const msg = input.trim()
    if (loading) return

    setInput('')
    setError(null)
    const updated = [...messages, { role: 'user' as const, content: msg }]
    if (msg) setMessages(updated)
    setLoading(true)

    try {
      const apiMessages = (msg ? updated : messages).map((m) => ({ role: m.role, content: m.content }))
      const { reply, trials } = await chat(apiMessages, patientZip || null)
      setMessages((prev) => {
        if (!msg) return [{ role: 'assistant', content: reply, trials: [] }]
        return [...prev, { role: 'assistant', content: reply, trials, patientSummary: getPatientSummary(prev) }]
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }, [input, loading, messages, getPatientSummary])

  return (
    <Routes>
      <Route path="/trial/:nctId" element={<TrialDetail />} />
      <Route path="/" element={
    <div className="h-screen flex flex-col bg-surface-50 overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 bg-white border-b border-surface-200 shadow-sm">
        <div className="w-full max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-xl font-semibold text-slate-800 tracking-tight">
            Trial Finder
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Clinical trial assistant for physicians — Gainesville, Dallas, Minneapolis, New York, Seattle & Los Angeles
          </p>
        </div>
      </header>

      {/* Main chat area */}
      <main className="flex-1 min-h-0 overflow-y-auto">
        <div className="w-full max-w-7xl mx-auto px-6 py-6">
        {messages.length === 0 && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center max-w-md">
              <p className="text-slate-500 text-base">
                I'll ask a few questions to narrow down the best trials. Enter the patient's zip code for distance-based matching (1–25, 26–50, 51–100, 100+ miles).
              </p>
              <p className="text-slate-400 text-sm mt-4">
                Click below to start, or type your patient's condition directly.
              </p>
              <button
                onClick={() => sendMessage()}
                disabled={loading}
                className="mt-6 px-6 py-3 rounded-xl bg-primary-500 text-white font-medium hover:bg-primary-600 disabled:opacity-50 transition-colors"
              >
                Start conversation
              </button>
            </div>
          </div>
        )}

        <div className="space-y-6">
          {messages.map((m, i) => (
            <div key={i} className="space-y-3">
              <div
                className={
                  m.role === 'user'
                    ? 'ml-auto max-w-[85%] rounded-2xl rounded-tr-md bg-primary-500 text-white px-4 py-3'
                    : 'mr-auto max-w-[85%] rounded-2xl rounded-tl-md bg-white border border-surface-200 shadow-sm px-4 py-3'
                }
              >
                {m.role === 'user' ? (
                  <p className="text-[15px] leading-relaxed">{m.content}</p>
                ) : (
                  <div className="prose prose-slate prose-sm max-w-none">
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                )}
              </div>

              {m.role === 'assistant' && m.trials && m.trials.length > 0 && (
                <div className="space-y-2 pl-1">
                  <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                    Matched trials (top 3)
                  </p>
                  <div className="grid gap-3">
                    {m.trials.map((t) => (
                      <Link
                        key={t.nct_id}
                        to={`/trial/${t.nct_id}`}
                        className="block bg-white rounded-xl border border-surface-200 shadow-sm p-4 hover:border-primary-300 hover:shadow-md transition-all cursor-pointer"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <p className="font-semibold text-slate-800 text-sm flex-1">
                            {t.title || t.nct_id}
                          </p>
                          {t.distance_label != null && (
                            <span className="text-xs font-medium text-primary-600 shrink-0">
                              {t.distance_label}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-slate-500 mt-0.5">{t.nct_id}</p>
                        {t.criteria_overview && (
                          <div className="text-xs text-slate-600 mt-2 space-y-0.5">
                            {(t.criteria_overview as string).split(/\s*\|\s*/).filter(Boolean).map((line, i) => (
                              <p key={i} className="line-clamp-1">{line.trim()}</p>
                            ))}
                          </div>
                        )}
                        {t.locations && (
                          <div className="text-xs text-slate-500 mt-2 space-y-0.5">
                            {(t.locations as string).split(/\s*\|\s*/).filter(Boolean).map((loc, i) => (
                              <p key={i} className="truncate" title={loc.trim()}>📍 {loc.trim()}</p>
                            ))}
                          </div>
                        )}
                        <p className="mt-3 text-xs font-medium text-primary-600">
                          View full details →
                        </p>
                      </Link>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="mr-auto max-w-[85%] rounded-2xl rounded-tl-md bg-white border border-surface-200 shadow-sm px-4 py-3">
              <div className="flex gap-1.5">
                <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 rounded-full bg-slate-300 animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 px-4 py-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
            {error}
          </div>
        )}
        </div>
      </main>

      {/* Input */}
      <div className="flex-shrink-0 bg-white border-t border-surface-200 py-4">
        <div className="w-full max-w-7xl mx-auto px-6">
          <div className="flex flex-wrap gap-3 items-end">
            <input
              type="text"
              value={patientZip}
              onChange={(e) => setPatientZip(e.target.value.replace(/\D/g, '').slice(0, 5))}
              placeholder="Zip"
              maxLength={5}
              disabled={loading}
              className="w-20 rounded-xl border border-surface-200 bg-surface-50 px-3 py-3 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-400 disabled:opacity-60"
              title="Patient zip code for distance calculation"
            />
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              placeholder="e.g. Heart attack, 65-year-old male…"
              disabled={loading}
              className="flex-1 min-w-[200px] rounded-xl border border-surface-200 bg-surface-50 px-4 py-3 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-400 focus:border-transparent disabled:opacity-60"
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="px-6 py-3 rounded-xl bg-primary-500 text-white font-medium hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Send
            </button>
          </div>
        </div>
      </div>

    </div>
      } />
    </Routes>
  )
}

export default App
