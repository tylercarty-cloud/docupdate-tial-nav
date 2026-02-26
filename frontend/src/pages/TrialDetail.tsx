import { useEffect, useState, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'

const DEFAULT_CREDENTIALS = 'Name: Pranav, Guduru\nOccupation: Doctor'
const CT_GOV_BASE = 'https://clinicaltrials.gov/study/'

interface Trial {
  nct_id: string
  title: string
  overall_status?: string
  gainesville_recruitment_status?: string
  gainesville_active_recruiting?: string
  los_angeles_recruitment_status?: string
  los_angeles_active_recruiting?: string
  dallas_recruitment_status?: string
  dallas_active_recruiting?: string
  minneapolis_recruitment_status?: string
  minneapolis_active_recruiting?: string
  new_york_recruitment_status?: string
  new_york_active_recruiting?: string
  seattle_recruitment_status?: string
  seattle_active_recruiting?: string
  conditions?: string
  locations?: string
  contacts?: string
  inclusion_criteria?: string
  exclusion_criteria?: string
  phase?: string
  enrollment?: string
  enrollment_type?: string
  sponsor?: string
  study_type?: string
  sex?: string
  minimum_age?: string
  maximum_age?: string
  healthy_volunteers?: string
  brief_summary?: string
  interventions?: string
  primary_outcomes?: string
  secondary_outcomes?: string
  start_date?: string
  primary_completion_date?: string
  completion_date?: string
  status_verified_date?: string
  [key: string]: unknown
}

const LOCATION_ACTIVE_FIELDS: { key: keyof Trial; label: string }[] = [
  { key: 'gainesville_active_recruiting', label: 'Gainesville, FL' },
  { key: 'los_angeles_active_recruiting', label: 'Los Angeles, CA' },
  { key: 'dallas_active_recruiting', label: 'Dallas, TX' },
  { key: 'minneapolis_active_recruiting', label: 'Minneapolis, MN' },
  { key: 'new_york_active_recruiting', label: 'New York, NY' },
  { key: 'seattle_active_recruiting', label: 'Seattle, WA' },
]

function formatRecruitingAt(t: Trial): string {
  const parts = LOCATION_ACTIVE_FIELDS
    .filter(({ key }) => String(t[key] || '').toUpperCase() === 'TRUE' || String(t[key] || '').toUpperCase() === 'YES')
    .map(({ label }) => label)
  return parts.length ? parts.join(', ') : '—'
}

const SECTION_NAV_ITEMS = [
  { id: 'overview', label: 'Overview' },
  { id: 'conditions', label: 'Conditions' },
  { id: 'locations', label: 'Locations' },
  { id: 'contacts', label: 'Contacts' },
  { id: 'brief-summary', label: 'Brief summary' },
  { id: 'interventions', label: 'Interventions' },
  { id: 'primary-outcomes', label: 'Primary outcomes' },
  { id: 'secondary-outcomes', label: 'Secondary outcomes' },
  { id: 'inclusion-criteria', label: 'Inclusion criteria' },
  { id: 'exclusion-criteria', label: 'Exclusion criteria' },
  { id: 'dates', label: 'Dates' },
]

function Section({
  id,
  title,
  children,
}: { id?: string; title: string; children: React.ReactNode }) {
  if (!children) return null
  return (
    <section id={id} className="mb-6 scroll-mt-24">
      <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-2">{title}</h3>
      <div className="text-slate-600 text-sm leading-relaxed whitespace-pre-wrap">{children}</div>
    </section>
  )
}

function scrollToSection(sectionId: string) {
  document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

export default function TrialDetail() {
  const { nctId } = useParams<{ nctId: string }>()
  const [trial, setTrial] = useState<Trial | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [emailDraft, setEmailDraft] = useState<{ email: string; contacts?: string } | null>(null)
  const [draftLoading, setDraftLoading] = useState(false)

  useEffect(() => {
    if (!nctId) return
    fetch(`/api/trial/${nctId}`)
      .then((r) => {
        if (!r.ok) throw new Error('Trial not found')
        return r.json()
      })
      .then(setTrial)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [nctId])

  const handleDraftEmail = useCallback(async () => {
    if (!trial) return
    setDraftLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/draft-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nct_id: trial.nct_id,
          patient_summary: DEFAULT_CREDENTIALS,
        }),
      })
      if (!res.ok) throw new Error('Failed to draft email')
      const data = await res.json()
      setEmailDraft({ email: data.email, contacts: trial.contacts })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to draft email')
    } finally {
      setDraftLoading(false)
    }
  }, [trial])

  const copyDraftToClipboard = () => {
    if (!emailDraft) return
    navigator.clipboard.writeText(emailDraft.email)
  }

  const extractEmails = (contacts: string): string[] => {
    const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g
    return [...new Set((contacts.match(emailRegex) || []))]
  }

  const copyContactEmails = () => {
    if (!emailDraft?.contacts) return
    navigator.clipboard.writeText(extractEmails(emailDraft.contacts).join(', '))
  }

  const updateDraftEmail = (text: string) => {
    if (!emailDraft) return
    setEmailDraft({ ...emailDraft, email: text })
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-surface-50">
        <p className="text-slate-500">Loading trial…</p>
      </div>
    )
  }

  if (error || !trial) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-surface-50 gap-4">
        <p className="text-red-600">{error || 'Trial not found'}</p>
        <Link to="/" className="text-primary-600 hover:underline">Back to Trial Finder</Link>
      </div>
    )
  }

  const ctGovUrl = `${CT_GOV_BASE}${trial.nct_id}`

  return (
    <div className="min-h-screen bg-surface-50">
      <header className="bg-white border-b border-surface-200 shadow-sm">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <Link to="/" className="text-sm text-primary-600 hover:underline mb-2 inline-block">
            ← Back to Trial Finder
          </Link>
          <h1 className="text-xl font-semibold text-slate-800">{trial.nct_id}</h1>
          <p className="text-slate-600 text-base mt-1">{trial.title}</p>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        {/* Mobile: horizontal scroll section picker */}
        <div className="lg:hidden -mx-6 px-6 pb-4 overflow-x-auto">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            Jump to section
          </p>
          <div className="flex gap-2">
            {SECTION_NAV_ITEMS.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                onClick={() => scrollToSection(id)}
                className="shrink-0 px-3 py-1.5 rounded-lg text-sm text-slate-600 bg-surface-100 hover:bg-primary-100 hover:text-primary-700 border border-surface-200"
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex gap-8">
          {/* Sticky section navigator (desktop) */}
          <nav
            aria-label="Jump to section"
            className="hidden lg:block shrink-0 w-48 pt-2 sticky top-4 self-start"
          >
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
              On this page
            </p>
            <ul className="space-y-1">
              {SECTION_NAV_ITEMS.map(({ id, label }) => (
                <li key={id}>
                  <button
                    type="button"
                    onClick={() => scrollToSection(id)}
                    className="text-left w-full text-sm text-slate-600 hover:text-primary-600 hover:underline py-0.5 truncate"
                  >
                    {label}
                  </button>
                </li>
              ))}
            </ul>
          </nav>

          <div className="min-w-0 flex-1">
          <div className="bg-white rounded-xl border border-surface-200 shadow-sm p-6 mb-8">
            <div id="overview" className="scroll-mt-24 grid gap-4 sm:grid-cols-2 mb-6">
              <div><span className="text-slate-500 text-sm">Status</span><p className="font-medium">{trial.overall_status}</p></div>
              <div><span className="text-slate-500 text-sm">Recruiting at</span><p className="font-medium">{formatRecruitingAt(trial)}</p></div>
              <div><span className="text-slate-500 text-sm">Phase</span><p className="font-medium">{trial.phase || '—'}</p></div>
              <div><span className="text-slate-500 text-sm">Enrollment</span><p className="font-medium">{trial.enrollment || '—'}</p></div>
              <div><span className="text-slate-500 text-sm">Sponsor</span><p className="font-medium">{trial.sponsor || '—'}</p></div>
              <div><span className="text-slate-500 text-sm">Study type</span><p className="font-medium">{trial.study_type || '—'}</p></div>
              <div><span className="text-slate-500 text-sm">Sex</span><p className="font-medium">{trial.sex || '—'}</p></div>
              <div><span className="text-slate-500 text-sm">Age</span><p className="font-medium">{[trial.minimum_age, trial.maximum_age].filter(Boolean).join(' – ') || '—'}</p></div>
            </div>

            <Section id="conditions" title="Conditions">{trial.conditions}</Section>
            <Section id="locations" title="Locations">{trial.locations}</Section>
            <Section id="contacts" title="Contacts">{trial.contacts}</Section>
            <Section id="brief-summary" title="Brief summary">{trial.brief_summary}</Section>
            <Section id="interventions" title="Interventions">{trial.interventions}</Section>
            <Section id="primary-outcomes" title="Primary outcomes">{trial.primary_outcomes}</Section>
            <Section id="secondary-outcomes" title="Secondary outcomes">{trial.secondary_outcomes}</Section>
            <Section id="inclusion-criteria" title="Inclusion criteria">{trial.inclusion_criteria}</Section>
            <Section id="exclusion-criteria" title="Exclusion criteria">{trial.exclusion_criteria}</Section>

            <section id="dates" className="mb-6 scroll-mt-24">
              <div className="grid gap-2 text-sm text-slate-500 mt-6">
                {trial.start_date && <p>Start date: {trial.start_date}</p>}
                {trial.primary_completion_date && <p>Primary completion: {trial.primary_completion_date}</p>}
                {trial.completion_date && <p>Completion: {trial.completion_date}</p>}
              </div>
            </section>
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-4 border-t border-surface-200">
            <button
              onClick={handleDraftEmail}
              disabled={draftLoading}
              className="px-5 py-2.5 rounded-xl bg-primary-500 text-white font-medium hover:bg-primary-600 disabled:opacity-50"
            >
              {draftLoading ? 'Drafting…' : 'Draft referral email'}
            </button>
            <a
              href={ctGovUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-slate-500 hover:text-slate-700 underline"
            >
              View on ClinicalTrials.gov
            </a>
          </div>
          </div>
        </div>
      </main>

      {emailDraft && (
        <div
          className="fixed inset-0 z-50 flex min-h-full min-w-full items-center justify-center overflow-y-auto bg-slate-900/40 p-4"
          onClick={() => setEmailDraft(null)}
        >
          <div
            className="m-auto max-h-[90vh] w-full max-w-2xl overflow-hidden rounded-2xl bg-white shadow-xl flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-6 py-4 border-b border-surface-200 flex items-center justify-between">
              <h2 className="font-semibold text-slate-800">Email draft — {trial.nct_id}</h2>
              <button onClick={() => setEmailDraft(null)} className="text-slate-400 hover:text-slate-600 text-xl leading-none">×</button>
            </div>
            {emailDraft.contacts && (
              <div className="px-6 py-3 border-b border-surface-200 bg-surface-50">
                <p className="text-xs font-medium text-slate-600 uppercase tracking-wide mb-2">Study contacts</p>
                <p className="text-sm text-slate-700 whitespace-pre-wrap">{emailDraft.contacts}</p>
                <button onClick={copyContactEmails} className="mt-2 text-xs font-medium text-primary-600 hover:text-primary-700">Copy contact emails</button>
              </div>
            )}
            <div className="flex-1 overflow-y-auto p-6">
              <p className="text-xs font-medium text-slate-600 uppercase tracking-wide mb-2">Email draft (editable)</p>
              <textarea
                value={emailDraft.email}
                onChange={(e) => updateDraftEmail(e.target.value)}
                className="w-full p-4 text-sm text-slate-700 font-mono resize-y min-h-[240px] rounded-lg border border-surface-200 focus:ring-2 focus:ring-primary-400 focus:border-transparent"
              />
            </div>
            <div className="px-6 py-4 border-t border-surface-200 flex justify-end gap-3">
              <button onClick={() => setEmailDraft(null)} className="px-4 py-2 text-slate-600 hover:text-slate-800">Close</button>
              <button onClick={copyDraftToClipboard} className="px-4 py-2 rounded-lg bg-primary-500 text-white hover:bg-primary-600">Copy draft to clipboard</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
