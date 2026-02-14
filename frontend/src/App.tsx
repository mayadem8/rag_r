import { useState } from 'react'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Backend base URL (set this in Vercel as VITE_API_URL)
  // Example value: https://rag-r.onrender.com
  const API = import.meta.env.VITE_API_URL || 'https://rag-r.onrender.com'

  const askQuestion = async () => {
    const trimmed = question.trim()
    if (!trimmed || loading) return

    setLoading(true)
    setError('')
    setAnswer('')

    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      })

      if (!res.ok) {
        let detail = ''
        try {
          const data = await res.clone().json()
          detail = (data as any)?.detail ?? ''
        } catch {
          try {
            detail = await res.text()
          } catch {
            detail = ''
          }
        }

        const message = detail
          ? `Request failed (${res.status}): ${detail}`
          : `Request failed (${res.status})`
        throw new Error(message)
      }

      const data = await res.json()
      setAnswer(data?.answer ?? 'No answer returned.')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Something went wrong.'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    void askQuestion()
  }

  return (
    <div className="app">
      <form className="search" onSubmit={onSubmit}>
        <input
          className="search-input"
          type="text"
          placeholder="Ask a question…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button className="search-button" type="submit" disabled={loading}>
          {loading ? 'Asking…' : 'Ask'}
        </button>
      </form>

      <section className="result" aria-live="polite">
        {error ? (
          <div className="error">{error}</div>
        ) : answer ? (
          <div className="answer">{answer}</div>
        ) : (
          <div className="placeholder">Your answer will appear here.</div>
        )}
      </section>
    </div>
  )
}

export default App
