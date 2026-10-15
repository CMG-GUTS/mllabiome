'use client'

import { useEffect } from 'react'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error('Application error:', error)
  }, [error])

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      padding: '2rem',
      background: 'var(--ml-bg, #0a0a0a)',
      color: 'var(--ml-text, #e5e5e5)',
    }}>
      <h2 style={{ marginBottom: '1rem', color: '#ef4444' }}>Something went wrong!</h2>
      <p style={{ marginBottom: '1.5rem', opacity: 0.7 }}>
        {error.message || 'An unexpected error occurred'}
      </p>
      <button
        onClick={() => reset()}
        style={{
          padding: '0.75rem 1.5rem',
          background: 'var(--ml-accent, #3b82f6)',
          color: 'white',
          border: 'none',
          borderRadius: '0.5rem',
          cursor: 'pointer',
          fontSize: '1rem',
        }}
      >
        Try again
      </button>
    </div>
  )
}
