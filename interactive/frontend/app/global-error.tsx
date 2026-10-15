'use client'

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <html>
      <body style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        padding: '2rem',
        background: '#0a0a0a',
        color: '#e5e5e5',
        fontFamily: 'system-ui, sans-serif',
      }}>
        <h2 style={{ marginBottom: '1rem', color: '#ef4444' }}>Something went wrong!</h2>
        <p style={{ marginBottom: '1.5rem', opacity: 0.7 }}>
          {error.message || 'A critical error occurred'}
        </p>
        <button
          onClick={() => reset()}
          style={{
            padding: '0.75rem 1.5rem',
            background: '#3b82f6',
            color: 'white',
            border: 'none',
            borderRadius: '0.5rem',
            cursor: 'pointer',
            fontSize: '1rem',
          }}
        >
          Try again
        </button>
      </body>
    </html>
  )
}
