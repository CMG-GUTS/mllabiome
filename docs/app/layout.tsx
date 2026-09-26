import '../styles/globals.css'
import type { Metadata, Viewport } from 'next'

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || ''

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
}

export const metadata: Metadata = {
  title: {
    template: '%s – mllabiome',
    default: 'mllabiome',
  },
  description: 'Machine learning for microbiome research: build, compare, combine and explain models from microbiome abundance data.',
  keywords: [
    'microbiome machine learning',
    'microbiota machine learning',
    'nested cross-validation',
    'leave-one-dataset-out',
    'ensemble learning',
    'multimodal learning',
    'explainable AI',
  ],
  openGraph: {
    title: 'mllabiome',
    description: 'Machine learning for microbiome research: build, compare, combine and explain models from microbiome abundance data.',
    type: 'website',
  },
  icons: {
    icon: [{ url: `${basePath}/favicon.svg`, type: 'image/svg+xml' }],
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  )
}
