import React from 'react'
import Link from 'next/link'
import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'

const navLinkStyle: React.CSSProperties = {
  fontSize: '0.78rem',
  color: '#64748b',
  textDecoration: 'none',
  padding: '0.3rem 0.75rem',
  border: '1px solid #e2e8f0',
  borderRadius: 5,
}

const navbar = (
  <Navbar
    logo={
      <span
        style={{
          fontWeight: 700,
          fontSize: '0.88rem',
          color: '#0f172a',
          letterSpacing: '-0.02em',
        }}
      >
        mllabiome
      </span>
    }
    logoLink="/"
    projectLink="https://github.com/CMG-GUTS/mllabiome"
  >
    <Link href="/docs/current/quickstart" style={navLinkStyle}>
      Quickstart
    </Link>
  </Navbar>
)

const footer = <Footer />

export default async function DocsLayout({ children }: { children: React.ReactNode }) {
  return (
    <Layout
      navbar={navbar}
      pageMap={await getPageMap('/docs/current')}
      docsRepositoryBase="https://github.com/CMG-GUTS/mllabiome/tree/main/docs"
      editLink={null}
      feedback={{ content: null }}
      toc={{ title: 'On this page' }}
      darkMode={false}
      footer={footer}
    >
      {children}
    </Layout>
  )
}
