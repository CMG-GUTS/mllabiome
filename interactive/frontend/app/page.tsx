'use client'

import { useRouter } from 'next/navigation'
import styles from './landing.module.css'
import ThemeToggle from '../components/ThemeToggle'

export default function Home() {
  const router = useRouter()

  return (
    <div className={styles.container}>
      <div className={styles.themeToggleWrapper}>
        <ThemeToggle />
      </div>
      
      <div className={styles.hero}>
        <div className={styles.heroGrid}></div>
        <h1>mllabiome</h1>
        <p className={styles.subtitle}>interactive inference for completed experiments</p>
        <p className={styles.description}>Explore inference results, upload sample profiles, and inspect prediction-level feature attributions.</p>
      </div>

      <div className={styles.modeSelector}>
        <div className={styles.modeCard} onClick={() => router.push('/inference')}>
          <div className={styles.cardHeader}>
            <h2>Model inference</h2>
          </div>
          <p>Use fitted models directly from the current experiment run.</p>
          <button className={styles.modeButton}>Open Inference</button>
        </div>
      </div>

      <div className={styles.info}>
        <h3>Interactive inference flow</h3>
        <div className={styles.steps}>
          <div className={styles.step}>
            <span className={styles.stepNumber}>01</span>
            <div>
              <h4>Select experiment model</h4>
              <p>Choose a fitted MPMA model from the completed experiment.</p>
            </div>
          </div>
          <div className={styles.step}>
            <span className={styles.stepNumber}>02</span>
            <div>
              <h4>Upload sample profile</h4>
              <p>Use sample-by-feature TSV/CSV or feature-by-sample TSV tables.</p>
            </div>
          </div>
          <div className={styles.step}>
            <span className={styles.stepNumber}>03</span>
            <div>
              <h4>Run local inference</h4>
              <p>Predictions use the original fitted model and its preprocessing.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
