import { useState, useEffect } from 'react'
import { getCases, type Case } from '../services/cases'

interface CaseSelectorProps {
  value?: string
  onChange: (caseId: string) => void
}

export function CaseSelector({ value, onChange }: CaseSelectorProps) {
  const [cases, setCases] = useState<Case[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadCases() {
      try {
        const data = await getCases()
        setCases(data)
        if (data.length > 0 && !value) {
          // Try to find E20-11 first, otherwise use first case
          const e20_11 = data.find((c) => c.label === 'E20-11' || c.label.includes('E20-11'))
          onChange(e20_11 ? e20_11.value : data[0].value)
        }
      } catch (error) {
        console.error('Failed to load cases:', error)
      } finally {
        setLoading(false)
      }
    }
    loadCases()
  }, [value, onChange])

  if (loading) {
    return <div>Loading cases...</div>
  }

  return (
    <select
      value={value || ''}
      onChange={(e) => onChange(e.target.value)}
      style={{
        padding: '0.5rem',
        fontSize: '0.95rem',
        borderRadius: '6px',
        border: '1px solid #d5dbdb',
        minWidth: '200px',
        backgroundColor: '#ffffff',
        color: '#2c3e50',
        cursor: 'pointer',
        transition: 'border-color 0.2s ease',
      }}
      onFocus={(e) => {
        e.target.style.borderColor = '#3498db'
        e.target.style.outline = 'none'
      }}
      onBlur={(e) => {
        e.target.style.borderColor = '#d5dbdb'
      }}
    >
      {cases.map((case_) => (
        <option key={case_.value} value={case_.value}>
          {case_.label}
        </option>
      ))}
    </select>
  )
}

