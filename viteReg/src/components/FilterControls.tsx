interface FilterControlsProps {
  blockIds: string[]
  selectedBlockId: string
  onlyAnnotated: boolean
  onBlockIdChange: (blockId: string) => void
  onOnlyAnnotatedChange: (onlyAnnotated: boolean) => void
}

export function FilterControls({
  blockIds,
  selectedBlockId,
  onlyAnnotated,
  onBlockIdChange,
  onOnlyAnnotatedChange,
}: FilterControlsProps) {
  return (
    <div
      style={{
        display: 'flex',
        gap: '1rem',
        alignItems: 'center',
        padding: '0.75rem 1rem',
        backgroundColor: '#ffffff',
        borderRadius: '6px',
        marginBottom: '0.75rem',
        border: '1px solid #e1e8ed',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <label htmlFor="block-filter" style={{ fontWeight: 'bold' }}>
          Block ID:
        </label>
        <select
          id="block-filter"
          value={selectedBlockId}
          onChange={(e) => onBlockIdChange(e.target.value)}
          style={{
            padding: '0.5rem',
            fontSize: '0.9rem',
            borderRadius: '6px',
            border: '1px solid #d5dbdb',
            backgroundColor: '#ffffff',
            color: '#2c3e50',
            cursor: 'pointer',
          }}
        >
          <option value="all">All Blocks</option>
          <option value="1">Block 1</option>
          {blockIds.map((blockId) => (
            <option key={blockId} value={blockId}>
              {blockId}
            </option>
          ))}
        </select>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <input
          type="checkbox"
          id="only-annotated"
          checked={onlyAnnotated}
          onChange={(e) => onOnlyAnnotatedChange(e.target.checked)}
        />
        <label htmlFor="only-annotated" style={{ cursor: 'pointer' }}>
          Only annotated slides
        </label>
      </div>
    </div>
  )
}

