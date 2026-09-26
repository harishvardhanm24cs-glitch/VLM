import React from 'react';

const TABS = [
  { id: 'landing', icon: '🏠', label: 'Landing Page' },
  { id: 'dashboard', icon: '📊', label: 'Dashboard' },
  { id: 'live', icon: '📹', label: 'Live Console' },
  { id: 'human', icon: '👤', label: 'Human Monitor' },
  { id: 'vehicle', icon: '🚗', label: 'Vehicle Monitor' },
  { id: 'alerts', icon: '🚨', label: 'Alert Center' },
  { id: 'anpr', icon: '📝', label: 'ANPR Log' },
  { id: 'status', icon: '⚙️', label: 'System Status' },
];

export default function Sidebar({ activeTab, setActiveTab }) {
  return (
    <div style={{ width: '250px', background: 'var(--bg-panel)', borderRight: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: 'var(--space-6)', borderBottom: '1px solid var(--border-subtle)', fontWeight: 'bold', fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ color: 'var(--color-blue)' }}>⚡</span> VLM CORE
      </div>
      <div style={{ padding: 'var(--space-4)', display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px',
              background: activeTab === tab.id ? 'var(--border-subtle)' : 'transparent',
              color: activeTab === tab.id ? 'var(--text-main)' : 'var(--text-muted)',
              border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
              fontWeight: activeTab === tab.id ? '600' : '400',
              textAlign: 'left', fontSize: '0.875rem',
              transition: 'var(--transition-fast)'
            }}
          >
            <span>{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  );
}
