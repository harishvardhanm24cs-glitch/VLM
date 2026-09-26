import React, { useState, useEffect } from 'react';

export default function Topbar({ soundEnabled, setSoundEnabled, wsConnected, sysStatus, activePageName = 'COMMAND DASHBOARD' }) {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div style={{
      height: '48px',
      background: 'var(--bg-panel)',
      borderBottom: '1px solid var(--border-subtle)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 var(--space-4)'
    }}>
      {/* LEFT: Identity & Page Name */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{ fontWeight: 700, color: 'var(--color-blue)', letterSpacing: '0.05em', fontSize: '0.9rem' }}>
          VLM SURVEILLANCE
        </div>
        <div style={{ width: '1px', height: '16px', background: 'var(--border-subtle)' }} />
        <div className="text-muted" style={{ fontSize: '0.875rem', fontWeight: 500, letterSpacing: '0.05em' }}>
          {activePageName}
        </div>
      </div>
      
      {/* RIGHT: Status & Controls */}
      <div style={{ display: 'flex', gap: 'var(--space-6)', alignItems: 'center' }}>
        
        <div style={{ display: 'flex', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.05em' }}>
            <span className={`status-dot ${sysStatus?.backend === 'ONLINE' ? 'dot-green' : 'dot-red'}`} />
            <span className="text-muted">API:</span>
            <span className={sysStatus?.backend === 'ONLINE' ? 'text-green' : 'text-red'} style={{ marginLeft: '4px' }}>
              {sysStatus?.backend || 'OFFLINE'}
            </span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.05em' }}>
            <span className={`status-dot ${sysStatus?.Video === 'ONLINE' ? 'dot-green' : 'dot-red'}`} />
            <span className="text-muted">CAM:</span>
            <span className={sysStatus?.Video === 'ONLINE' ? 'text-green' : 'text-red'} style={{ marginLeft: '4px' }}>
              {sysStatus?.Video || 'OFFLINE'}
            </span>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.05em' }}>
            <span className={`status-dot ${wsConnected ? 'dot-green' : 'dot-red'}`} />
            <span className="text-muted">WS:</span>
            <span className={wsConnected ? 'text-green' : 'text-red'} style={{ marginLeft: '4px' }}>
              {wsConnected ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
        </div>

        <div style={{ width: '1px', height: '16px', background: 'var(--border-subtle)' }} />

        <div className="text-mono text-muted" style={{ fontSize: '0.75rem' }}>
          {currentTime.toLocaleTimeString()}
        </div>
        
        <div style={{ width: '1px', height: '16px', background: 'var(--border-subtle)' }} />
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '24px', height: '24px', background: 'var(--border-subtle)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem' }}>
            OP
          </div>
          <span className="text-caption">OPERATOR_1</span>
        </div>

      </div>
    </div>
  );
}
