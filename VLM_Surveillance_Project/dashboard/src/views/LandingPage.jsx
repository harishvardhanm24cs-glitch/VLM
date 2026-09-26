import React, { useState, useEffect } from 'react';

const API_URL = "http://localhost:8000";

export default function LandingPage({ setActiveTab, sysStatus, wsConnected }) {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const isBackendOnline = sysStatus?.backend === 'ONLINE';
  const isVideoOnline = sysStatus?.Video === 'ONLINE';
  const isAiOnline = sysStatus?.['CCTV Edge'] === 'ONLINE' || sysStatus?.VLM === 'ONLINE';

  return (
    <div className="view-container animate-fade" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '0', background: 'var(--bg-app)' }}>
      
      {/* MAIN SPLIT VIEW */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        
        {/* LEFT COLUMN: IDENTITY & CAPABILITIES */}
        <div style={{ flex: 1, padding: 'var(--space-8)', display: 'flex', flexDirection: 'column', justifyContent: 'center', borderRight: '1px solid var(--border-subtle)' }}>
          <div style={{ maxWidth: '600px', margin: '0 auto', width: '100%' }}>
            
            <div style={{ marginBottom: 'var(--space-8)' }}>
              <h1 style={{ fontSize: '2.5rem', letterSpacing: '0.05em', margin: 0, color: 'var(--text-main)' }}>
                VLM SURVEILLANCE PLATFORM
              </h1>
              <div className="text-caption" style={{ fontSize: '1rem', marginTop: '12px', color: 'var(--text-muted)' }}>
                Real-time AI-assisted surveillance and event monitoring.
              </div>
            </div>

            <div style={{ marginBottom: 'var(--space-8)' }}>
              <div className="text-caption text-mono" style={{ marginBottom: '16px', letterSpacing: '0.1em' }}>CORE CAPABILITIES</div>
              <ul style={{ listStyle: 'none', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', padding: 0 }}>
                {['Human Detection', 'Vehicle Classification', 'Behavioral Monitoring', 'Facial Expression Analysis', 'ANPR', 'Real-Time Alerts'].map((cap, idx) => (
                  <li key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.95rem', color: 'var(--text-main)' }}>
                    <span className="text-blue">●</span> {cap}
                  </li>
                ))}
              </ul>
            </div>

            <div style={{ display: 'flex', gap: '16px' }}>
              <button 
                onClick={() => setActiveTab('dashboard')}
                style={{ 
                  padding: '14px 28px', 
                  background: 'var(--color-blue)', 
                  color: '#fff', 
                  border: 'none', 
                  borderRadius: 'var(--radius-sm)', 
                  fontSize: '0.9rem', 
                  fontWeight: '600', 
                  letterSpacing: '0.1em',
                  cursor: 'pointer', 
                  transition: 'var(--transition-fast)'
                }}>
                ENTER COMMAND CENTER
              </button>
              <button 
                onClick={() => setActiveTab('status')}
                style={{ 
                  padding: '14px 28px', 
                  background: 'transparent', 
                  color: 'var(--text-main)', 
                  border: '1px solid var(--border-subtle)', 
                  borderRadius: 'var(--radius-sm)', 
                  fontSize: '0.9rem', 
                  fontWeight: '600', 
                  letterSpacing: '0.1em',
                  cursor: 'pointer', 
                  transition: 'var(--transition-fast)'
                }}>
                SYSTEM STATUS
              </button>
            </div>

          </div>
        </div>

        {/* RIGHT COLUMN: CAMERA PREVIEW & STATUS */}
        <div style={{ width: '450px', background: 'var(--bg-panel)', display: 'flex', flexDirection: 'column' }}>
          
          <div style={{ flex: 1, padding: 'var(--space-6)', display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
            
            <div className="text-caption text-mono" style={{ letterSpacing: '0.1em' }}>LIVE FEED PREVIEW</div>
            
            <div style={{ width: '100%', height: '240px', background: '#000', borderRadius: 'var(--radius-sm)', overflow: 'hidden', border: '1px solid var(--border-subtle)', position: 'relative' }}>
              {isVideoOnline ? (
                <img src={`${API_URL}/api/stream`} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <div className="flex-center" style={{ height: '100%', flexDirection: 'column', color: 'var(--color-red)' }}>
                  <span className="text-mono" style={{ fontSize: '0.8rem' }}>CAMERA DISCONNECTED</span>
                </div>
              )}
            </div>

            <div className="text-caption text-mono" style={{ letterSpacing: '0.1em', marginTop: 'var(--space-2)' }}>SYSTEM STATUS</div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>Backend:</span>
                <span className={`text-mono ${isBackendOnline ? 'text-green' : 'text-red'}`} style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
                  {isBackendOnline ? 'ONLINE' : 'OFFLINE'}
                </span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>Camera:</span>
                <span className={`text-mono ${isVideoOnline ? 'text-green' : 'text-red'}`} style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
                  {isVideoOnline ? 'CONNECTED' : 'DISCONNECTED'}
                </span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>WebSocket:</span>
                <span className={`text-mono ${wsConnected ? 'text-green' : 'text-red'}`} style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
                  {wsConnected ? 'CONNECTED' : 'DISCONNECTED'}
                </span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>AI Pipeline:</span>
                <span className={`text-mono ${isAiOnline ? 'text-green' : 'text-red'}`} style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
                  {isAiOnline ? 'ONLINE' : 'OFFLINE'}
                </span>
              </div>
            </div>

          </div>
        </div>

      </div>

      {/* FOOTER */}
      <div style={{ 
        height: '40px', 
        background: 'var(--bg-panel)', 
        borderTop: '1px solid var(--border-subtle)', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        padding: '0 var(--space-6)',
        fontSize: '0.75rem'
      }} className="text-mono text-muted">
        <div>
          SYS_VERSION: 2.4.1-PROD
        </div>
        <div style={{ display: 'flex', gap: '24px' }}>
          <div>
            STATUS: <span className={wsConnected && isBackendOnline ? 'text-green' : 'text-red'}>
              {wsConnected && isBackendOnline ? 'SYSTEM NOMINAL' : 'CONNECTIVITY ISSUES'}
            </span>
          </div>
          <div>
            LOCAL TIME: {currentTime.toLocaleString()}
          </div>
        </div>
      </div>

    </div>
  );
}
