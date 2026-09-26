import React, { useState, useEffect, useRef } from 'react';

const API_URL = "http://localhost:8000";

export default function LiveConsole({ sysStatus, stats, events, expressions }) {
  const containerRef = useRef(null);
  
  // Controls
  const [isPaused, setIsPaused] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [camera, setCamera] = useState('CAM_01');

  // Pause UI Rendering state (frontend only)
  const [localStats, setLocalStats] = useState(stats);
  const [localEvents, setLocalEvents] = useState(events);
  const [localExpressions, setLocalExpressions] = useState(expressions);

  useEffect(() => {
    if (!isPaused) {
      setLocalStats(stats);
      setLocalEvents(events);
      setLocalExpressions(expressions);
    }
  }, [stats, events, expressions, isPaused]);

  const activeTracks = localStats?.tracks || {};
  const tracksArray = Object.values(activeTracks);
  
  const persons = tracksArray.filter(t => t.class === 'person');
  const vehicles = tracksArray.filter(t => ['car', 'truck', 'bus', 'motorcycle', 'four-wheeler', 'two-wheeler'].includes(t.class?.toLowerCase()));
  
  const liveEvents = [...localEvents].sort((a, b) => b.timestamp - a.timestamp).slice(0, 50);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handleFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  const lastEvent = liveEvents.length > 0 ? liveEvents[0] : null;

  return (
    <div ref={containerRef} className="view-container animate-fade" style={{ background: isFullscreen ? '#000' : 'transparent', height: '100%', display: 'flex', flexDirection: 'column', padding: isFullscreen ? '0' : 'var(--space-4)', gap: isFullscreen ? '0' : 'var(--space-4)', overflow: 'hidden' }}>
      
      {/* TOP CONTROLS */}
      {!isFullscreen && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <select 
              value={camera} 
              onChange={(e) => setCamera(e.target.value)}
              style={{ padding: '4px 8px', background: 'var(--bg-panel)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}
            >
              <option value="CAM_01">CAM 01 - MAIN GATE</option>
            </select>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button onClick={() => setIsPaused(!isPaused)} style={{ padding: '4px 12px', background: isPaused ? 'rgba(245, 158, 11, 0.2)' : 'transparent', color: isPaused ? 'var(--color-amber)' : 'var(--text-main)', border: `1px solid ${isPaused ? 'var(--color-amber)' : 'var(--border-subtle)'}`, borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontWeight: 'bold', fontSize: '0.75rem', letterSpacing: '0.05em' }}>
              {isPaused ? '▶ RESUME UI' : '⏸ PAUSE UI'}
            </button>
            <button onClick={toggleFullscreen} style={{ padding: '4px 12px', background: 'transparent', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontWeight: 'bold', fontSize: '0.75rem', letterSpacing: '0.05em' }}>
              ⛶ FULLSCREEN
            </button>
          </div>
        </div>
      )}

      {/* MAIN SPLIT VIEW */}
      <div style={{ display: 'grid', gridTemplateColumns: isFullscreen ? '1fr' : '1fr 350px', gap: isFullscreen ? '0' : 'var(--space-4)', flex: 1, overflow: 'hidden' }}>
        
        {/* LEFT: LIVE CAMERA FEED */}
        <div className={isFullscreen ? '' : 'panel'} style={{ flex: 1, padding: 0, position: 'relative', background: '#000', overflow: 'hidden', border: isFullscreen ? 'none' : '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column' }}>
          
          <div style={{ flex: 1, position: 'relative' }}>
            {sysStatus?.Video === 'ONLINE' ? (
              <img src={`${API_URL}/api/stream`} alt="Live Feed" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            ) : (
              <div className="flex-center" style={{ height: '100%', color: 'var(--color-red)', flexDirection: 'column' }}>
                <span className="text-mono">WAITING FOR VIDEO SIGNAL</span>
              </div>
            )}
            
            {/* UNOBTRUSIVE HUD OVERLAY */}
            <div style={{ position: 'absolute', top: '16px', left: '16px', display: 'flex', flexDirection: 'column', gap: '8px', pointerEvents: 'none' }}>
              {persons.slice(0, 5).map(p => (
                <div key={`hud-p-${p.track_id}`} style={{ border: '1px solid rgba(59, 130, 246, 0.5)', background: 'rgba(0,0,0,0.5)', padding: '2px 6px', color: '#fff', fontSize: '0.7rem', display: 'inline-block', width: 'fit-content' }}>
                  Track #{p.track_id}
                </div>
              ))}
              {vehicles.slice(0, 5).map(v => (
                <div key={`hud-v-${v.track_id}`} style={{ border: `1px solid ${v.vehicle_category === 'Military' ? 'rgba(16, 185, 129, 0.5)' : 'rgba(107, 114, 128, 0.5)'}`, background: 'rgba(0,0,0,0.5)', padding: '2px 6px', color: '#fff', fontSize: '0.7rem', display: 'inline-block', width: 'fit-content' }}>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span>Track #{v.track_id}</span>
                    <span className={v.vehicle_category === 'Military' ? 'text-green' : 'text-gray'}>
                      {v.vehicle_category === 'Military' ? v.subtype?.toUpperCase() || 'MILITARY' : 'NORMAL'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* BOTTOM STATUS BAR (INSIDE VIDEO FRAME) */}
          <div style={{ height: '24px', background: 'rgba(0,0,0,0.8)', borderTop: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', padding: '0 12px', gap: '24px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className={`status-dot ${sysStatus?.Video === 'ONLINE' ? 'dot-green' : 'dot-red'}`} style={{ width: '6px', height: '6px' }} />
              <span>CAM {sysStatus?.Video === 'ONLINE' ? 'ONLINE' : 'OFFLINE'}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className={`status-dot ${sysStatus?.backend === 'ONLINE' ? 'dot-green' : 'dot-red'}`} style={{ width: '6px', height: '6px' }} />
              <span>WS CONNECTED</span>
            </div>
            <div className="text-mono">
              LAST EVENT: {lastEvent ? new Date(lastEvent.timestamp * 1000).toLocaleTimeString([], { hour12: false }) : 'WAITING'}
            </div>
          </div>

        </div>

        {/* RIGHT: LIVE EVENTS */}
        {!isFullscreen && (
          <div className="panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.05em' }}>
              LIVE EVENTS
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-3)' }}>
              {liveEvents.length === 0 ? (
                <div className="text-caption flex-center" style={{ height: '100%' }}>NO EVENTS YET</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {liveEvents.map(evt => {
                    const isCrit = evt.severity === 'CRITICAL' || evt.severity === 'HIGH';
                    const isWarn = evt.severity === 'WARNING';
                    
                    return (
                      <div key={`le-${evt.event_id}`} style={{ paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <div className="flex-between">
                          <span className="text-mono text-muted" style={{ fontSize: '0.7rem' }}>
                            {new Date(evt.timestamp * 1000).toLocaleTimeString([], { hour12: false })}
                          </span>
                          {evt.track_id && <span className="text-mono text-blue" style={{ fontSize: '0.7rem', fontWeight: 'bold' }}>#{evt.track_id}</span>}
                        </div>
                        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: isCrit ? 'var(--color-red)' : (isWarn ? 'var(--color-amber)' : 'var(--text-main)') }}>
                          {evt.event_type.replace(/_/g, ' ')}
                        </div>
                        <div className="text-caption" style={{ fontSize: '0.75rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {evt.description}
                        </div>
                        {evt.confidence && (
                          <div className="text-mono text-muted" style={{ fontSize: '0.7rem', marginTop: '2px' }}>
                            CONF: {(evt.confidence * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
