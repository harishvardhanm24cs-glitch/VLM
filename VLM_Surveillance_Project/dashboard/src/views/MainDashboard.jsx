import React from 'react';

const API_URL = "http://localhost:8000";

export default function MainDashboard({ stats, events, sysStatus, expressions }) {
  const activeAlerts = events.filter(e => e.severity === 'CRITICAL' || e.severity === 'HIGH').length;
  
  const activeTracks = stats?.tracks || {};
  const tracksArray = Object.values(activeTracks);
  
  const persons = tracksArray.filter(t => t.class === 'person');
  const vehicles = tracksArray.filter(t => ['car', 'truck', 'bus', 'motorcycle', 'four-wheeler', 'two-wheeler'].includes(t.class?.toLowerCase()));
  const milVehicles = vehicles.filter(v => v.vehicle_category === 'Military');

  // For Event Stream (Right Column)
  const recentEvents = [...events].sort((a, b) => b.timestamp - a.timestamp).slice(0, 15);

  // For Bottom Sections
  const recentPersons = [...persons].slice(0, 5); // Just taking first 5 for preview
  const recentVehiclesList = [...vehicles].slice(0, 5);

  return (
    <div className="view-container animate-fade" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', gap: 'var(--space-4)' }}>
      
      {/* ==================== FIRST ROW: COMPACT KPIs ==================== */}
      <div className="panel" style={{ padding: 'var(--space-3) var(--space-4)', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 0 }}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <span className="text-mono" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{persons.length}</span>
          <span className="text-caption">ACTIVE PEOPLE</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className="text-mono" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{vehicles.length}</span>
          <span className="text-caption">TOTAL VEHICLES</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className="text-mono text-green" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{milVehicles.length}</span>
          <span className="text-caption">MILITARY ASSETS</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className={`text-mono ${activeAlerts > 0 ? 'text-red' : 'text-green'}`} style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>
            {activeAlerts}
          </span>
          <span className="text-caption">CRITICAL ALERTS</span>
        </div>
      </div>

      {/* ==================== MAIN CONTENT: 65/35 SPLIT ==================== */}
      <div style={{ display: 'grid', gridTemplateColumns: '65% 1fr', gap: 'var(--space-4)', flex: 1, minHeight: '350px' }}>
        
        {/* LEFT: LIVE MONITOR */}
        <div className="panel" style={{ padding: 0, position: 'relative', background: '#000', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
          {sysStatus?.Video === 'ONLINE' ? (
            <img src={`${API_URL}/api/stream`} alt="Live Feed" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
          ) : (
            <div className="flex-center" style={{ height: '100%', flexDirection: 'column', color: 'var(--color-red)' }}>
              <span className="text-mono">CAMERA DISCONNECTED</span>
            </div>
          )}

          {/* Minimal Overlay - top right */}
          <div style={{ position: 'absolute', top: '12px', right: '12px', display: 'flex', flexDirection: 'column', gap: '4px', pointerEvents: 'none' }}>
             {/* Render max 3 active items to avoid covering video completely */}
             {persons.slice(0, 3).map(p => {
                const expr = expressions[p.track_id]?.expression?.toUpperCase();
                return (
                  <div key={p.track_id} style={{ background: 'rgba(0,0,0,0.6)', border: '1px solid var(--border-subtle)', padding: '4px 8px', borderRadius: '4px', display: 'flex', gap: '12px', fontSize: '0.75rem', backdropFilter: 'blur(2px)' }}>
                    <span className="text-blue font-bold">P#{p.track_id}</span>
                    <span className="text-main">{expr || 'TRACKING'}</span>
                  </div>
                )
             })}
             {vehicles.slice(0, 3).map(v => (
                <div key={v.track_id} style={{ background: 'rgba(0,0,0,0.6)', border: '1px solid var(--border-subtle)', padding: '4px 8px', borderRadius: '4px', display: 'flex', gap: '12px', fontSize: '0.75rem', backdropFilter: 'blur(2px)' }}>
                  <span className={v.vehicle_category === 'Military' ? 'text-green font-bold' : 'text-gray font-bold'}>V#{v.track_id}</span>
                  <span className="text-main">
                    {v.vehicle_category === 'Military' ? v.subtype?.toUpperCase() || 'MILITARY' : v.class?.toUpperCase() || 'VEHICLE'}
                  </span>
                </div>
             ))}
          </div>
        </div>

        {/* RIGHT: EVENT STREAM */}
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.05em' }}>
            EVENT STREAM
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-3)' }}>
            {recentEvents.length === 0 ? (
              <div className="text-caption flex-center" style={{ height: '100%' }}>WAITING FOR EVENTS</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {recentEvents.map((evt, idx) => {
                  const isCrit = evt.severity === 'CRITICAL' || evt.severity === 'HIGH';
                  const isWarn = evt.severity === 'WARNING';
                  const dotClass = isCrit ? 'dot-red' : (isWarn ? 'dot-amber' : 'dot-blue');
                  
                  return (
                    <div key={`es-${evt.event_id || idx}`} style={{ display: 'flex', flexDirection: 'column', gap: '2px', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
                      <div className="flex-between">
                        <span className="text-mono text-muted" style={{ fontSize: '0.7rem' }}>
                          {new Date(evt.timestamp * 1000).toLocaleTimeString([], { hour12: false })}
                        </span>
                        {evt.track_id && <span className="text-mono text-blue" style={{ fontSize: '0.7rem', fontWeight: 'bold' }}>#{evt.track_id}</span>}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', fontWeight: 600, color: isCrit ? 'var(--color-red)' : 'var(--text-main)' }}>
                        <span className={`status-dot ${dotClass}`} style={{ width: '6px', height: '6px', marginRight: 0 }} />
                        {evt.event_type.replace(/_/g, ' ')}
                      </div>
                      <div className="text-caption" style={{ fontSize: '0.75rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {evt.description}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

      </div>

      {/* ==================== BOTTOM: 3 PRACTICAL SECTIONS ==================== */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-4)', height: '150px' }}>
        
        {/* RECENT VEHICLES */}
        <div className="panel" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '6px 12px', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
            RECENT VEHICLES
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
            {recentVehiclesList.length === 0 ? <div className="text-caption">NO DATA</div> : recentVehiclesList.map(v => (
              <div key={v.track_id} className="flex-between" style={{ fontSize: '0.75rem', marginBottom: '4px' }}>
                <span className="text-mono text-muted">#{v.track_id}</span>
                <span className={v.vehicle_category === 'Military' ? 'text-green' : 'text-gray'}>
                  {v.vehicle_category === 'Military' ? v.subtype?.toUpperCase() : v.class?.toUpperCase()}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* RECENT PERSON ACTIVITY */}
        <div className="panel" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '6px 12px', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
            RECENT PERSON ACTIVITY
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
            {recentPersons.length === 0 ? <div className="text-caption">NO DATA</div> : recentPersons.map(p => {
               const expr = expressions[p.track_id]?.expression?.toUpperCase();
               return (
                <div key={p.track_id} className="flex-between" style={{ fontSize: '0.75rem', marginBottom: '4px' }}>
                  <span className="text-mono text-muted">#{p.track_id}</span>
                  <span className={p.loitering ? 'text-amber' : 'text-main'}>
                    {p.loitering ? 'STATIONARY' : (expr || 'MOVING')}
                  </span>
                </div>
               )
            })}
          </div>
        </div>

        {/* SYSTEM HEALTH */}
        <div className="panel" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '6px 12px', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
            SYSTEM HEALTH
          </div>
          <div style={{ flex: 1, padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.75rem' }}>
            <div className="flex-between">
              <span className="text-muted">BACKEND API:</span>
              <span className={`text-mono ${sysStatus?.backend === 'ONLINE' ? 'text-green' : 'text-red'}`}>{sysStatus?.backend || 'OFFLINE'}</span>
            </div>
            <div className="flex-between">
              <span className="text-muted">VIDEO INGEST:</span>
              <span className={`text-mono ${sysStatus?.Video === 'ONLINE' ? 'text-green' : 'text-red'}`}>{sysStatus?.Video || 'OFFLINE'}</span>
            </div>
            <div className="flex-between">
              <span className="text-muted">EDGE AI LOOP:</span>
              <span className={`text-mono ${sysStatus?.['CCTV Edge'] === 'ONLINE' ? 'text-green' : 'text-red'}`}>{sysStatus?.['CCTV Edge'] || 'OFFLINE'}</span>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
}
