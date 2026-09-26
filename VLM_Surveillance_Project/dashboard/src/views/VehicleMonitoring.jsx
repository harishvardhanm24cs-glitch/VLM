import React, { useState, useMemo } from 'react';

export default function VehicleMonitoring({ stats, events }) {
  const [catFilter, setCatFilter] = useState('ALL');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [selectedVehicleId, setSelectedVehicleId] = useState(null);
  
  const activeTracks = stats?.tracks || {};
  const vehicles = Object.values(activeTracks).filter(t => ['car', 'truck', 'bus', 'motorcycle', 'four-wheeler', 'two-wheeler'].includes(t.class?.toLowerCase()));

  // Extract latest ANPR read per track_id
  const anprDict = useMemo(() => {
    const dict = {};
    events.forEach(e => {
      if (e.event_type?.includes('ANPR') && e.track_id) {
        // Keep the most recent plate
        if (!dict[e.track_id] || dict[e.track_id].timestamp < e.timestamp) {
          dict[e.track_id] = e;
        }
      }
    });
    return dict;
  }, [events]);

  // KPI Calculations
  const totalVehicles = vehicles.length;
  const militaryCount = vehicles.filter(v => v.vehicle_category?.toUpperCase() === 'MILITARY').length;
  const normalCount = vehicles.filter(v => v.vehicle_category?.toUpperCase() === 'NORMAL').length;
  const uncertainCount = totalVehicles - militaryCount - normalCount;

  // Apply filters
  const filteredVehicles = vehicles.filter(v => {
    const cat = v.vehicle_category?.toUpperCase() || 'UNCERTAIN';
    
    // Category Filter
    if (catFilter !== 'ALL') {
      if (catFilter === 'MILITARY' && cat !== 'MILITARY') return false;
      if (catFilter === 'NORMAL' && cat !== 'NORMAL') return false;
      if (catFilter === 'UNCERTAIN' && ['MILITARY', 'NORMAL'].includes(cat)) return false;
    }

    // YOLO Type Filter
    if (typeFilter !== 'ALL') {
      if (v.class?.toLowerCase() !== typeFilter.toLowerCase()) return false;
    }

    return true;
  });

  const selectedVehicle = vehicles.find(v => v.track_id === selectedVehicleId);
  const selectedAnpr = selectedVehicleId ? anprDict[selectedVehicleId] : null;

  return (
    <div className="view-container animate-fade" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      
      {/* ==================== TOP: COMPACT KPIs ==================== */}
      <div className="panel" style={{ padding: 'var(--space-3) var(--space-4)', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <span className="text-mono" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{totalVehicles}</span>
          <span className="text-caption">TOTAL VEHICLES</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className="text-mono text-green" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{militaryCount}</span>
          <span className="text-caption">MILITARY</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className="text-mono text-gray" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{normalCount}</span>
          <span className="text-caption">NORMAL</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className="text-mono text-amber" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{uncertainCount}</span>
          <span className="text-caption">UNCERTAIN</span>
        </div>
      </div>

      {/* ==================== HEADER BAR FOR TABLE ==================== */}
      <div className="flex-between panel" style={{ padding: '8px 16px', flexDirection: 'row', marginBottom: 'var(--space-4)' }}>
        <h1 style={{ fontSize: '1.1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="text-blue">●</span> L2 VEHICLE MONITORING
        </h1>
        
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span className="text-caption">CATEGORY:</span>
            <select 
              value={catFilter}
              onChange={(e) => setCatFilter(e.target.value)}
              style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}
            >
              <option value="ALL">All Categories</option>
              <option value="MILITARY">Military</option>
              <option value="NORMAL">Normal</option>
              <option value="UNCERTAIN">Uncertain</option>
            </select>
          </div>
          
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span className="text-caption">YOLO TYPE:</span>
            <select 
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}
            >
              <option value="ALL">All Types</option>
              <option value="CAR">Car</option>
              <option value="TRUCK">Truck</option>
              <option value="BUS">Bus</option>
              <option value="MOTORCYCLE">Motorcycle</option>
            </select>
          </div>
        </div>
      </div>

      <div className="split-view" style={{ flex: 1, overflow: 'hidden', gridTemplateColumns: selectedVehicle ? '1fr 350px' : '1fr' }}>
        
        {/* ==================== MAIN AREA: DATA TABLE ==================== */}
        <div className="data-table-container" style={{ height: '100%', overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>TRACK</th>
                <th>YOLO TYPE</th>
                <th>CATEGORY</th>
                <th>SUBTYPE</th>
                <th>CONFIDENCE</th>
                <th>PLATE</th>
                <th>CAMERA</th>
                <th>LAST SEEN</th>
              </tr>
            </thead>
            <tbody>
              {filteredVehicles.length === 0 ? (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                    <span className="text-mono text-muted">NO DATA</span>
                  </td>
                </tr>
              ) : (
                filteredVehicles.map(v => {
                  const isMilitary = v.vehicle_category?.toUpperCase() === 'MILITARY';
                  const isNormal = v.vehicle_category?.toUpperCase() === 'NORMAL';
                  const anpr = anprDict[v.track_id];

                  return (
                    <tr key={v.track_id} onClick={() => setSelectedVehicleId(v.track_id)} style={{ cursor: 'pointer', background: selectedVehicleId === v.track_id ? 'var(--bg-panel-hover)' : 'transparent' }}>
                      <td className="text-mono text-blue font-bold">#{v.track_id}</td>
                      <td>
                        <span style={{ textTransform: 'lowercase' }}>{v.class || 'unknown'}</span>
                      </td>
                      <td>
                        {isMilitary ? (
                          <span className="badge badge-green">MILITARY</span>
                        ) : isNormal ? (
                          <span className="badge badge-gray">NORMAL</span>
                        ) : (
                          <span className="badge badge-amber">UNCERTAIN</span>
                        )}
                      </td>
                      <td>
                        {isMilitary ? (
                          <span className="text-green" style={{ fontWeight: 'bold' }}>{v.subtype?.toUpperCase() || '—'}</span>
                        ) : (
                          <span className="text-muted">—</span>
                        )}
                      </td>
                      <td className="text-mono text-muted">
                        {v.classification_confidence ? `${(v.classification_confidence * 100).toFixed(0)}%` : '—'}
                      </td>
                      <td className="text-mono font-bold">
                        {anpr?.plate_text || '—'}
                      </td>
                      <td className="text-mono text-muted">
                        CAM-01
                      </td>
                      <td className="text-mono text-muted">
                        {v.last_seen ? new Date(v.last_seen * 1000).toLocaleTimeString([], { hour12: false }) : new Date().toLocaleTimeString([], { hour12: false })}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* ==================== VEHICLE DETAIL DRAWER ==================== */}
        {selectedVehicle && (
          <div className="panel" style={{ padding: '0', overflowY: 'auto' }}>
            <div className="flex-between" style={{ padding: '16px', borderBottom: '1px solid var(--border-subtle)' }}>
              <h2 style={{ margin: 0, fontSize: '1rem', color: 'var(--color-blue)' }}>VEHICLE #{selectedVehicle.track_id}</h2>
              <button onClick={() => setSelectedVehicleId(null)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}>×</button>
            </div>
            
            <div style={{ padding: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '16px', fontSize: '0.875rem' }}>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">TRACK ID</span>
                  <span className="text-mono text-blue font-bold">#{selectedVehicle.track_id}</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">YOLO DETECTION</span>
                  <span className="font-bold" style={{ textTransform: 'uppercase' }}>{selectedVehicle.class || 'UNKNOWN'}</span>
                </div>
                
                <div style={{ marginTop: '8px', background: 'var(--bg-app)', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
                  <div className="text-caption" style={{ marginBottom: '12px', letterSpacing: '0.05em', color: 'var(--color-blue)' }}>L1 CLASSIFICATION</div>
                  <div className="flex-between" style={{ marginBottom: '8px' }}>
                    <span className="text-caption">CLASSIFICATION</span>
                    <span className={`font-bold ${selectedVehicle.vehicle_category === 'Military' ? 'text-green' : selectedVehicle.vehicle_category === 'Normal' ? 'text-gray' : 'text-amber'}`}>
                      {selectedVehicle.vehicle_category?.toUpperCase() || 'UNCERTAIN'}
                    </span>
                  </div>
                  <div className="flex-between">
                    <span className="text-caption">CONFIDENCE</span>
                    <span className="text-mono">{selectedVehicle.classification_confidence ? `${(selectedVehicle.classification_confidence * 100).toFixed(1)}%` : '—'}</span>
                  </div>
                </div>

                <div style={{ background: 'var(--bg-app)', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
                  <div className="text-caption" style={{ marginBottom: '12px', letterSpacing: '0.05em', color: 'var(--color-blue)' }}>L2 SUBTYPE</div>
                  <div className="flex-between" style={{ marginBottom: '8px' }}>
                    <span className="text-caption">SUBTYPE</span>
                    <span className={`font-bold ${selectedVehicle.vehicle_category === 'Military' ? 'text-green' : 'text-muted'}`}>
                      {selectedVehicle.vehicle_category === 'Military' ? selectedVehicle.subtype?.toUpperCase() || 'UNKNOWN' : '—'}
                    </span>
                  </div>
                  <div className="flex-between">
                    <span className="text-caption">CONFIDENCE</span>
                    <span className="text-mono">—</span> {/* Assuming backend L2 confidence is not explicitly piped or merged with L1 for now */}
                  </div>
                </div>

                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px', marginTop: '8px' }}>
                  <span className="text-caption">ANPR</span>
                  <span className="text-mono font-bold">{selectedAnpr?.plate_text || '—'}</span>
                </div>
                
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">CAMERA</span>
                  <span className="text-mono">CAM-01</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">FIRST SEEN</span>
                  <span className="text-mono">{selectedVehicle.first_seen ? new Date(selectedVehicle.first_seen * 1000).toLocaleTimeString([], { hour12: false }) : 'N/A'}</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">LAST SEEN</span>
                  <span className="text-mono">{selectedVehicle.last_seen ? new Date(selectedVehicle.last_seen * 1000).toLocaleTimeString([], { hour12: false }) : 'N/A'}</span>
                </div>

              </div>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
