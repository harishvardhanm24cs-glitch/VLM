import React, { useState, useMemo } from 'react';

export default function AnprLog({ events, stats }) {
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('All');
  const [sortOrder, setSortOrder] = useState('Newest');
  const [selectedTrackId, setSelectedTrackId] = useState(null);

  const anprEvents = events.filter(e => e.event_type?.includes('ANPR'));
  const activeTracks = stats?.tracks || {};

  const enrichedEvents = useMemo(() => {
    return anprEvents.map(evt => {
      const trackData = activeTracks[evt.track_id] || {};
      const plateRaw = evt.plate_text || '';
      
      // Handle low confidence or empty reads
      let plateDisplay = plateRaw;
      let plateStatus = 'SUCCESS';
      
      if (!plateRaw || plateRaw.trim() === '') {
        plateDisplay = 'NO PLATE DETECTED';
        plateStatus = 'EMPTY';
      } else if (evt.confidence && evt.confidence < 0.6) {
        // Threshold for low confidence
        plateStatus = 'LOW_CONFIDENCE';
      }

      return {
        ...evt,
        plateText: plateDisplay,
        plateStatus,
        vehicleType: evt.vehicle_type || trackData.class || 'unknown',
        vehicleCategory: evt.vehicle_category || trackData.vehicle_category || 'UNCERTAIN',
        subtype: trackData.subtype || null,
        yoloConfidence: trackData.classification_confidence || 0,
        first_seen: trackData.first_seen || null,
        last_seen: trackData.last_seen || null
      };
    });
  }, [anprEvents, activeTracks]);

  const processedEvents = useMemo(() => {
    let result = enrichedEvents;

    if (search) {
      const q = search.toLowerCase();
      result = result.filter(e => 
        e.plateText.toLowerCase().includes(q) || 
        (e.track_id || '').toString().includes(q) ||
        e.vehicleType.toLowerCase().includes(q)
      );
    }

    if (filterType !== 'All') {
      result = result.filter(e => {
        const cat = e.vehicleCategory.toUpperCase();
        if (filterType === 'Military') return cat === 'MILITARY';
        if (filterType === 'Normal') return cat === 'NORMAL';
        if (filterType === 'Uncertain') return !['MILITARY', 'NORMAL'].includes(cat);
        return true;
      });
    }

    result.sort((a, b) => {
      if (sortOrder === 'Newest') return b.timestamp - a.timestamp;
      if (sortOrder === 'Oldest') return a.timestamp - b.timestamp;
      return 0;
    });

    return result;
  }, [enrichedEvents, search, filterType, sortOrder]);

  const totalDetections = anprEvents.length;
  const successfulReads = enrichedEvents.filter(e => e.plateStatus === 'SUCCESS').length;
  const recentDetections = enrichedEvents.filter(e => (Date.now() / 1000) - e.timestamp < 3600).length; // Last 1 hour

  const selectedVehicle = selectedTrackId ? enrichedEvents.find(e => e.track_id === selectedTrackId) : null;

  return (
    <div className="view-container animate-fade" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      
      {/* ==================== TOP KPIs ==================== */}
      <div className="panel" style={{ padding: 'var(--space-3) var(--space-4)', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <span className="text-mono" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{totalDetections}</span>
          <span className="text-caption">PLATES DETECTED (TOTAL)</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className="text-mono" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{recentDetections}</span>
          <span className="text-caption">RECENT DETECTIONS (1H)</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className="text-mono text-green" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{successfulReads}</span>
          <span className="text-caption">SUCCESSFUL READS</span>
        </div>
      </div>

      {/* ==================== FILTER BAR ==================== */}
      <div className="flex-between panel" style={{ padding: '8px 16px', flexDirection: 'row', marginBottom: 'var(--space-4)' }}>
        <h1 style={{ fontSize: '1.1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="text-blue">●</span> ANPR SUBSYSTEM
        </h1>
        
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <input 
            type="text" 
            placeholder="Search plates..." 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }} 
          />
          
          <select 
            value={filterType} 
            onChange={(e) => setFilterType(e.target.value)}
            style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}
          >
            <option value="All">All Categories</option>
            <option value="Military">Military</option>
            <option value="Normal">Normal</option>
          </select>

          <select 
            value={sortOrder} 
            onChange={(e) => setSortOrder(e.target.value)}
            style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}
          >
            <option value="Newest">Newest First</option>
            <option value="Oldest">Oldest First</option>
          </select>
        </div>
      </div>

      <div className="split-view" style={{ flex: 1, overflow: 'hidden', gridTemplateColumns: selectedVehicle ? '1fr 350px' : '1fr' }}>
        
        {/* ==================== MAIN TABLE ==================== */}
        <div className="data-table-container" style={{ height: '100%', overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>TIME</th>
                <th>PLATE</th>
                <th>TRACK</th>
                <th>VEHICLE</th>
                <th>CATEGORY</th>
                <th>CONFIDENCE</th>
                <th>CAMERA</th>
              </tr>
            </thead>
            <tbody>
              {processedEvents.length === 0 ? (
                <tr>
                  <td colSpan="7" style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                    <span className="text-mono text-muted">NO ANPR RECORDS FOUND</span>
                  </td>
                </tr>
              ) : (
                processedEvents.map(evt => {
                  const cat = (evt.vehicleCategory || 'UNCERTAIN').toUpperCase();
                  const isArmy = cat === 'MILITARY';
                  const isNormal = cat === 'NORMAL';
                  
                  let badgeClass = 'badge-amber';
                  if (isArmy) badgeClass = 'badge-green';
                  else if (isNormal) badgeClass = 'badge-gray';

                  const isSelected = selectedTrackId === evt.track_id;

                  return (
                    <tr 
                      key={evt.event_id} 
                      onClick={() => setSelectedTrackId(evt.track_id)}
                      style={{ 
                        background: isSelected ? 'var(--bg-panel-hover)' : 'transparent',
                        cursor: 'pointer'
                      }}
                    >
                      <td className="text-mono text-muted">
                        {new Date(evt.timestamp * 1000).toLocaleTimeString([], { hour12: false })}
                      </td>
                      <td>
                        <span className="text-mono" style={{ 
                          fontSize: '1.1rem', fontWeight: 'bold', letterSpacing: '2px',
                          color: evt.plateStatus === 'SUCCESS' ? '#fff' : (evt.plateStatus === 'LOW_CONFIDENCE' ? 'var(--color-amber)' : 'var(--text-muted)')
                        }}>
                          {evt.plateText}
                        </span>
                        {evt.plateStatus === 'LOW_CONFIDENCE' && <div className="text-caption text-amber" style={{ fontSize: '0.65rem', marginTop: '2px' }}>LOW CONFIDENCE</div>}
                      </td>
                      <td className="text-mono text-blue font-bold">
                        {evt.track_id ? `#${evt.track_id}` : '-'}
                      </td>
                      <td style={{ textTransform: 'lowercase' }}>
                        {evt.vehicleType}
                      </td>
                      <td>
                        <span className={`badge ${badgeClass}`}>
                          {cat}
                        </span>
                      </td>
                      <td className="text-mono text-muted">
                        {evt.confidence ? `${(evt.confidence * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="text-mono text-muted">
                        CAM-01
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* ==================== VEHICLE TRACEABILITY DRAWER ==================== */}
        {selectedVehicle && (
          <div className="panel" style={{ padding: '0', overflowY: 'auto' }}>
            <div className="flex-between" style={{ padding: '16px', borderBottom: '1px solid var(--border-subtle)' }}>
              <h2 style={{ margin: 0, fontSize: '1rem', color: 'var(--color-blue)' }}>VEHICLE #{selectedVehicle.track_id}</h2>
              <button onClick={() => setSelectedTrackId(null)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}>×</button>
            </div>
            
            <div style={{ padding: '16px' }}>
              
              <div style={{ marginBottom: '24px', background: 'var(--bg-app)', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                <div className="text-caption" style={{ marginBottom: '8px', letterSpacing: '0.05em' }}>DETECTED PLATE</div>
                <div className="text-mono" style={{ fontSize: '1.5rem', fontWeight: 'bold', letterSpacing: '2px', color: selectedVehicle.plateStatus === 'SUCCESS' ? '#fff' : (selectedVehicle.plateStatus === 'LOW_CONFIDENCE' ? 'var(--color-amber)' : 'var(--text-muted)') }}>
                  {selectedVehicle.plateText}
                </div>
                {selectedVehicle.plateStatus === 'LOW_CONFIDENCE' && (
                  <div className="text-caption text-amber" style={{ marginTop: '4px' }}>LOW CONFIDENCE OCR READ</div>
                )}
                <div className="text-mono text-muted" style={{ fontSize: '0.75rem', marginTop: '8px' }}>
                  ANPR CONFIDENCE: {selectedVehicle.confidence ? `${(selectedVehicle.confidence * 100).toFixed(1)}%` : '—'}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '16px', fontSize: '0.875rem' }}>
                
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">TRACK ID</span>
                  <span className="text-mono text-blue font-bold">#{selectedVehicle.track_id}</span>
                </div>
                
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">YOLO CLASS</span>
                  <span className="font-bold" style={{ textTransform: 'uppercase' }}>{selectedVehicle.vehicleType}</span>
                </div>
                
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">CATEGORY (L1)</span>
                  <span className={`font-bold ${selectedVehicle.vehicleCategory?.toUpperCase() === 'MILITARY' ? 'text-green' : selectedVehicle.vehicleCategory?.toUpperCase() === 'NORMAL' ? 'text-gray' : 'text-amber'}`}>
                    {selectedVehicle.vehicleCategory?.toUpperCase()}
                  </span>
                </div>
                
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">SUBTYPE (L2)</span>
                  <span className={`font-bold ${selectedVehicle.vehicleCategory?.toUpperCase() === 'MILITARY' ? 'text-green' : 'text-muted'}`}>
                    {selectedVehicle.vehicleCategory?.toUpperCase() === 'MILITARY' ? (selectedVehicle.subtype?.toUpperCase() || 'UNKNOWN') : '—'}
                  </span>
                </div>

                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">L1 CONFIDENCE</span>
                  <span className="text-mono">{selectedVehicle.yoloConfidence ? `${(selectedVehicle.yoloConfidence * 100).toFixed(1)}%` : '—'}</span>
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
