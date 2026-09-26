import React, { useState } from 'react';

export default function HumanMonitoring({ stats, expressions }) {
  const [filter, setFilter] = useState('ALL');
  const [selectedPersonId, setSelectedPersonId] = useState(null);
  
  const activeTracks = stats?.tracks || {};
  const persons = Object.values(activeTracks).filter(t => t.class === 'person');

  // KPI Calculations
  const activePeopleCount = persons.length;
  const facesDetectedCount = persons.filter(p => expressions[p.track_id] && expressions[p.track_id].status !== 'NO_FACE').length;
  const stationaryCount = persons.filter(p => p.loitering).length;

  // Apply filters
  const filteredPersons = persons.filter(p => {
    if (filter === 'ALL') return true;
    if (filter === 'STATIONARY') return p.loitering;
    const exprData = expressions[p.track_id];
    const expr = exprData?.status !== 'NO_FACE' ? exprData?.expression?.toUpperCase() : '';
    if (filter === 'ANOMALOUS') return ['ANGRY', 'FEAR', 'DISGUST'].includes(expr);
    return true;
  });

  const selectedPerson = persons.find(p => p.track_id === selectedPersonId);
  const selectedExprData = selectedPersonId ? expressions[selectedPersonId] : null;

  return (
    <div className="view-container animate-fade" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      
      {/* ==================== TOP: COMPACT KPIs ==================== */}
      <div className="panel" style={{ padding: 'var(--space-3) var(--space-4)', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <span className="text-mono" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{activePeopleCount}</span>
          <span className="text-caption">ACTIVE PEOPLE</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className="text-mono" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{facesDetectedCount}</span>
          <span className="text-caption">FACES DETECTED</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className={`text-mono ${stationaryCount > 0 ? 'text-red' : 'text-green'}`} style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>
            {stationaryCount}
          </span>
          <span className="text-caption">STATIONARY ALERTS</span>
        </div>
      </div>

      {/* HEADER BAR FOR TABLE */}
      <div className="flex-between panel" style={{ padding: '8px 16px', flexDirection: 'row', marginBottom: 'var(--space-4)' }}>
        <h1 style={{ fontSize: '1.1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="text-blue">●</span> HUMAN MONITORING
        </h1>
        
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span className="text-caption">FILTER:</span>
          <select 
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}
          >
            <option value="ALL">ALL TRACKS</option>
            <option value="STATIONARY">STATIONARY ONLY</option>
            <option value="ANOMALOUS">ANOMALOUS EXPRESSIONS</option>
          </select>
        </div>
      </div>

      <div className="split-view" style={{ flex: 1, overflow: 'hidden', gridTemplateColumns: selectedPerson ? '1fr 350px' : '1fr' }}>
        
        {/* ==================== MAIN AREA: DATA TABLE ==================== */}
        <div className="data-table-container" style={{ height: '100%', overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>TRACK</th>
                <th>STATUS</th>
                <th>RECOGNITION</th>
                <th>IDENTITY</th>
                <th>EXPRESSION</th>
                <th>MOVEMENT</th>
                <th>DURATION</th>
                <th>CAMERA</th>
                <th>LAST SEEN</th>
              </tr>
            </thead>
            <tbody>
              {filteredPersons.length === 0 ? (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                    <span className="text-mono text-muted">NO DATA</span>
                  </td>
                </tr>
              ) : (
                filteredPersons.map(p => {
                  const exprData = expressions[p.track_id];
                  const faceStatus = p.face_detected ? 'DETECTED' : 'NO FACE';
                  const expr = faceStatus === 'DETECTED' ? (exprData?.expression?.toUpperCase() || 'UNCERTAIN') : '-';
                  
                  let exprClass = 'text-gray';
                  if (['ANGRY', 'FEAR', 'DISGUST'].includes(expr)) exprClass = 'text-red font-bold';
                  else if (['HAPPY', 'CALM', 'NEUTRAL'].includes(expr)) exprClass = 'text-green';
                  else if (['SAD', 'SURPRISE'].includes(expr)) exprClass = 'text-amber';
                  
                  const matchStatus = p.match_status || 'PROCESSING';
                  const isMatched = matchStatus === 'MATCHED';
                  const isUnknown = matchStatus === 'UNKNOWN' || matchStatus === 'UNCERTAIN';
                  
                  let recClass = 'text-muted';
                  if (isMatched) recClass = 'text-green font-bold';
                  else if (isUnknown) recClass = 'text-main';
                  else if (matchStatus === 'NO_FACE') recClass = 'text-muted';

                  const identity = isMatched ? p.identity : '—';

                  return (
                    <tr key={p.track_id} onClick={() => setSelectedPersonId(p.track_id)} style={{ cursor: 'pointer', background: selectedPersonId === p.track_id ? 'var(--bg-panel-hover)' : 'transparent' }}>
                      <td className="text-mono text-blue font-bold">#{p.track_id}</td>
                      <td>
                        <span className="badge badge-green">ACTIVE</span>
                      </td>
                      <td className={recClass}>{matchStatus}</td>
                      <td className="font-bold">{identity}</td>
                      <td className={exprClass}>
                        {expr}
                      </td>
                      <td>
                        {p.loitering ? (
                          <span className="text-red font-bold">STATIONARY</span>
                        ) : (
                          <span className="text-muted">MOVING</span>
                        )}
                      </td>
                      <td className="text-mono text-muted">
                        {p.duration ? `${p.duration.toFixed(1)}s` : '-'}
                      </td>
                      <td className="text-mono text-muted">
                        CAM-01
                      </td>
                      <td className="text-mono text-muted">
                        {p.last_seen ? new Date(p.last_seen * 1000).toLocaleTimeString([], { hour12: false }) : new Date().toLocaleTimeString([], { hour12: false })}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* ==================== PERSON DETAIL DRAWER ==================== */}
        {selectedPerson && (
          <div className="panel" style={{ padding: '0', overflowY: 'auto' }}>
            <div className="flex-between" style={{ padding: '16px', borderBottom: '1px solid var(--border-subtle)' }}>
              <h2 style={{ margin: 0, fontSize: '1rem', color: 'var(--color-blue)' }}>PERSON #{selectedPerson.track_id}</h2>
              <button onClick={() => setSelectedPersonId(null)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}>×</button>
            </div>
            
            <div style={{ padding: '16px' }}>
              
              {/* STATIONARY ALERT WARNING */}
              {selectedPerson.loitering && (
                <div style={{ background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid var(--color-red)', padding: '12px', borderRadius: '4px', marginBottom: '24px' }}>
                  <div className="text-red" style={{ fontWeight: 'bold', fontSize: '0.85rem', marginBottom: '8px', letterSpacing: '0.05em' }}>
                    ⚠ STATIONARY PERSON
                  </div>
                  <div className="text-mono text-main" style={{ fontSize: '0.75rem', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <span>Track #{selectedPerson.track_id}</span>
                    <span>Duration: {selectedPerson.duration?.toFixed(1)}s</span>
                    <span>Camera: CAM-01</span>
                  </div>
                </div>
              )}

              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '16px', fontSize: '0.875rem' }}>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">TRACK ID</span>
                  <span className="text-mono text-blue font-bold">#{selectedPerson.track_id}</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">CAMERA</span>
                  <span className="text-mono">CAM-01</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">MOVEMENT STATE</span>
                  <span className={selectedPerson.loitering ? 'text-red font-bold' : 'text-main'}>
                    {selectedPerson.loitering ? 'STATIONARY' : 'MOVING'}
                  </span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">DURATION</span>
                  <span className="text-mono">{selectedPerson.duration ? `${selectedPerson.duration.toFixed(1)}s` : 'N/A'}</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">FIRST SEEN</span>
                  <span className="text-mono">{selectedPerson.first_seen ? new Date(selectedPerson.first_seen * 1000).toLocaleTimeString([], { hour12: false }) : 'N/A'}</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">LAST SEEN</span>
                  <span className="text-mono">{selectedPerson.last_seen ? new Date(selectedPerson.last_seen * 1000).toLocaleTimeString([], { hour12: false }) : 'N/A'}</span>
                </div>

                {/* L6 FACE RECOGNITION CARD */}
                <div style={{ marginTop: '16px', background: 'var(--bg-app)', border: '1px solid var(--border-subtle)', padding: '16px', borderRadius: 'var(--radius-sm)' }}>
                  <div className="text-caption" style={{ marginBottom: '16px', letterSpacing: '0.05em', color: 'var(--color-blue)' }}>L6 FACE RECOGNITION</div>
                  <div className="flex-between" style={{ marginBottom: '12px' }}>
                    <span className="text-caption">FACE</span>
                    <span className="text-mono">
                      {selectedPerson.face_detected ? 'DETECTED' : 'NO FACE'}
                    </span>
                  </div>
                  <div className="flex-between" style={{ marginBottom: '12px' }}>
                    <span className="text-caption">RECOGNITION</span>
                    <span className={selectedPerson.match_status === 'MATCHED' ? 'text-green font-bold' : 'text-main'}>
                      {selectedPerson.match_status || 'PROCESSING'}
                    </span>
                  </div>
                  <div className="flex-between" style={{ marginBottom: '12px' }}>
                    <span className="text-caption">IDENTITY</span>
                    <span className="font-bold">
                      {selectedPerson.match_status === 'MATCHED' ? selectedPerson.identity : '—'}
                    </span>
                  </div>
                  <div className="flex-between">
                    <span className="text-caption">SIMILARITY</span>
                    <span className="text-mono text-muted">
                      {selectedPerson.similarity > 0 ? `${(selectedPerson.similarity * 100).toFixed(1)}%` : '—'}
                    </span>
                  </div>
                </div>

                <div style={{ marginTop: '16px', background: 'var(--bg-app)', border: '1px solid var(--border-subtle)', padding: '16px', borderRadius: 'var(--radius-sm)' }}>
                  <div className="text-caption" style={{ marginBottom: '16px', letterSpacing: '0.05em', color: 'var(--color-blue)' }}>L5 FACIAL ANALYSIS</div>
                  <div className="flex-between" style={{ marginBottom: '12px' }}>
                    <span className="text-caption">FACE STATUS</span>
                    <span className="text-mono">
                      {selectedExprData?.status === 'NO_FACE' ? 'NO FACE' : (selectedExprData?.status === 'LOW_QUALITY' ? 'LOW QUALITY' : (selectedExprData ? 'DETECTED' : 'UNCERTAIN'))}
                    </span>
                  </div>
                  <div className="flex-between" style={{ marginBottom: '12px' }}>
                    <span className="text-caption">FACIAL EXPRESSION</span>
                    <span className="font-bold">
                      {selectedExprData?.status !== 'NO_FACE' ? (selectedExprData?.expression?.toUpperCase() || 'UNCERTAIN') : '-'}
                    </span>
                  </div>
                  <div className="flex-between">
                    <span className="text-caption">CONFIDENCE</span>
                    <span className="text-mono">
                      {selectedExprData?.confidence ? `${(selectedExprData.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                </div>

              </div>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
