import React, { useState, useMemo } from 'react';

const API_URL = "http://localhost:8000";

export default function AlertCenter({ events, vlmAnalyses }) {
  // Filters
  const [sevFilter, setSevFilter] = useState('All');
  const [typeFilter, setTypeFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const [search, setSearch] = useState('');
  
  const [resolvedStatus, setResolvedStatus] = useState({});
  const [selectedEventId, setSelectedEventId] = useState(null);

  const getVlmForEvent = (eventId) => vlmAnalyses.find(v => v.event_id === eventId);

  const sortedEvents = useMemo(() => {
    return [...events].sort((a, b) => b.timestamp - a.timestamp);
  }, [events]);

  const eventTypes = useMemo(() => {
    const types = new Set(events.map(e => e.event_type));
    return Array.from(types);
  }, [events]);

  const activeAlerts = events.filter(e => !resolvedStatus[e.event_id]).length;
  // Approximating 'TODAY' as last 24h since we don't have absolute midnight sync easily here without bloat
  const todayAlerts = events.filter(e => (Date.now() / 1000) - e.timestamp < 86400).length;
  const criticalAlerts = events.filter(e => ['CRITICAL', 'HIGH'].includes((e.severity || '').toUpperCase()) && !resolvedStatus[e.event_id]).length;

  const filteredEvents = useMemo(() => {
    let result = sortedEvents;

    if (sevFilter !== 'All') {
      result = result.filter(evt => {
        const severity = (evt.severity || 'INFO').toUpperCase();
        if (sevFilter === 'Critical') return severity === 'CRITICAL' || severity === 'HIGH';
        if (sevFilter === 'Warning') return severity === 'WARNING';
        if (sevFilter === 'Information') return severity === 'INFO' || severity === 'LOW';
        return true;
      });
    }

    if (typeFilter !== 'All') {
      result = result.filter(evt => evt.event_type === typeFilter);
    }

    if (statusFilter !== 'All') {
      result = result.filter(evt => {
        const isResolved = resolvedStatus[evt.event_id] === true;
        if (statusFilter === 'Unread') return !isResolved;
        if (statusFilter === 'Resolved') return isResolved;
        return true;
      });
    }

    if (search) {
      const q = search.toLowerCase();
      result = result.filter(evt => 
        (evt.event_type || '').toLowerCase().includes(q) ||
        (evt.description || '').toLowerCase().includes(q) ||
        (evt.track_id || '').toString().includes(q)
      );
    }

    return result;
  }, [sortedEvents, sevFilter, typeFilter, statusFilter, search, resolvedStatus]);

  const toggleResolved = (e, eventId) => {
    e.stopPropagation();
    setResolvedStatus(prev => ({ ...prev, [eventId]: !prev[eventId] }));
  };

  const selectedEvent = events.find(e => e.event_id === selectedEventId);

  return (
    <div className="view-container animate-fade" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      
      {/* ==================== HEADER KPIs ==================== */}
      <div className="panel" style={{ padding: 'var(--space-3) var(--space-4)', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <span className="text-mono" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{activeAlerts}</span>
          <span className="text-caption">ACTIVE ALERTS</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className="text-mono" style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{todayAlerts}</span>
          <span className="text-caption">TODAY (24H)</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--border-subtle)', paddingLeft: 'var(--space-4)' }}>
          <span className={`text-mono ${criticalAlerts > 0 ? 'text-red' : 'text-green'}`} style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>
            {criticalAlerts}
          </span>
          <span className="text-caption">CRITICAL (UNRESOLVED)</span>
        </div>
      </div>

      {/* ==================== FILTER BAR ==================== */}
      <div className="flex-between panel" style={{ padding: '8px 16px', flexDirection: 'row', marginBottom: 'var(--space-4)', flexWrap: 'wrap', gap: '8px' }}>
        <h1 style={{ fontSize: '1.1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '8px', minWidth: 'fit-content' }}>
          <span className="text-red">●</span> ALERT CENTER
        </h1>
        
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          <input 
            type="text" 
            placeholder="Search alerts..." 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }} 
          />
          
          <select value={sevFilter} onChange={(e) => setSevFilter(e.target.value)} style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}>
            <option value="All">All Severities</option>
            <option value="Critical">Critical</option>
            <option value="Warning">Warning</option>
            <option value="Information">Information</option>
          </select>
          
          <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}>
            <option value="All">All Types</option>
            {eventTypes.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-main)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}>
            <option value="All">All Statuses</option>
            <option value="Unread">Unread</option>
            <option value="Resolved">Resolved</option>
          </select>

          <select disabled style={{ padding: '4px 8px', background: 'var(--bg-app)', color: 'var(--text-muted)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', fontSize: '0.875rem' }}>
            <option>CAM-01</option>
          </select>
        </div>
      </div>
      
      <div className="split-view" style={{ flex: 1, overflow: 'hidden', gridTemplateColumns: selectedEvent ? '1fr 400px' : '1fr' }}>
        
        {/* ==================== MAIN TABLE ==================== */}
        <div className="data-table-container" style={{ height: '100%', overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>TIME</th>
                <th>TYPE</th>
                <th>TRACK</th>
                <th>CAMERA</th>
                <th>DESCRIPTION</th>
                <th>SEVERITY</th>
                <th>STATUS</th>
              </tr>
            </thead>
            <tbody>
              {filteredEvents.length === 0 ? (
                <tr>
                  <td colSpan="7" style={{ textAlign: 'center', padding: 'var(--space-8)' }}>
                    <span className="text-mono text-muted">NO ALERTS FOUND</span>
                  </td>
                </tr>
              ) : (
                filteredEvents.map(evt => {
                  const severity = (evt.severity || 'INFO').toUpperCase();
                  const isCrit = severity === 'CRITICAL' || severity === 'HIGH';
                  const isWarn = severity === 'WARNING';
                  const isResolved = resolvedStatus[evt.event_id] === true;
                  const isSelected = selectedEventId === evt.event_id;
                  
                  let dotClass = 'dot-blue';
                  if (isCrit) dotClass = 'dot-red';
                  else if (isWarn) dotClass = 'dot-amber';

                  return (
                    <tr 
                      key={evt.event_id} 
                      onClick={() => setSelectedEventId(evt.event_id)}
                      style={{ 
                        background: isSelected ? 'var(--bg-panel-hover)' : 'transparent',
                        opacity: isResolved ? 0.5 : 1,
                        cursor: 'pointer'
                      }}
                    >
                      <td className="text-mono text-muted">
                        {new Date(evt.timestamp * 1000).toLocaleTimeString([], { hour12: false })}
                      </td>
                      <td style={{ fontWeight: '600', color: isCrit && !isResolved ? 'var(--color-red)' : 'var(--text-main)' }}>
                        {evt.event_type.replace(/_/g, ' ')}
                      </td>
                      <td className="text-mono text-blue font-bold">
                        {evt.track_id ? `#${evt.track_id}` : '-'}
                      </td>
                      <td className="text-mono text-muted">
                        CAM-01
                      </td>
                      <td style={{ maxWidth: '200px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {evt.description || '-'}
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span className={`status-dot ${dotClass}`} />
                          <span className="text-mono" style={{ fontSize: '0.75rem' }}>{severity}</span>
                        </div>
                      </td>
                      <td>
                        <button onClick={(e) => toggleResolved(e, evt.event_id)} style={{
                          padding: '2px 8px', borderRadius: '4px', border: '1px solid', cursor: 'pointer', fontSize: '0.7rem', fontWeight: 'bold',
                          background: isResolved ? 'rgba(16, 185, 129, 0.1)' : 'transparent',
                          borderColor: isResolved ? 'var(--color-green)' : 'var(--border-subtle)',
                          color: isResolved ? 'var(--color-green)' : 'var(--text-muted)'
                        }}>
                          {isResolved ? 'RESOLVED' : 'UNREAD'}
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* ==================== DETAIL DRAWER ==================== */}
        {selectedEvent && (
          <div className="panel" style={{ padding: '0', overflowY: 'auto' }}>
            <div className="flex-between" style={{ padding: '16px', borderBottom: '1px solid var(--border-subtle)' }}>
              <h2 style={{ margin: 0, fontSize: '1rem', color: 'var(--color-red)' }}>INCIDENT DETAILS</h2>
              <button onClick={() => setSelectedEventId(null)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}>×</button>
            </div>
            
            <div style={{ padding: '16px' }}>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '16px', fontSize: '0.875rem' }}>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">ALERT ID</span>
                  <span className="text-mono text-muted">{selectedEvent.event_id}</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">TIMESTAMP</span>
                  <span className="text-mono">{new Date(selectedEvent.timestamp * 1000).toLocaleString()}</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">TYPE</span>
                  <span className="font-bold">{selectedEvent.event_type}</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">SEVERITY</span>
                  <span className={`text-mono ${selectedEvent.severity === 'CRITICAL' || selectedEvent.severity === 'HIGH' ? 'text-red' : (selectedEvent.severity === 'WARNING' ? 'text-amber' : 'text-blue')}`}>
                    {selectedEvent.severity?.toUpperCase() || 'INFO'}
                  </span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">TRACK ID</span>
                  <span className="text-mono text-blue font-bold">{selectedEvent.track_id ? `#${selectedEvent.track_id}` : 'N/A'}</span>
                </div>
                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                  <span className="text-caption">CAMERA</span>
                  <span className="text-mono">CAM-01</span>
                </div>
                
                <div style={{ marginTop: '8px' }}>
                  <span className="text-caption" style={{ display: 'block', marginBottom: '8px' }}>DESCRIPTION</span>
                  <div style={{ background: 'var(--bg-app)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                    {selectedEvent.description || 'No description provided.'}
                  </div>
                </div>

                <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px', marginTop: '8px' }}>
                  <span className="text-caption">CONFIDENCE</span>
                  <span className="text-mono">{selectedEvent.confidence ? `${(selectedEvent.confidence * 100).toFixed(1)}%` : 'N/A'}</span>
                </div>

                {selectedEvent.snapshot && (
                  <div style={{ marginTop: '16px' }}>
                    <span className="text-caption" style={{ display: 'block', marginBottom: '8px' }}>AVAILABLE SNAPSHOT</span>
                    <div style={{ width: '100%', borderRadius: 'var(--radius-sm)', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
                      <img src={`${API_URL}${selectedEvent.snapshot}`} alt="Evidence" style={{ width: '100%', display: 'block' }} />
                    </div>
                  </div>
                )}

                {/* VLM ANALYSIS INJECTION */}
                {(() => {
                  const vlm = getVlmForEvent(selectedEvent.event_id);
                  if (vlm) {
                    const analysis = typeof vlm.vlm_analysis === 'object' ? vlm.vlm_analysis.analysis : vlm.vlm_analysis;
                    return (
                      <div style={{ marginTop: '16px', background: 'rgba(59, 130, 246, 0.05)', border: '1px solid rgba(59, 130, 246, 0.2)', padding: '16px', borderRadius: 'var(--radius-sm)' }}>
                        <div className="text-caption text-blue" style={{ marginBottom: '8px', fontWeight: 'bold' }}>VLM ANALYSIS</div>
                        <div style={{ color: 'var(--text-main)', fontSize: '0.875rem', lineHeight: '1.6' }}>
                          {analysis}
                        </div>
                      </div>
                    );
                  }
                  return null;
                })()}

              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
