import React, { useState, useEffect } from 'react';
import './index.css';

const API_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws";
const LOITERING_THRESHOLD_SECONDS = 900; // 15 minutes

function App() {
  const [sysStatus, setSysStatus] = useState({ backend: "OFFLINE", pipeline: false });
  const [wsConnected, setWsConnected] = useState(false);
  
  // Real-time metrics
  const [stats, setStats] = useState({ person_count: 0, vehicle_count: 0, object_count: 0, tracks: {} });
  const [events, setEvents] = useState([]);
  const [vlmAnalyses, setVlmAnalyses] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(null);
  
  // Loitering Popup State
  const [loiteringPopup, setLoiteringPopup] = useState(null);
  const [seenLoiteringEvents, setSeenLoiteringEvents] = useState(new Set());
  
  // Vehicle Alert Popup State
  const [vehicleAlertPopup, setVehicleAlertPopup] = useState(null);
  const [seenVehicleEvents, setSeenVehicleEvents] = useState(new Set());
  
  // Sound toggle
  const [soundEnabled, setSoundEnabled] = useState(true);

  // Health check polling
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/api/health`);
        const data = await res.json();
        setSysStatus(data);
      } catch (e) {
        setSysStatus({ FastAPI: "OFFLINE" });
      }
    };
    checkHealth();
    const intv = setInterval(checkHealth, 5000);
    return () => clearInterval(intv);
  }, []);

  // Play alert sound function
  const playAlertSound = (severity) => {
    if (!soundEnabled) return;
    try {
      // Create simple synth beep to avoid needing audio files
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      
      if (severity === 'CRITICAL' || severity === 'HIGH') {
        oscillator.type = 'square';
        oscillator.frequency.value = 800; // High pitch
        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        oscillator.start();
        setTimeout(() => oscillator.stop(), 500);
      } else {
        oscillator.type = 'sine';
        oscillator.frequency.value = 400; // Low pitch
        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        oscillator.start();
        setTimeout(() => oscillator.stop(), 200);
      }
    } catch(e) { console.error("Audio play blocked", e); }
  };

  // WebSocket connection with Auto-Reconnect
  useEffect(() => {
    let ws = null;
    let reconnectTimeout = null;

    const connect = () => {
      ws = new WebSocket(WS_URL);
      
      ws.onopen = () => setWsConnected(true);

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          
          if (msg.type === "STATS") {
            setStats(msg.data);
          } else if (msg.type === "EVENT") {
            const newEvent = msg.data;
            setEvents(prev => {
              if (prev.some(e => e.event_id === newEvent.event_id)) return prev;
              
              // New event logic
              if (newEvent.severity === 'CRITICAL' || newEvent.severity === 'HIGH') {
                playAlertSound(newEvent.severity);
              }
              
              return [newEvent, ...prev];
            });
            
            // Loitering Popup Logic
            if ((newEvent.event_type === "SUSPICIOUS_LOITERING" || newEvent.event_type?.includes("LOITERING")) && 
                newEvent.duration >= LOITERING_THRESHOLD_SECONDS) {
              setSeenLoiteringEvents(prev => {
                if (!prev.has(newEvent.event_id)) {
                  const newSet = new Set(prev);
                  newSet.add(newEvent.event_id);
                  setLoiteringPopup(newEvent);
                  playAlertSound('HIGH');
                  return newSet;
                }
                return prev;
              });
            }
            
            // Vehicle Alert Popup Logic
            if (newEvent.event_type === "NORMAL_VEHICLE_ALERT" || newEvent.event_type === "SUSPICIOUS_VEHICLE") {
              setSeenVehicleEvents(prev => {
                if (!prev.has(newEvent.event_id)) {
                  const newSet = new Set(prev);
                  newSet.add(newEvent.event_id);
                  setVehicleAlertPopup(newEvent);
                  playAlertSound('HIGH');
                  return newSet;
                }
                return prev;
              });
            }
          } else if (msg.type === "VLM") {
            setVlmAnalyses(prev => {
              if (prev.some(v => v.event_id === msg.data.event_id)) return prev;
              return [msg.data, ...prev];
            });
          }
        } catch (e) {}
      };

      ws.onclose = () => {
        setWsConnected(false);
        reconnectTimeout = setTimeout(connect, 3000);
      };

      ws.onerror = (err) => ws.close();
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [soundEnabled]);

  // Initial Fetch & Deduplication
  useEffect(() => {
    fetch(`${API_URL}/api/events`).then(r => r.json()).then(data => {
      const uniqueEvents = [];
      const ids = new Set();
      for (const e of data.reverse()) {
        if (!ids.has(e.event_id)) {
          uniqueEvents.push(e);
          ids.add(e.event_id);
        }
      }
      setEvents(uniqueEvents);
    }).catch(() => {});
    
    fetch(`${API_URL}/api/vlm`).then(r => r.json()).then(data => {
      const uniqueVlm = [];
      const ids = new Set();
      for (const v of data.reverse()) {
        if (!ids.has(v.event_id)) {
          uniqueVlm.push(v);
          ids.add(v.event_id);
        }
      }
      setVlmAnalyses(uniqueVlm);
    }).catch(() => {});
  }, []);

  const getVlmForEvent = (eventId) => vlmAnalyses.find(v => v.event_id === eventId);
  
  // Data extraction for panels
  const activeTracks = stats.tracks || {};
  const persons = Object.values(activeTracks).filter(t => t.class === 'person');
  const vehicles = Object.values(activeTracks).filter(t => ['car', 'truck', 'bus', 'motorcycle', 'four-wheeler', 'two-wheeler'].includes(t.class?.toLowerCase()));
  const objects = Object.values(activeTracks).filter(t => t.class !== 'person' && !['car', 'truck', 'bus', 'motorcycle', 'four-wheeler', 'two-wheeler'].includes(t.class?.toLowerCase()));
  const anprEvents = events.filter(e => e.event_type?.includes('ANPR'));

  return (
    <div className="soc-container">
      
      {/* 1. SYSTEM STATUS BAR */}
      <div className="system-status-bar">
        <div className="status-title">SYSTEM STATUS</div>
        <div className="status-items">
          <span className={`status-item ${sysStatus.Video === 'ONLINE' ? 'online' : 'offline'}`}>● CAMERA</span>
          <span className={`status-item ${sysStatus['CCTV Edge'] === 'ONLINE' ? 'online' : 'offline'}`}>● AI ENGINE</span>
          <span className={`status-item ${sysStatus.VLM === 'ONLINE' ? 'online' : 'offline'}`}>● VLM</span>
          <span className={`status-item ${sysStatus['CCTV Edge'] === 'ONLINE' ? 'online' : 'offline'}`}>● ANPR</span>
          <span className={`status-item ${sysStatus.PostgreSQL === 'ONLINE' ? 'online' : 'offline'}`}>● DATABASE</span>
          <span className={`status-item ${wsConnected ? 'online' : 'offline'}`}>● WEBSOCKET</span>
        </div>
        <button className="sound-btn" onClick={() => setSoundEnabled(!soundEnabled)}>
          {soundEnabled ? '🔊 SOUND ON' : '🔇 SOUND OFF'}
        </button>
      </div>

      {/* 2. LIVE COUNTERS */}
      <div className="live-counters">
        <div className="counter-card">
          <div className="counter-icon">👤</div>
          <div className="counter-details">
            <div className="counter-title">PERSONS</div>
            <div className="counter-value">{stats.person_count || 0} PERSON{stats.person_count !== 1 ? 'S' : ''}</div>
          </div>
        </div>
        <div className="counter-card">
          <div className="counter-icon">🚗</div>
          <div className="counter-details">
            <div className="counter-title">VEHICLES</div>
            <div className="counter-value">{stats.vehicle_count || 0} VEHICLE{stats.vehicle_count !== 1 ? 'S' : ''}</div>
          </div>
        </div>
        <div className="counter-card">
          <div className="counter-icon">📦</div>
          <div className="counter-details">
            <div className="counter-title">OBJECTS</div>
            <div className="counter-value">{stats.object_count || 0} OBJECT{stats.object_count !== 1 ? 'S' : ''}</div>
          </div>
        </div>
        <div className="counter-card alert-counter">
          <div className="counter-icon">🚨</div>
          <div className="counter-details">
            <div className="counter-title">ACTIVE ALERTS</div>
            <div className="counter-value">{events.length}</div>
          </div>
        </div>
      </div>

      <div className="main-grid">
        {/* 3. CAMERA VIEW (LEFT) */}
        <div className="camera-section">
          <div className="panel camera-panel">
            <div className="panel-header">
              <h3>LIVE CAMERA</h3>
              <span className={`camera-badge ${sysStatus.Video === 'ONLINE' ? 'live' : 'offline'}`}>
                ● {sysStatus.Video === 'ONLINE' ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>
            <div className="video-wrapper">
              {sysStatus.Video === 'ONLINE' ? (
                <img src={`${API_URL}/api/stream`} alt="Live CCTV Feed" className="live-feed" />
              ) : (
                <div className="offline-screen">
                  ⚠ CAMERA STREAM LOST<br/>Reconnecting...
                </div>
              )}
              <div className="camera-overlay-info">
                <span>CAM 01 — MAIN</span>
                <span>AI: ACTIVE</span>
              </div>
            </div>
          </div>
          
          {/* BOTTOM PANELS */}
          <div className="bottom-panels">
            {/* PERSON DETECTION */}
            <div className="panel sub-panel">
              <h4>PERSON DETECTION</h4>
              <div className="panel-content">
                <div className="sub-count">CURRENT PERSONS: {persons.length}</div>
                {persons.length > 0 ? persons.map(p => (
                  <div key={p.track_id} className="track-item">
                    <strong>PERSON #{p.track_id}</strong>
                    <span>● ACTIVE</span>
                    <span>Duration: {p.time_in_frame ? p.time_in_frame.toFixed(1) : 0}s</span>
                  </div>
                )) : <div className="empty-state">No persons detected.</div>}
              </div>
            </div>

            {/* VEHICLE DETECTION */}
            <div className="panel sub-panel">
              <h4>VEHICLE CLASSIFICATION & ANPR</h4>
              <div className="panel-content">
                <div className="sub-count">
                  DETECTED: {vehicles.length} | 
                  MILITARY: {vehicles.filter(v => v.vehicle_category === 'Military').length} | 
                  NORMAL: {vehicles.filter(v => v.vehicle_category === 'Normal').length} | 
                  UNCERTAIN: {vehicles.filter(v => v.vehicle_category === 'uncertain' || !v.vehicle_category).length}
                </div>
                {vehicles.length > 0 ? vehicles.map(v => {
                  const isArmy = v.vehicle_category === 'Military';
                  const isNormal = v.vehicle_category === 'Normal';
                  const isUncertain = !isArmy && !isNormal;
                  
                  // Alert logic: Normal civilian vehicles are Alerts. Army vehicles are No Alert. Uncertain is No Alert.
                  const isAlert = isNormal;
                  
                  return (
                  <div key={v.track_id} className="track-item" style={{display: 'flex', flexDirection: 'column', gap: '4px', padding: '10px', background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.1)', marginBottom: '8px'}}>
                    <div style={{fontWeight: 'bold', fontSize: '1.1em'}}>
                      {isArmy ? '🪖 MILITARY VEHICLE' : 
                       isNormal ? '🚨 NORMAL VEHICLE' : 
                       '❓ CAN NOT DETECT'}
                    </div>
                    <div style={{marginTop: '4px', lineHeight: '1.4'}}>
                      <div style={{color: '#aaa'}}>YOLO Type: <span style={{color: '#fff'}}>{(v.object_type || v.class || 'UNKNOWN').toUpperCase()}</span></div>
                      <div style={{color: '#aaa'}}>Category: <span style={{color: '#fff'}}>{isUncertain ? 'CAN NOT DETECT' : (v.vehicle_category || '').toUpperCase()}</span></div>
                      
                      {isArmy && (
                        <div style={{color: '#aaa'}}>Subtype: <span style={{color: '#fff'}}>{(v.subtype || 'N/A').toUpperCase()}</span></div>
                      )}
                      
                      {v.classification_confidence > 0 && <div style={{color: '#aaa'}}>Confidence: <span style={{color: '#fff'}}>{(v.classification_confidence * 100).toFixed(0)}%</span></div>}
                      {isArmy && v.subtype_confidence > 0 && <div style={{color: '#aaa'}}>Subtype Conf: <span style={{color: '#fff'}}>{(v.subtype_confidence * 100).toFixed(0)}%</span></div>}
                    </div>
                    <div style={{marginTop: '8px', fontWeight: 'bold', color: isAlert ? '#ff4444' : '#00ff88'}}>
                      STATUS: {isAlert ? 'ALERT' : 'NO ALERT'}
                    </div>
                  </div>
                )}) : <div className="empty-state">No vehicles detected.</div>}
                
                {anprEvents.length > 0 && (
                  <div className="anpr-latest">
                    <h5>LATEST PLATE</h5>
                    <div className="plate-box">{anprEvents[0].plate_text || anprEvents[0].description}</div>
                  </div>
                )}
              </div>
            </div>

            {/* OBJECT DETECTION */}
            <div className="panel sub-panel">
              <h4>OBJECT DETECTION</h4>
              <div className="panel-content">
                <div className="sub-count">CURRENT OBJECTS: {objects.length}</div>
                {objects.length > 0 ? objects.map(o => (
                  <div key={o.track_id} className="track-item">
                    <strong>{o.class?.toUpperCase()}</strong>
                    <span>Conf: {(o.confidence * 100).toFixed(1)}%</span>
                  </div>
                )) : <div className="empty-state">No objects detected.</div>}
              </div>
            </div>

            {/* VLM INTELLIGENCE */}
            <div className="panel sub-panel vlm-panel">
              <h4>🧠 VLM INTELLIGENCE</h4>
              <div className="panel-content">
                {vlmAnalyses.length > 0 ? (
                  <div className="latest-vlm">
                    <div className="vlm-status">
                       Status: {typeof vlmAnalyses[0].vlm_analysis === 'object' && vlmAnalyses[0].vlm_analysis.object_in_hand 
                       ? 'CONFIRMED' : 'UNCERTAIN'}
                    </div>
                    {typeof vlmAnalyses[0].vlm_analysis === 'object' ? (
                      <>
                        <div className="vlm-threat">Threat: {vlmAnalyses[0].vlm_analysis.threat_level}</div>
                        <div className="vlm-text">"{vlmAnalyses[0].vlm_analysis.analysis}"</div>
                      </>
                    ) : (
                      <div className="vlm-text">"{vlmAnalyses[0].vlm_analysis}"</div>
                    )}
                  </div>
                ) : <div className="empty-state">No AI analysis yet.</div>}
              </div>
            </div>
          </div>
          
          {/* EVENT TIMELINE */}
          <div className="panel timeline-panel">
            <h4>EVENT TIMELINE</h4>
            <div className="timeline-container">
              {events.slice(0, 10).map(e => (
                <div key={e.event_id} className="timeline-item">
                  <span className="time">{new Date(e.timestamp * 1000).toLocaleTimeString()}</span>
                  <span className="timeline-desc">{e.event_type.replace('_', ' ')}: {e.description || `ID #${e.track_id}`}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 4. ACTIVE ALERTS (RIGHT) */}
        <div className="alerts-section">
          <div className="panel alerts-panel">
            <h3>ACTIVE ALERTS</h3>
            {!wsConnected && (
              <div className="ws-offline">⚠ WEBSOCKET DISCONNECTED - Attempting reconnection...</div>
            )}
            <div className="alerts-list">
              {events.length === 0 ? (
                <div className="empty-state">No events detected.</div>
              ) : (
                events.map(evt => {
                  const vlm = getVlmForEvent(evt.event_id);
                  const severity = evt.severity?.toLowerCase() || 'info';
                  return (
                    <div key={evt.event_id} className={`alert-card severity-${severity}`} onClick={() => setSelectedEvent({ ...evt, vlm })}>
                      <div className="alert-header">
                        <span className="alert-type">
                          {evt.event_type === 'SUSPICIOUS_LOITERING' ? '🚨 LOITERING DETECTED' : 
                           evt.event_type === 'VIRTUAL_FENCE_INTRUSION' ? '🚨 VIRTUAL FENCE INTRUSION' :
                           evt.event_type?.includes('ANPR') ? '🚗 ANPR DETECTION' :
                           `🔴 ${evt.event_type.replace(/_/g, ' ')}`}
                        </span>
                        <span className="alert-severity">{severity.toUpperCase()}</span>
                      </div>
                      
                      <div className="alert-body">
                        {evt.track_id && <div>ID: {evt.track_id}</div>}
                        <div>Camera: CAM 01</div>
                        {evt.duration > 0 && <div>Duration: {evt.duration}s</div>}
                        {evt.description && <div className="alert-desc">{evt.description}</div>}
                      </div>

                      <div className="alert-footer">
                        <button className="view-btn">[ VIEW EVIDENCE ]</button>
                        {vlm && <span className="vlm-badge">🧠 VLM</span>}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>

      {/* LOITERING POPUP */}
      {loiteringPopup && (
        <div className="loitering-popup-overlay">
          <div className="loitering-popup">
            <div className="popup-header">🚨 LOITERING ALERT</div>
            <div className="popup-body">
              <p>Person #{loiteringPopup.track_id} has remained stationary for more than {LOITERING_THRESHOLD_SECONDS / 60} minutes.</p>
              <ul>
                <li>Camera: CAM 01</li>
                <li>Duration: {Math.floor(loiteringPopup.duration / 60)}m {Math.floor(loiteringPopup.duration % 60)}s</li>
              </ul>
            </div>
            <div className="popup-footer">
              <button onClick={() => { setSelectedEvent(loiteringPopup); setLoiteringPopup(null); }}>VIEW EVIDENCE</button>
              <button className="close-btn" onClick={() => setLoiteringPopup(null)}>CLOSE</button>
            </div>
          </div>
        </div>
      )}

      {/* VEHICLE ALERT POPUP */}
      {vehicleAlertPopup && (
        <div className="loitering-popup-overlay">
          <div className="loitering-popup">
            <div className="popup-header">🚨 VEHICLE ALERT</div>
            <div className="popup-body">
              <p>Normal vehicle detected in monitored zone.</p>
              <ul>
                <li>Vehicle ID: #{vehicleAlertPopup.track_id}</li>
                <li>Category confidence: {vehicleAlertPopup.classification_confidence ? (vehicleAlertPopup.classification_confidence * 100).toFixed(1) : (vehicleAlertPopup.confidence ? (vehicleAlertPopup.confidence * 100).toFixed(1) : '?')}%</li>
                {vehicleAlertPopup.plate_text && <li>Plate: {vehicleAlertPopup.plate_text}</li>}
                <li>Camera: CAM 01</li>
              </ul>
            </div>
            <div className="popup-footer">
              <button onClick={() => { setSelectedEvent(vehicleAlertPopup); setVehicleAlertPopup(null); }}>VIEW EVIDENCE</button>
              <button className="close-btn" onClick={() => setVehicleAlertPopup(null)}>CLOSE</button>
            </div>
          </div>
        </div>
      )}

      {/* EVIDENCE MODAL */}
      {selectedEvent && (
        <div className="modal-overlay" onClick={() => setSelectedEvent(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedEvent(null)}>×</button>
            <h2>EVIDENCE VIEWER</h2>
            
            <div className="modal-details">
              <div><strong>Event Type:</strong> {selectedEvent.event_type}</div>
              <div><strong>ID:</strong> {selectedEvent.track_id}</div>
              <div><strong>Time:</strong> {new Date(selectedEvent.timestamp * 1000).toLocaleString()}</div>
              <div><strong>Camera:</strong> CAM 01</div>
              {selectedEvent.description && <div><strong>Desc:</strong> {selectedEvent.description}</div>}
              {selectedEvent.plate_text && <div><strong>Plate:</strong> {selectedEvent.plate_text}</div>}
            </div>

            {selectedEvent.snapshot && (
              <div className="modal-image">
                <img src={`${API_URL}${selectedEvent.snapshot}`} alt="Evidence" />
              </div>
            )}

            {selectedEvent.vlm && (
              <div className="modal-vlm-panel">
                <h3>🧠 VLM ANALYSIS</h3>
                {typeof selectedEvent.vlm.vlm_analysis === 'object' ? (
                  <div className="vlm-grid">
                    <div><strong>Object in Hand:</strong> {selectedEvent.vlm.vlm_analysis.object_in_hand?.toString()}</div>
                    <div><strong>Threat:</strong> {selectedEvent.vlm.vlm_analysis.threat_level}</div>
                    <div className="full-width"><strong>Analysis:</strong> <em>{selectedEvent.vlm.vlm_analysis.analysis}</em></div>
                  </div>
                ) : (
                  <div>{selectedEvent.vlm.vlm_analysis}</div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
