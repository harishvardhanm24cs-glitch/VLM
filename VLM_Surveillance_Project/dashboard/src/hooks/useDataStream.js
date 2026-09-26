import { useState, useEffect } from 'react';

const WS_URL = "ws://localhost:8000/ws";
const LOITERING_THRESHOLD_SECONDS = 900;

export function useDataStream(setEvents, setVlmAnalyses, setExpressions, soundEnabled) {
  const [wsConnected, setWsConnected] = useState(false);
  const [stats, setStats] = useState({ person_count: 0, vehicle_count: 0, object_count: 0, tracks: {} });
  const [loiteringPopup, setLoiteringPopup] = useState(null);
  const [vehicleAlertPopup, setVehicleAlertPopup] = useState(null);
  const [seenLoiteringEvents, setSeenLoiteringEvents] = useState(new Set());
  const [seenVehicleEvents, setSeenVehicleEvents] = useState(new Set());

  const playAlertSound = (severity) => {
    return; // Disabled by user request, ESP32 buzzer handles all sounds
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();
      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      
      if (severity === 'CRITICAL' || severity === 'HIGH') {
        oscillator.type = 'square';
        oscillator.frequency.value = 800;
        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        oscillator.start();
        setTimeout(() => oscillator.stop(), 500);
      } else {
        oscillator.type = 'sine';
        oscillator.frequency.value = 400;
        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        oscillator.start();
        setTimeout(() => oscillator.stop(), 200);
      }
    } catch(e) {}
  };

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
            setExpressions(prev => {
              const activeIds = Object.keys(msg.data.tracks || {});
              const newExp = { ...prev };
              let changed = false;
              for (const id in newExp) {
                if (!activeIds.includes(id.toString())) {
                  delete newExp[id];
                  changed = true;
                }
              }
              return changed ? newExp : prev;
            });
          } else if (msg.event_type === "FACE_EXPRESSION") {
            setExpressions(prev => ({ ...prev, [msg.track_id]: msg }));
          } else if (msg.type === "EVENT") {
            const newEvent = msg.data;
            setEvents(prev => {
              if (prev.some(e => e.event_id === newEvent.event_id)) return prev;
              if (newEvent.severity === 'CRITICAL' || newEvent.severity === 'HIGH') {
                playAlertSound(newEvent.severity);
              }
              return [newEvent, ...prev];
            });
            
            if (newEvent.event_type === "STATIONARY_PERSON" || newEvent.event_type === "SUSPICIOUS_LOITERING" || newEvent.event_type?.includes("LOITERING")) {
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

      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [soundEnabled, setEvents, setExpressions, setVlmAnalyses]);

  return { wsConnected, stats, loiteringPopup, setLoiteringPopup, vehicleAlertPopup, setVehicleAlertPopup };
}
