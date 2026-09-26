import { useState, useEffect } from 'react';

const API_URL = "http://localhost:8000";

export function useApiData() {
  const [sysStatus, setSysStatus] = useState({ backend: "OFFLINE", pipeline: false });
  const [esp32Sensors, setEsp32Sensors] = useState(null);
  const [events, setEvents] = useState([]);
  const [vlmAnalyses, setVlmAnalyses] = useState([]);
  const [expressions, setExpressions] = useState({});

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
      
      try {
        const espRes = await fetch(`${API_URL}/api/iot/esp32/sensors`);
        const espData = await espRes.json();
        setEsp32Sensors(espData);
      } catch (e) {
        setEsp32Sensors({ status: "OFFLINE" });
      }
    };
    checkHealth();
    const intv = setInterval(checkHealth, 5000);
    return () => clearInterval(intv);
  }, []);

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

    fetch(`${API_URL}/api/expressions`).then(r => r.json()).then(data => {
      const expMap = {};
      data.forEach(d => expMap[d.track_id] = d);
      setExpressions(expMap);
    }).catch(() => {});
  }, []);

  return { sysStatus, esp32Sensors, events, setEvents, vlmAnalyses, setVlmAnalyses, expressions, setExpressions };
}
