import React, { useState } from 'react';
import { useApiData } from './hooks/useApiData';
import { useDataStream } from './hooks/useDataStream';
import Sidebar from './components/layout/Sidebar';
import Topbar from './components/layout/Topbar';
import LandingPage from './views/LandingPage';
import MainDashboard from './views/MainDashboard';
import HumanMonitoring from './views/HumanMonitoring';
import VehicleMonitoring from './views/VehicleMonitoring';
import AlertCenter from './views/AlertCenter';
import AnprLog from './views/AnprLog';
import LiveConsole from './views/LiveConsole';
import SystemStatus from './views/SystemStatus';
import './index.css';

const API_URL = "http://localhost:8000";

function App() {
  const [activeTab, setActiveTab] = useState('landing');
  const [soundEnabled, setSoundEnabled] = useState(true);

  const { sysStatus, esp32Sensors, events, setEvents, vlmAnalyses, setVlmAnalyses, expressions, setExpressions } = useApiData();
  const { wsConnected, stats, loiteringPopup, setLoiteringPopup, vehicleAlertPopup, setVehicleAlertPopup } = useDataStream(setEvents, setVlmAnalyses, setExpressions, soundEnabled);

  const renderView = () => {
    switch (activeTab) {
      case 'landing': return <LandingPage setActiveTab={setActiveTab} sysStatus={sysStatus} wsConnected={wsConnected} />
      case 'dashboard': return <MainDashboard stats={stats} events={events} sysStatus={sysStatus} expressions={expressions} wsConnected={wsConnected} />;
      case 'live': return <LiveConsole sysStatus={sysStatus} stats={stats} events={events} expressions={expressions} />;
      case 'human': return <HumanMonitoring stats={stats} expressions={expressions} />;
      case 'vehicle': return <VehicleMonitoring stats={stats} events={events} />;
      case 'alerts': return <AlertCenter events={events} vlmAnalyses={vlmAnalyses} />;
      case 'anpr': return <AnprLog events={events} stats={stats} />;
      case 'status': return <SystemStatus sysStatus={sysStatus} wsConnected={wsConnected} esp32Sensors={esp32Sensors} />;
      default: return <LandingPage setActiveTab={setActiveTab} sysStatus={sysStatus} wsConnected={wsConnected} />
    }
  };

  const getPageName = () => {
    switch (activeTab) {
      case 'landing': return 'INITIALIZATION';
      case 'dashboard': return 'COMMAND DASHBOARD';
      case 'live': return 'LIVE CONSOLE';
      case 'human': return 'L5 HUMAN MONITOR';
      case 'vehicle': return 'L2 VEHICLE MONITOR';
      case 'alerts': return 'ALERT CENTER';
      case 'anpr': return 'ANPR SUBSYSTEM';
      case 'status': return 'SYSTEM STATUS';
      default: return 'COMMAND DASHBOARD';
    }
  };

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="main-content">
        <Topbar soundEnabled={soundEnabled} setSoundEnabled={setSoundEnabled} wsConnected={wsConnected} sysStatus={sysStatus} activePageName={getPageName()} />
        
        {renderView()}

        {/* COMPACT ALERTS POPUPS */}
        {(loiteringPopup || vehicleAlertPopup) && (
          <div style={{ position: 'fixed', bottom: '24px', right: '24px', zIndex: 1000, display: 'flex', flexDirection: 'column', gap: '8px', pointerEvents: 'none' }}>
            
            {loiteringPopup && (
              <div className="panel" style={{ padding: '16px', width: '350px', borderLeft: '4px solid var(--color-red)', boxShadow: '0 10px 25px rgba(0,0,0,0.5)', pointerEvents: 'auto' }}>
                <div style={{ color: 'var(--color-red)', fontWeight: 'bold', marginBottom: '8px', fontSize: '0.875rem', letterSpacing: '0.05em' }}>
                  ⚠ STATIONARY PERSON ALERT
                </div>
                <div className="text-caption text-mono" style={{ marginBottom: '12px' }}>TARGET: #{loiteringPopup.track_id} | DUR: {loiteringPopup.duration || 'N/A'}s</div>
                <div className="flex-between">
                  <button onClick={() => setLoiteringPopup(null)} style={{ background: 'transparent', border: '1px solid var(--border-subtle)', color: 'var(--text-muted)', padding: '4px 12px', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontSize: '0.75rem' }}>DISMISS</button>
                  <button onClick={() => { setActiveTab('alerts'); setLoiteringPopup(null); }} style={{ background: 'var(--color-red)', border: 'none', color: '#fff', padding: '4px 12px', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontSize: '0.75rem', fontWeight: 'bold' }}>VIEW IN LOG</button>
                </div>
              </div>
            )}

            {vehicleAlertPopup && (
              <div className="panel" style={{ padding: '16px', width: '350px', borderLeft: '4px solid var(--color-red)', boxShadow: '0 10px 25px rgba(0,0,0,0.5)', pointerEvents: 'auto' }}>
                <div style={{ color: 'var(--color-red)', fontWeight: 'bold', marginBottom: '8px', fontSize: '0.875rem', letterSpacing: '0.05em' }}>
                  🚨 CIVILIAN VEHICLE ALERT
                </div>
                <div className="text-caption" style={{ marginBottom: '12px' }}>Unauthorized civilian vehicle #{vehicleAlertPopup.track_id} detected.</div>
                <div className="flex-between">
                  <button onClick={() => setVehicleAlertPopup(null)} style={{ background: 'transparent', border: '1px solid var(--border-subtle)', color: 'var(--text-muted)', padding: '4px 12px', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontSize: '0.75rem' }}>DISMISS</button>
                  <button onClick={() => { setActiveTab('alerts'); setVehicleAlertPopup(null); }} style={{ background: 'var(--color-red)', border: 'none', color: '#fff', padding: '4px 12px', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontSize: '0.75rem', fontWeight: 'bold' }}>VIEW IN LOG</button>
                </div>
              </div>
            )}
            
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
