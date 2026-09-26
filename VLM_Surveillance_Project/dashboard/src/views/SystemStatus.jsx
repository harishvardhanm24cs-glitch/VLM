import React from 'react';

export default function SystemStatus({ sysStatus, wsConnected, esp32Sensors }) {
  
  // Map backend status to pipeline nodes
  const getStatus = (nodeName) => {
    // If backend doesn't exist at all, everything dependent is offline/unknown
    if (!sysStatus) return 'UNKNOWN';

    switch (nodeName) {
      case 'CAMERA':
      case 'VIDEO INPUT':
        return sysStatus['Video'] === 'ONLINE' ? 'ONLINE' : 'OFFLINE';
      
      case 'YOLO':
      case 'BYTE TRACK':
      case 'L1 VEHICLE CLASSIFIER':
      case 'L2 MILITARY CLASSIFIER':
      case 'L3 BEHAVIOR':
      case 'L5 EXPRESSION':
      case 'ANPR':
        // These are typically sub-modules of the CCTV Edge pipeline
        return sysStatus['CCTV Edge'] === 'ONLINE' ? 'ONLINE' : (sysStatus['CCTV Edge'] ? 'OFFLINE' : 'UNKNOWN');
        
      case 'VLM':
        return sysStatus['VLM'] === 'ONLINE' ? 'ONLINE' : (sysStatus['VLM'] ? 'OFFLINE' : 'UNKNOWN');

      case 'FASTAPI':
        // API health might be returned as 'FastAPI' or 'backend'
        return (sysStatus['FastAPI'] === 'ONLINE' || sysStatus['backend'] === 'ONLINE') ? 'ONLINE' : 'OFFLINE';
        
      case 'WEBSOCKET':
        return wsConnected ? 'ONLINE' : 'OFFLINE';
        
      case 'DASHBOARD':
        return 'ONLINE'; // We are literally rendering it

      default:
        return 'UNKNOWN';
    }
  };

  const getStatusColor = (status) => {
    if (status === 'ONLINE') return 'var(--color-green)';
    if (status === 'OFFLINE') return 'var(--color-red)';
    if (status === 'DEGRADED') return 'var(--color-amber)';
    return 'var(--text-muted)'; // UNKNOWN
  };

  const pipelineNodes = [
    'CAMERA',
    'VIDEO INPUT',
    'YOLO',
    'BYTE TRACK',
    'L1 VEHICLE CLASSIFIER',
    'L2 MILITARY CLASSIFIER',
    'L3 BEHAVIOR',
    'L5 EXPRESSION',
    'ANPR',
    'VLM',
    'FASTAPI',
    'WEBSOCKET',
    'DASHBOARD'
  ];

  return (
    <div className="view-container animate-fade" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      
      <div className="flex-between panel" style={{ padding: '8px 16px', flexDirection: 'row', marginBottom: 'var(--space-6)' }}>
        <h1 style={{ fontSize: '1.1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="text-blue">●</span> SYSTEM INFRASTRUCTURE STATUS
        </h1>
        <div className="text-caption text-mono" style={{ display: 'flex', gap: '16px' }}>
          <span>VERSION: 2.4.1-PROD</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: 'var(--space-6)', flex: 1, overflow: 'hidden' }}>
        
        {/* ==================== LEFT: PIPELINE HEALTH ==================== */}
        <div className="panel" style={{ padding: 'var(--space-6)', display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
          <h2 style={{ fontSize: '0.875rem', marginBottom: 'var(--space-6)', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>INTEGRATION PIPELINE</h2>
          
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px', paddingBottom: '32px' }}>
            {pipelineNodes.map((node, idx) => {
              const status = getStatus(node);
              const isLast = idx === pipelineNodes.length - 1;

              return (
                <React.Fragment key={node}>
                  <div style={{ 
                    width: '300px', 
                    border: '1px solid var(--border-subtle)', 
                    background: 'var(--bg-app)', 
                    padding: '12px 16px', 
                    borderRadius: '4px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}>
                    <span className="text-mono" style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>{node}</span>
                    <span className="text-mono" style={{ fontSize: '0.75rem', color: getStatusColor(status), fontWeight: 'bold' }}>{status}</span>
                  </div>
                  
                  {!isLast && (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{ width: '2px', height: '16px', background: 'var(--border-subtle)' }} />
                      <div style={{ width: '0', height: '0', borderLeft: '4px solid transparent', borderRight: '4px solid transparent', borderTop: '6px solid var(--border-subtle)' }} />
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* ==================== RIGHT: CONNECTIONS & MODELS ==================== */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
          
          <div className="panel" style={{ padding: 'var(--space-4)' }}>
            <h2 style={{ fontSize: '0.875rem', marginBottom: 'var(--space-4)', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>CORE CONNECTIONS</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>Backend API:</span>
                <span className="text-mono" style={{ fontWeight: 'bold', fontSize: '0.85rem', color: getStatusColor(getStatus('FASTAPI')) }}>
                  {getStatus('FASTAPI')}
                </span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>WebSocket:</span>
                <span className="text-mono" style={{ fontWeight: 'bold', fontSize: '0.85rem', color: getStatusColor(getStatus('WEBSOCKET')) }}>
                  {getStatus('WEBSOCKET')}
                </span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>Camera Stream:</span>
                <span className="text-mono" style={{ fontWeight: 'bold', fontSize: '0.85rem', color: getStatusColor(getStatus('CAMERA')) }}>
                  {getStatus('CAMERA')}
                </span>
              </div>
            </div>
          </div>

          <div className="panel" style={{ padding: 'var(--space-4)' }}>
            <h2 style={{ fontSize: '0.875rem', marginBottom: 'var(--space-4)', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>IOT / ESP32 HARDWARE</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>CONNECTION:</span>
                <span className="text-mono" style={{ fontWeight: 'bold', fontSize: '0.85rem', color: getStatusColor(esp32Sensors?.status !== 'OFFLINE' ? 'ONLINE' : 'OFFLINE') }}>
                  {esp32Sensors?.status !== 'OFFLINE' ? 'CONNECTED' : 'OFFLINE'}
                </span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>PIR MOTION:</span>
                <span className="text-mono" style={{ fontSize: '0.85rem' }}>{esp32Sensors?.pir ? 'MOTION' : 'CLEAR'}</span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>IR OBJECT:</span>
                <span className="text-mono" style={{ fontSize: '0.85rem' }}>{esp32Sensors?.ir ? 'OBJECT' : 'CLEAR'}</span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>HC-SR04:</span>
                <span className="text-mono" style={{ fontSize: '0.85rem' }}>{esp32Sensors?.hc_sr04_cm?.toFixed(1) || '0.0'} cm</span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>UL53LDK:</span>
                <span className="text-mono" style={{ fontSize: '0.85rem' }}>{esp32Sensors?.ul53ldk_cm?.toFixed(1) || '0.0'} cm</span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>MPU6050:</span>
                <span className="text-mono" style={{ fontSize: '0.85rem' }}>{esp32Sensors?.mpu6050 ? 'ACTIVE' : 'INACTIVE'}</span>
              </div>
              <div className="flex-between" style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid var(--border-subtle)' }}>
                <span className="text-mono text-muted" style={{ fontSize: '0.85rem' }}>ALARM (BUZZER):</span>
                <span className="text-mono" style={{ fontWeight: 'bold', fontSize: '0.85rem', color: esp32Sensors?.alarm ? 'var(--color-red)' : 'var(--text-muted)' }}>
                  {esp32Sensors?.alarm ? 'ON' : 'OFF'}
                </span>
              </div>
            </div>
          </div>

          <div className="panel" style={{ padding: 'var(--space-4)' }}>
            <h2 style={{ fontSize: '0.875rem', marginBottom: 'var(--space-4)', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>MODEL INFORMATION</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                <span className="text-mono text-muted" style={{ fontSize: '0.75rem' }}>L1 MODEL</span>
                <span className="text-mono" style={{ fontSize: '0.75rem' }}>UNKNOWN</span>
              </div>
              <div className="flex-between" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
                <span className="text-mono text-muted" style={{ fontSize: '0.75rem' }}>L2 MODEL</span>
                <span className="text-mono" style={{ fontSize: '0.75rem' }}>UNKNOWN</span>
              </div>
              <div className="flex-between">
                <span className="text-mono text-muted" style={{ fontSize: '0.75rem' }}>L5 MODEL</span>
                <span className="text-mono" style={{ fontSize: '0.75rem' }}>UNKNOWN</span>
              </div>
            </div>
            <div className="text-caption text-muted" style={{ marginTop: '12px', fontSize: '0.65rem' }}>
              Models are active but exact versioning is not exposed by the current API configuration.
            </div>
          </div>

          <div className="panel" style={{ padding: 'var(--space-4)', flex: 1 }}>
            <h2 style={{ fontSize: '0.875rem', marginBottom: 'var(--space-4)', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>RECENT SYSTEM EVENTS</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto' }}>
               {/* As requested, only show actual system events if they exist. Since we don't have a dedicated system event feed from the backend, we show nothing rather than fabricating events. */}
               <div className="text-caption" style={{ textAlign: 'center', marginTop: '32px' }}>NO RECENT SYSTEM EVENTS</div>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
