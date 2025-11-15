import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import './App.css';

// Components
import Dashboard from './components/Dashboard';
import VolumeWalls from './components/VolumeWalls';
import FlowScanner from './components/FlowScanner';
import OptionFinder from './components/OptionFinder';
import AlertsConfig from './components/AlertsConfig';

// Services
import { NotificationService } from './services/notifications';
import { API_BASE_URL } from './services/api';
import { helloFunction, connectSSE } from './services/functions';

function App() {
  const [authStatus, setAuthStatus] = useState<boolean | null>(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [fnOk, setFnOk] = useState<boolean | null>(null);
  const [sseLast, setSseLast] = useState<string | null>(null);

  useEffect(() => {
    // Check authentication status
    checkAuthStatus();
    // Check Netlify function health
    helloFunction().then(() => setFnOk(true)).catch(() => setFnOk(false));
    // Connect to simple SSE stream (non-blocking)
    const disconnect = connectSSE((d) => setSseLast(d?.event || 'tick'));
    const t = setTimeout(() => disconnect(), 12000); // auto-close after ~10s
    
    // Check notification permission
    if ('Notification' in window) {
      setNotificationsEnabled(Notification.permission === 'granted');
    }
    return () => { clearTimeout(t); disconnect(); };
  }, []);

  const checkAuthStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/status`);
      const data = await response.json();
      setAuthStatus(data.authenticated);
    } catch (error) {
      console.error('Auth check failed:', error);
      setAuthStatus(false);
    }
  };

  const handleEnableNotifications = async () => {
    const enabled = await NotificationService.requestPermission();
    setNotificationsEnabled(enabled);
    if (enabled) {
      await NotificationService.subscribe();
    }
  };

  if (authStatus === null) {
    return (
      <div className="app-loading">
        <div className="spinner"></div>
        <p>Connecting to Schwab API...</p>
      </div>
    );
  }

  if (authStatus === false) {
    const handleRefreshToken = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
          method: 'POST'
        });
        
        if (response.ok) {
          setAuthStatus(null); // Show loading
          setTimeout(() => checkAuthStatus(), 1000); // Recheck after 1 second
        } else {
          alert('Token refresh failed. You may need to re-authenticate using the backend setup script.');
        }
      } catch (error) {
        console.error('Refresh failed:', error);
        alert('Unable to refresh token. Please check backend logs.');
      }
    };

    return (
      <div className="app-error">
        <h2>⚠️ Authentication Failed</h2>
        <p>Unable to connect to Schwab API. Your token may have expired.</p>
        <div style={{ marginTop: '20px', display: 'flex', gap: '10px', justifyContent: 'center' }}>
          <button className="btn btn-secondary" onClick={checkAuthStatus}>
            Retry Connection
          </button>
          <button className="btn btn-primary" onClick={handleRefreshToken}>
            Refresh Token
          </button>
        </div>
        <div style={{ marginTop: '30px', padding: '15px', background: '#f8f9fa', borderRadius: '8px', maxWidth: '500px', margin: '30px auto' }}>
          <p style={{ fontSize: '14px', margin: 0 }}>
            <strong>Need to re-authenticate?</strong><br/>
            Run from terminal:<br/>
            <code style={{ display: 'block', marginTop: '8px', padding: '8px', background: '#fff', borderRadius: '4px' }}>
              cd /Users/piyushkhaitan/schwab/options<br/>
              python3 pwa-app/backend/refresh_auth.py
            </code>
          </p>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <div className="app">
        <Header 
          notificationsEnabled={notificationsEnabled}
          onEnableNotifications={handleEnableNotifications}
          fnOk={fnOk}
          sseLast={sseLast}
        />
        
        <main className="app-main">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/volume-walls" element={<VolumeWalls />} />
            <Route path="/flow-scanner" element={<FlowScanner />} />
            <Route path="/option-finder" element={<OptionFinder />} />
            <Route path="/alerts" element={<AlertsConfig />} />
          </Routes>
        </main>

        <BottomNav />
      </div>
    </Router>
  );
}

// Header component
function Header({ notificationsEnabled, onEnableNotifications, fnOk, sseLast }: any) {
  return (
    <header className="app-header">
      <div className="container">
        <div className="header-content">
          <h1 className="app-title">📊 Flow Pro</h1>
          
          {!notificationsEnabled && (
            <button 
              className="btn-notification"
              onClick={onEnableNotifications}
            >
              🔔 Enable Alerts
            </button>
          )}
          <div style={{ display: 'flex', gap: 8, marginLeft: 'auto' }}>
            <StatusPill label="API" ok={true} />
            <StatusPill label="Fn" ok={fnOk} />
            {sseLast && <span className="status-pill neutral">SSE:{sseLast}</span>}
          </div>
        </div>
      </div>
    </header>
  );
}

// Bottom navigation for mobile
function BottomNav() {
  const location = useLocation();
  
  const isActive = (path: string) => {
    return location.pathname === path ? 'active' : '';
  };

  return (
    <nav className="bottom-nav">
      <Link to="/" className={`nav-item ${isActive('/')}`}>
        <span className="nav-icon">🏠</span>
        <span className="nav-label">Home</span>
      </Link>
      
      <Link to="/volume-walls" className={`nav-item ${isActive('/volume-walls')}`}>
        <span className="nav-icon">🧱</span>
        <span className="nav-label">Walls</span>
      </Link>
      
      <Link to="/flow-scanner" className={`nav-item ${isActive('/flow-scanner')}`}>
        <span className="nav-icon">🌊</span>
        <span className="nav-label">Flow</span>
      </Link>
      
      <Link to="/option-finder" className={`nav-item ${isActive('/option-finder')}`}>
        <span className="nav-icon">🎯</span>
        <span className="nav-label">Finder</span>
      </Link>
      
      <Link to="/alerts" className={`nav-item ${isActive('/alerts')}`}>
        <span className="nav-icon">🔔</span>
        <span className="nav-label">Alerts</span>
      </Link>
    </nav>
  );
}

export default App;

function StatusPill({ label, ok }: { label: string; ok: boolean | null }) {
  const cls = ok == null ? 'neutral' : ok ? 'ok' : 'bad';
  return <span className={`status-pill ${cls}`}>{label}:{ok == null ? '…' : ok ? 'ok' : 'err'}</span>;
}
