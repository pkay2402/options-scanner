import React, { useState, useEffect } from 'react';
import { NotificationService } from '../services/notifications';
import './AlertsConfig.css';

interface AlertRule {
  id: string;
  enabled: boolean;
  type: 'volume' | 'price' | 'gamma' | 'iv' | 'premium';
  symbol: string;
  condition: 'above' | 'below' | 'equals';
  threshold: number;
  description: string;
}

const AlertsConfig: React.FC = () => {
  const [notificationEnabled, setNotificationEnabled] = useState(false);
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [newRule, setNewRule] = useState({
    type: 'premium' as const,
    symbol: 'SPY',
    condition: 'above' as const,
    threshold: 100000,
    description: ''
  });

  useEffect(() => {
    // Check notification permission status
    if ('Notification' in window) {
      setNotificationEnabled(Notification.permission === 'granted');
    }

    // Load saved rules from localStorage
    const saved = localStorage.getItem('alertRules');
    if (saved) {
      try {
        setRules(JSON.parse(saved));
      } catch (err) {
        console.error('Failed to load rules:', err);
      }
    } else {
      // Default rules
      setRules([
        {
          id: '1',
          enabled: true,
          type: 'premium',
          symbol: 'SPY',
          condition: 'above',
          threshold: 500000,
          description: 'SPY trades over $500K'
        },
        {
          id: '2',
          enabled: true,
          type: 'volume',
          symbol: 'AAPL',
          condition: 'above',
          threshold: 10000,
          description: 'AAPL volume spike'
        }
      ]);
    }
  }, []);

  useEffect(() => {
    // Save rules to localStorage whenever they change
    localStorage.setItem('alertRules', JSON.stringify(rules));
  }, [rules]);

  const handleEnableNotifications = async () => {
    try {
      await NotificationService.requestPermission();
      await NotificationService.subscribe();
      setNotificationEnabled(true);
    } catch (err) {
      console.error('Failed to enable notifications:', err);
      alert('Failed to enable notifications. Please check your browser settings.');
    }
  };

  const handleTestNotification = () => {
    NotificationService.sendAlert(
      '🔔 Test Notification',
      'Notifications are working correctly!'
    );
  };

  const addRule = () => {
    if (!newRule.symbol.trim()) {
      alert('Please enter a symbol');
      return;
    }

    const rule: AlertRule = {
      id: Date.now().toString(),
      enabled: true,
      ...newRule,
      description: newRule.description || generateDescription(newRule)
    };

    setRules([...rules, rule]);

    // Reset form
    setNewRule({
      type: 'premium',
      symbol: 'SPY',
      condition: 'above',
      threshold: 100000,
      description: ''
    });
  };

  const generateDescription = (rule: typeof newRule) => {
    const typeLabels = {
      volume: 'Volume',
      price: 'Price',
      gamma: 'Gamma',
      iv: 'IV',
      premium: 'Premium'
    };

    const conditionLabels = {
      above: '>',
      below: '<',
      equals: '='
    };

    return `${rule.symbol} ${typeLabels[rule.type]} ${conditionLabels[rule.condition]} ${formatValue(rule.type, rule.threshold)}`;
  };

  const formatValue = (type: string, value: number) => {
    if (type === 'premium') {
      return `$${(value / 1000).toFixed(0)}K`;
    }
    if (type === 'price') {
      return `$${value.toFixed(2)}`;
    }
    if (type === 'iv') {
      return `${value.toFixed(1)}%`;
    }
    return value.toLocaleString();
  };

  const toggleRule = (id: string) => {
    setRules(rules.map(rule => 
      rule.id === id ? { ...rule, enabled: !rule.enabled } : rule
    ));
  };

  const deleteRule = (id: string) => {
    if (window.confirm('Are you sure you want to delete this rule?')) {
      setRules(rules.filter(rule => rule.id !== id));
    }
  };

  return (
    <div className="alerts-config">
      <div className="page-header">
        <h1>⚙️ Alert Configuration</h1>
        <p>Set up custom notifications for trading events</p>
      </div>

      <div className="notification-section">
        <div className="notification-card">
          <div className="notification-info">
            <h3>Push Notifications</h3>
            <p>
              {notificationEnabled 
                ? '✅ Notifications are enabled' 
                : '❌ Notifications are disabled'}
            </p>
          </div>
          <div className="notification-actions">
            {!notificationEnabled ? (
              <button onClick={handleEnableNotifications} className="btn-primary">
                Enable Notifications
              </button>
            ) : (
              <button onClick={handleTestNotification} className="btn-secondary">
                Test Notification
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="rules-section">
        <h2>Alert Rules ({rules.filter(r => r.enabled).length} active)</h2>

        <div className="rules-list">
          {rules.length === 0 ? (
            <div className="empty-state">
              <p>No alert rules configured</p>
              <p className="text-secondary">Add your first rule below</p>
            </div>
          ) : (
            rules.map(rule => (
              <div key={rule.id} className={`rule-card ${rule.enabled ? 'enabled' : 'disabled'}`}>
                <div className="rule-header">
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={rule.enabled}
                      onChange={() => toggleRule(rule.id)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                  <div className="rule-description">{rule.description}</div>
                </div>
                <div className="rule-details">
                  <span className="badge">{rule.type}</span>
                  <span className="rule-symbol">{rule.symbol}</span>
                  <span className="rule-condition">
                    {rule.condition} {formatValue(rule.type, rule.threshold)}
                  </span>
                </div>
                <button onClick={() => deleteRule(rule.id)} className="delete-btn">
                  🗑️ Delete
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="add-rule-section">
        <h2>Add New Rule</h2>
        <div className="add-rule-form">
          <div className="form-row">
            <div className="form-group">
              <label>Alert Type</label>
              <select 
                value={newRule.type}
                onChange={(e) => setNewRule({ ...newRule, type: e.target.value as any })}
                className="form-input"
              >
                <option value="premium">Premium</option>
                <option value="volume">Volume</option>
                <option value="price">Price</option>
                <option value="gamma">Gamma</option>
                <option value="iv">Implied Volatility</option>
              </select>
            </div>

            <div className="form-group">
              <label>Symbol</label>
              <input
                type="text"
                value={newRule.symbol}
                onChange={(e) => setNewRule({ ...newRule, symbol: e.target.value.toUpperCase() })}
                placeholder="SPY"
                className="form-input"
              />
            </div>

            <div className="form-group">
              <label>Condition</label>
              <select
                value={newRule.condition}
                onChange={(e) => setNewRule({ ...newRule, condition: e.target.value as any })}
                className="form-input"
              >
                <option value="above">Above</option>
                <option value="below">Below</option>
                <option value="equals">Equals</option>
              </select>
            </div>

            <div className="form-group">
              <label>Threshold</label>
              <input
                type="number"
                value={newRule.threshold}
                onChange={(e) => setNewRule({ ...newRule, threshold: Number(e.target.value) })}
                className="form-input"
              />
            </div>
          </div>

          <div className="form-group">
            <label>Description (optional)</label>
            <input
              type="text"
              value={newRule.description}
              onChange={(e) => setNewRule({ ...newRule, description: e.target.value })}
              placeholder="Leave blank for auto-generated description"
              className="form-input"
            />
          </div>

          <button onClick={addRule} className="btn-primary btn-block">
            Add Rule
          </button>
        </div>
      </div>

      <div className="info-box">
        <h3>💡 Tips</h3>
        <ul>
          <li>Enable notifications to receive real-time alerts</li>
          <li>Premium alerts are great for catching big money trades</li>
          <li>Volume alerts help identify unusual activity</li>
          <li>Gamma alerts track dealer hedging pressure</li>
          <li>IV alerts spot volatility expansion/contraction</li>
          <li>Rules are saved automatically and persist between sessions</li>
        </ul>
      </div>
    </div>
  );
};

export default AlertsConfig;
