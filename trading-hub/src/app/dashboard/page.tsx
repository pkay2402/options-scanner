'use client';

import { useState, useEffect } from 'react';
import { 
  MarketPulse, 
  WatchlistTable, 
  ScannerFeed, 
  WhaleFlowsPanel,
  PriceChart,
  ZeroDTEPanel,
  RelativeRotationPanel,
} from '@/components';
import { Target, Settings, RefreshCw, LayoutGrid, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

function FooterTime() {
  const [time, setTime] = useState<string>('');
  
  useEffect(() => {
    const update = () => setTime(new Date().toLocaleTimeString());
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);
  
  return <span>{time || '--:--:--'}</span>;
}

type ViewMode = 'default' | '0dte' | 'rotation';

export default function TradingDashboard() {
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [viewMode, setViewMode] = useState<ViewMode>('default');

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Market Pulse Header */}
      <MarketPulse />

      {/* Main Header */}
      <header className="px-6 py-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Target className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-xl font-bold">Trading Hub</h1>
              <p className="text-slate-400 text-sm">One-stop options intelligence</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {/* View Mode Toggle */}
            <div className="flex gap-1 bg-slate-700 rounded p-0.5">
              <button
                onClick={() => setViewMode('default')}
                className={cn(
                  "px-3 py-1.5 text-xs rounded flex items-center gap-1 transition-colors",
                  viewMode === 'default' 
                    ? "bg-blue-500 text-white" 
                    : "text-slate-400 hover:text-white"
                )}
              >
                <LayoutGrid className="w-3 h-3" />
                Main
              </button>
              <button
                onClick={() => setViewMode('0dte')}
                className={cn(
                  "px-3 py-1.5 text-xs rounded flex items-center gap-1 transition-colors",
                  viewMode === '0dte' 
                    ? "bg-yellow-500 text-black" 
                    : "text-slate-400 hover:text-white"
                )}
              >
                <Zap className="w-3 h-3" />
                0DTE
              </button>
              <button
                onClick={() => setViewMode('rotation')}
                className={cn(
                  "px-3 py-1.5 text-xs rounded flex items-center gap-1 transition-colors",
                  viewMode === 'rotation' 
                    ? "bg-purple-500 text-white" 
                    : "text-slate-400 hover:text-white"
                )}
              >
                🔄 RRG
              </button>
            </div>
            
            <button className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors">
              <RefreshCw className="w-4 h-4" />
              <span className="text-sm">Refresh All</span>
            </button>
            <button className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
              <Settings className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        {viewMode === 'default' && (
          <DefaultView 
            selectedSymbol={selectedSymbol} 
            setSelectedSymbol={setSelectedSymbol} 
          />
        )}
        
        {viewMode === '0dte' && (
          <ZeroDTEView 
            selectedSymbol={selectedSymbol} 
            setSelectedSymbol={setSelectedSymbol} 
          />
        )}
        
        {viewMode === 'rotation' && (
          <RotationView 
            selectedSymbol={selectedSymbol} 
            setSelectedSymbol={setSelectedSymbol} 
          />
        )}
      </main>

      {/* Footer */}
      <footer className="px-6 py-4 border-t border-slate-700 text-center text-slate-500 text-sm">
        <p>
          Data from Schwab API & Custom Scanners • 
          Last updated: <FooterTime />
        </p>
      </footer>
    </div>
  );
}

function DefaultView({ selectedSymbol, setSelectedSymbol }: { 
  selectedSymbol: string; 
  setSelectedSymbol: (s: string) => void;
}) {
  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Left Column: Chart + Watchlist */}
      <div className="col-span-12 lg:col-span-8 space-y-6">
        <PriceChart symbol={selectedSymbol} />
        <WatchlistTable 
          selectedSymbol={selectedSymbol}
          onSelectSymbol={setSelectedSymbol}
        />
      </div>

      {/* Right Column: Scanner + Whale Flows */}
      <div className="col-span-12 lg:col-span-4 space-y-6">
        <div className="h-[400px]">
          <ScannerFeed onSelectSymbol={setSelectedSymbol} />
        </div>
        <WhaleFlowsPanel onSelectSymbol={setSelectedSymbol} />
      </div>
    </div>
  );
}

function ZeroDTEView({ selectedSymbol, setSelectedSymbol }: { 
  selectedSymbol: string; 
  setSelectedSymbol: (s: string) => void;
}) {
  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Left: 0DTE Panel + Chart */}
      <div className="col-span-12 lg:col-span-8 space-y-6">
        <ZeroDTEPanel />
        <PriceChart symbol={selectedSymbol} />
      </div>

      {/* Right: Scanner + Whale */}
      <div className="col-span-12 lg:col-span-4 space-y-6">
        <div className="h-[400px]">
          <ScannerFeed onSelectSymbol={setSelectedSymbol} />
        </div>
        <WhaleFlowsPanel onSelectSymbol={setSelectedSymbol} />
      </div>
    </div>
  );
}

function RotationView({ selectedSymbol, setSelectedSymbol }: { 
  selectedSymbol: string; 
  setSelectedSymbol: (s: string) => void;
}) {
  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Left: RRG Panel */}
      <div className="col-span-12 lg:col-span-8 space-y-6">
        <RelativeRotationPanel />
        <WatchlistTable 
          selectedSymbol={selectedSymbol}
          onSelectSymbol={setSelectedSymbol}
        />
      </div>

      {/* Right: Scanner + Whale */}
      <div className="col-span-12 lg:col-span-4 space-y-6">
        <div className="h-[400px]">
          <ScannerFeed onSelectSymbol={setSelectedSymbol} />
        </div>
        <WhaleFlowsPanel onSelectSymbol={setSelectedSymbol} />
      </div>
    </div>
  );
}
