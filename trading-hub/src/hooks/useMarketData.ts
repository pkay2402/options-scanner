import { useQuery } from '@tanstack/react-query';
import {
  fetchWatchlist,
  fetchWhaleFlows,
  fetchScannerSignals,
  fetchMarketPulse,
  fetchPriceHistory,
  fetchOptionsChain,
  fetchVolumeWalls,
  fetchGEXData,
} from '@/lib/api';

/**
 * Schwab API Rate Limits: 120 calls/minute
 * 
 * Budget allocation:
 * - Market Pulse: 1 call/min (3 symbols batched) = 1 call
 * - Price History: 1 call/3min = 0.33 calls/min
 * - Volume Walls: 1 call/3min = 0.33 calls/min  
 * - GEX: 1 call/3min = 0.33 calls/min
 * - Whale Flows: 10 calls/5min = 2 calls/min
 * 
 * Total: ~4 calls/min (safe margin for manual refreshes)
 * 
 * Droplet API (no rate limit):
 * - Watchlist, Scanner signals refresh more frequently
 */

// ==================== DROPLET API (no rate limit) ====================

export function useWatchlist() {
  return useQuery({
    queryKey: ['watchlist'],
    queryFn: fetchWatchlist,
    staleTime: 30 * 1000,
    refetchInterval: 60 * 1000, // Every 60s - droplet API
  });
}

export function useScannerSignals() {
  return useQuery({
    queryKey: ['scannerSignals'],
    queryFn: fetchScannerSignals,
    staleTime: 30 * 1000,
    refetchInterval: 60 * 1000, // Every 60s - droplet API
  });
}

// ==================== SCHWAB API (rate limited) ====================

export function useMarketPulse() {
  return useQuery({
    queryKey: ['marketPulse'],
    queryFn: fetchMarketPulse,
    staleTime: 45 * 1000,
    refetchInterval: 60 * 1000, // Every 60s - 1 call (batched)
  });
}

export function usePriceHistory(symbol: string, period: '1D' | '5D' | '1M' | '3M' = '1D') {
  return useQuery({
    queryKey: ['priceHistory', symbol, period],
    queryFn: () => fetchPriceHistory(symbol, period),
    staleTime: 2 * 60 * 1000,
    refetchInterval: 3 * 60 * 1000, // Every 3min - 1 call
    enabled: !!symbol,
  });
}

export function useOptionsChain(symbol: string, expiry?: string) {
  return useQuery({
    queryKey: ['optionsChain', symbol, expiry],
    queryFn: () => fetchOptionsChain(symbol, expiry),
    staleTime: 3 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000, // Every 5min - 1 call
    enabled: !!symbol,
  });
}

export function useVolumeWalls(symbol: string, expiry?: string) {
  return useQuery({
    queryKey: ['volumeWalls', symbol, expiry],
    queryFn: () => fetchVolumeWalls(symbol, expiry),
    staleTime: 2 * 60 * 1000,
    refetchInterval: 3 * 60 * 1000, // Every 3min - 1 call
    enabled: !!symbol,
  });
}

export function useGEXData(symbol: string) {
  return useQuery({
    queryKey: ['gexData', symbol],
    queryFn: () => fetchGEXData(symbol),
    staleTime: 2 * 60 * 1000,
    refetchInterval: 3 * 60 * 1000, // Every 3min - 1 call
    enabled: !!symbol,
  });
}

export function useWhaleFlows() {
  return useQuery({
    queryKey: ['whaleFlows'],
    queryFn: fetchWhaleFlows,
    staleTime: 4 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000, // Every 5min - ~10 calls
  });
}
