export async function helloFunction(): Promise<{ ok: boolean; service: string; time: string }> {
  const res = await fetch('/.netlify/functions/hello');
  if (!res.ok) throw new Error('Hello function failed');
  return res.json();
}

export function connectSSE(onMessage: (data: any) => void): () => void {
  const source = new EventSource('/.netlify/functions/sse');
  const handler = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data);
      onMessage(data);
    } catch (e) {
      // ignore parse errors
    }
  };
  source.addEventListener('message', handler as any);
  source.onerror = () => {
    // Close on error to avoid retries in tests
    source.close();
  };
  return () => source.close();
}
