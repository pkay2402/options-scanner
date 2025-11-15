// Node runtime-friendly SSE: not truly streaming in dev, but emits SSE-formatted lines
exports.handler = async () => {
  const events = [];
  events.push(`data: ${JSON.stringify({ ok: true, event: 'start', time: new Date().toISOString() })}\n`);
  for (let i = 1; i <= 5; i++) {
    events.push(`data: ${JSON.stringify({ ok: true, event: 'tick', count: i, time: new Date().toISOString() })}\n`);
  }
  const body = events.join('\n') + '\n\n';
  return {
    statusCode: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*'
    },
    body
  };
};
