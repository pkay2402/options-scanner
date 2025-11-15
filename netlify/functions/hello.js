exports.handler = async () => {
  return {
    statusCode: 200,
    headers: {
      'content-type': 'application/json'
    },
    body: JSON.stringify({ ok: true, service: 'netlify-fn', time: new Date().toISOString() })
  };
};