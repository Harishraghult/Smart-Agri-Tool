// Thin fetch wrappers: throw on non-2xx so modules show a real error instead of stale/fake data.

async function parse(response) {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      // non-JSON error body
    }
    throw new Error(`Backend error: ${detail}`);
  }
  return response.json();
}

export async function postForm(url, formData) {
  return parse(await fetch(url, { method: 'POST', body: formData }));
}

export async function postJson(url, body) {
  return parse(await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }));
}

// Real test-set photos served from /public/samples
export async function loadSampleFile(name) {
  const response = await fetch(`/samples/${name}`);
  if (!response.ok) throw new Error(`Sample image ${name} not found`);
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || 'image/jpeg' });
}

export function errorMessage(err) {
  return err instanceof TypeError
    ? 'Cannot reach the backend. Is the FastAPI server running on port 8000?'
    : err.message;
}
