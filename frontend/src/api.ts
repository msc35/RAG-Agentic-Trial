import type { AskResponse, Source } from './types'

/** Non-streaming: run the full agent loop and return the complete response. */
export async function ask(question: string): Promise<AskResponse> {
  const res = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail))
  }
  return res.json()
}

/**
 * Streaming: yields tokens as they arrive, then yields a final object with sources.
 *
 * The backend sends Server-Sent Events:
 *   data: <token>\n\n          — one or more tokens
 *   data: [SOURCES]<json>\n\n  — source list
 *   data: [DONE]\n\n           — end of stream
 */
export async function* askStream(
  question: string
): AsyncGenerator<{ token?: string; sources?: Source[] }> {
  const res = await fetch('/ask?stream=true', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail))
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // SSE events are delimited by double newlines.
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''

    for (const event of events) {
      if (!event.startsWith('data: ')) continue
      const data = event.slice(6) // do NOT trim — tokens have meaningful leading spaces

      if (data === '[DONE]') return

      if (data.startsWith('[SOURCES]')) {
        const sources: Source[] = JSON.parse(data.slice(9))
        yield { sources }
      } else {
        yield { token: data }
      }
    }
  }
}
