export interface Source {
  filename: string
  page: number
}

export interface AskResponse {
  answer: string
  sources: Source[]
  latency_ms: number
  iterations: number
  tool_calls_made: string[]
}

export interface StreamState {
  answer: string
  sources: Source[]
  done: boolean
}
