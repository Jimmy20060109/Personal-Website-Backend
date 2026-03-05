export type LlmProvider = 'openai' | 'deepseek'

export interface AskRequest {
  question: string
  topK?: number
  lang?: 'zh' | 'en' | 'auto'
  topic?: 'project' | 'work' | 'education' | 'skills' | 'contact' | 'location'
}

export interface AskSource {
  id: string
  title: string
  source?: string
  score: number
}

export interface AskResponse {
  answer: string
  sources: AskSource[]
  provider: LlmProvider
}

export interface ReindexResponse {
  ok: boolean
  message: string
  chunkCount?: number
  newCount?: number
  reusedCount?: number
  removedCount?: number
}
