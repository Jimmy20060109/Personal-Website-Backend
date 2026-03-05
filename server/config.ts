import dotenv from 'dotenv'
import type { LlmProvider } from './types'

dotenv.config()

const DEFAULT_PORT = 8787

export interface AppConfig {
  port: number
  corsOrigins: string[]
  llmProvider: LlmProvider
  embedProvider: LlmProvider
  llmModel: string
  embedModel: string
  openaiApiKey?: string
  deepseekApiKey?: string
}

function parsePort(raw: string | undefined): number {
  if (!raw) {
    return DEFAULT_PORT
  }
  const port = Number(raw)
  if (Number.isNaN(port) || port <= 0) {
    throw new Error('Invalid PORT in environment variables')
  }
  return port
}

function parseProvider(raw: string | undefined): LlmProvider {
  if (raw === 'deepseek') {
    return 'deepseek'
  }
  return 'openai'
}

function parseCorsOrigins(raw: string | undefined): string[] {
  const fallback = ['http://localhost:5173']
  if (!raw || raw.trim().length === 0) {
    return fallback
  }

  const origins = raw
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0)

  return origins.length > 0 ? origins : fallback
}

export function loadConfig(): AppConfig {
  const llmProvider = parseProvider(process.env.LLM_PROVIDER)
  const embedProvider = parseProvider(process.env.EMBED_PROVIDER)

  const config: AppConfig = {
    port: parsePort(process.env.PORT),
    corsOrigins: parseCorsOrigins(process.env.CORS_ORIGIN),
    llmProvider,
    embedProvider,
    llmModel: process.env.LLM_MODEL ?? 'gpt-4.1-mini',
    embedModel: process.env.EMBED_MODEL ?? 'text-embedding-3-small',
    openaiApiKey: process.env.OPENAI_API_KEY,
    deepseekApiKey: process.env.DEEPSEEK_API_KEY
  }

  if ((config.llmProvider === 'openai' || config.embedProvider === 'openai') && !config.openaiApiKey) {
    throw new Error('Missing OPENAI_API_KEY when LLM_PROVIDER/EMBED_PROVIDER uses openai')
  }

  if ((config.llmProvider === 'deepseek' || config.embedProvider === 'deepseek') && !config.deepseekApiKey) {
    throw new Error('Missing DEEPSEEK_API_KEY when LLM_PROVIDER/EMBED_PROVIDER uses deepseek')
  }

  return config
}
