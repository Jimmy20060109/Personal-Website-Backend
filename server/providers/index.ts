import type { AppConfig } from '../config'

export interface ChatGenerationInput {
  question: string
  contextChunks: string[]
  lang: 'zh' | 'en' | 'auto'
}

export interface ChatGenerationOutput {
  answer: string
}

interface EmbeddingResponse {
  embedding: number[]
}

function getProviderBaseUrl(provider: 'openai' | 'deepseek'): string {
  return 'https://api.openai.com/v1'
}

function getProviderApiKey(config: AppConfig, provider: 'openai' | 'deepseek'): string {
  const key = config.openaiApiKey
  if (!key) {
    throw new Error(`Missing API key for provider: ${provider}`)
  }
  return key
}

async function postJson<T>(url: string, apiKey: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify(body)
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Provider API error ${response.status}: ${errorText}`)
  }

  return (await response.json()) as T
}

export async function generateAnswer(
  config: AppConfig,
  input: ChatGenerationInput
): Promise<ChatGenerationOutput> {
  const apiKey = getProviderApiKey(config, config.llmProvider)
  const baseUrl = getProviderBaseUrl(config.llmProvider)

  const languageInstruction =
    input.lang === 'zh'
      ? 'Answer in Chinese unless the user explicitly asks for another language.'
      : input.lang === 'en'
        ? 'Answer in English unless the user explicitly asks for another language.'
        : 'Detect the language of the user question and answer in the same language. If the question mixes languages, answer in the dominant language. Only fall back to another language if the user explicitly requests it.'

  const contextBlock = input.contextChunks.length > 0 ? input.contextChunks.join('\n\n---\n\n') : 'No context found.'

  const payload = {
    model: config.llmModel,
    temperature: 0.2,
    messages: [
      {
        role: 'system',
        content:
          'You are a personal portfolio assistant. Answer strictly based on retrieved context. If context is insufficient, clearly say you do not have enough information. Follow the language instruction exactly.'
      },
      {
        role: 'user',
        content: `${languageInstruction}\n\nQuestion:\n${input.question}\n\nRetrieved context:\n${contextBlock}`
      }
    ]
  }

  const data = await postJson<{
    choices?: Array<{
      message?: {
        content?: string
      }
    }>
  }>(`${baseUrl}/chat/completions`, apiKey, payload)

  const answer = data.choices?.[0]?.message?.content?.trim()
  if (!answer) {
    throw new Error('Model response is empty')
  }

  return {
    answer
  }
}

export async function generateEmbedding(config: AppConfig, text: string): Promise<EmbeddingResponse> {
  const provider = config.embedProvider
  const apiKey = getProviderApiKey(config, provider)
  const baseUrl = getProviderBaseUrl(provider)

  const data = await postJson<{
    data?: Array<{
      embedding?: number[]
    }>
  }>(`${baseUrl}/embeddings`, apiKey, {
    model: config.embedModel,
    input: text
  })

  const embedding = data.data?.[0]?.embedding
  if (!embedding || embedding.length === 0) {
    throw new Error('Embedding response is empty')
  }

  return { embedding }
}