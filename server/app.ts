import cors from 'cors'
import express from 'express'
import { loadConfig, type AppConfig } from './config'
import { askWithRag, reindexKnowledge } from './rag/index'
import type { AskRequest } from './types'

const app = express()
const RAG_BUILD = 'rag-intent-hard-filter-v4-cors'
const corsOrigins = (process.env.CORS_ORIGIN ?? 'http://localhost:5173')
  .split(',')
  .map((item) => item.trim())
  .filter((item) => item.length > 0)

function getRuntimeConfig(): AppConfig {
  return loadConfig()
}

app.use(
  cors({
    origin: (origin, callback) => {
      // Non-browser clients (curl/postman) often have no origin header.
      if (!origin) {
        callback(null, true)
        return
      }

      // In local development, allow localhost/127.0.0.1 on any port.
      if (/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(origin)) {
        callback(null, true)
        return
      }

      if (corsOrigins.includes(origin)) {
        callback(null, true)
        return
      }

      callback(new Error(`CORS blocked for origin: ${origin}`))
    }
  })
)
app.use(express.json())

app.get('/api/health', (_req, res) => {
  try {
    const config = getRuntimeConfig()
    res.json({ ok: true, service: 'rag-server', provider: config.llmProvider, build: RAG_BUILD, configValid: true })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown config error'
    res.status(200).json({ ok: true, service: 'rag-server', build: RAG_BUILD, configValid: false, configError: message })
  }
})

app.post('/api/ask', async (req, res) => {
  const body = req.body as AskRequest
  if (!body?.question || typeof body.question !== 'string') {
    return res.status(400).json({ error: 'question is required and must be a string' })
  }

  try {
    const config = getRuntimeConfig()
    const data = await askWithRag(config, {
      question: body.question,
      topK: body.topK,
      lang: body.lang ?? 'auto',
      topic: body.topic
    })
    return res.json(data)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error'
    return res.status(500).json({ error: message })
  }
})

app.post('/api/reindex', async (_req, res) => {
  try {
    const config = getRuntimeConfig()
    const data = await reindexKnowledge(config)
    return res.json(data)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error'
    return res.status(500).json({ error: message })
  }
})

export default app
