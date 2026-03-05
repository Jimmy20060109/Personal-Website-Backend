import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import type { AppConfig } from '../config'
import { generateAnswer, generateEmbedding } from '../providers/index'
import type { AskRequest, AskResponse, ReindexResponse } from '../types'

const VECTOR_INDEX_PATH = path.resolve(process.cwd(), 'data', 'vector-index.json')
const KNOWLEDGE_FILE_PATH = path.resolve(process.cwd(), 'MY_INFO.md')
const MAX_CHUNK_CHARS = 700
const CHUNK_OVERLAP = 120
const DEFAULT_TOP_K = 4
const MIN_RETRIEVAL_SCORE = 0.1
const MIN_CONFIDENCE_SCORE = 0.12
const SEMANTIC_WEIGHT = 0.78
const KEYWORD_WEIGHT = 0.22
const INTENT_MATCH_BOOST = 0.18
const INTENT_MISMATCH_PENALTY = 0.06

const STOPWORDS = new Set([
  'a',
  'an',
  'the',
  'and',
  'or',
  'to',
  'of',
  'in',
  'on',
  'for',
  'with',
  'is',
  'are',
  'was',
  'were',
  'be',
  'this',
  'that',
  'it',
  'i',
  'you',
  'he',
  'she',
  'we',
  'they',
  '请',
  '介绍',
  '一下',
  '关于',
  '什么',
  '如何',
  '哪些'
])

interface VectorChunk {
  id: string
  title: string
  content: string
  embedding: number[]
  source: string
  hash: string
  updatedAt: string
}

interface RawSection {
  title: string
  content: string
  source: string
}

type IntentCategory = 'project' | 'work' | 'education' | 'skills' | 'contact' | 'location'

function hasMeaningfulContent(text: string): boolean {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)

  if (lines.length === 0) {
    return false
  }

  // Bullet "Field: value" lines with empty values are template noise.
  const usefulLines = lines.filter((line) => {
    const bulletFieldMatch = line.match(/^-+\s*[^:]{1,80}:\s*(.*)$/)
    if (!bulletFieldMatch) {
      return true
    }

    const value = bulletFieldMatch[1].trim()
    return value.length > 0 && value !== '-' && value.toLowerCase() !== 'n/a'
  })

  if (usefulLines.length === 0) {
    return false
  }

  const semanticChars = usefulLines.join(' ').replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, '')
  return semanticChars.length >= 8
}

function readLocalVectorIndex(): VectorChunk[] {
  if (!fs.existsSync(VECTOR_INDEX_PATH)) {
    return []
  }

  try {
    const raw = fs.readFileSync(VECTOR_INDEX_PATH, 'utf-8')
    const parsed = JSON.parse(raw) as VectorChunk[]
    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed
      .map((chunk) => ({
        ...chunk,
        hash: chunk.hash || hashChunk(chunk.title, chunk.source, chunk.content)
      }))
      .filter((chunk) => Array.isArray(chunk.embedding) && chunk.embedding.length > 0)
  } catch {
    return []
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) {
    return 0
  }

  let dot = 0
  let normA = 0
  let normB = 0

  for (let index = 0; index < a.length; index += 1) {
    dot += a[index] * b[index]
    normA += a[index] * a[index]
    normB += b[index] * b[index]
  }

  if (normA === 0 || normB === 0) {
    return 0
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

function splitMarkdownSections(markdown: string): RawSection[] {
  const lines = markdown.split(/\r?\n/)
  const sections: RawSection[] = []

  let currentTitle = 'General'
  let buffer: string[] = []

  const flush = () => {
    const text = buffer.join('\n').trim()
    if (text.length > 0) {
      sections.push({
        title: currentTitle,
        content: text,
        source: `MY_INFO.md#${currentTitle}`
      })
    }
    buffer = []
  }

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/)
    if (headingMatch) {
      flush()
      currentTitle = headingMatch[2].trim()
      continue
    }
    buffer.push(line)
  }

  flush()
  return sections
}

function chunkSection(section: RawSection): RawSection[] {
  if (section.content.length <= MAX_CHUNK_CHARS) {
    return [section]
  }

  const chunks: RawSection[] = []
  let start = 0
  let chunkNo = 1

  while (start < section.content.length) {
    const end = Math.min(section.content.length, start + MAX_CHUNK_CHARS)
    const slice = section.content.slice(start, end).trim()
    if (slice.length > 0) {
      chunks.push({
        title: `${section.title} (Part ${chunkNo})`,
        content: slice,
        source: section.source
      })
      chunkNo += 1
    }

    if (end === section.content.length) {
      break
    }

    start = Math.max(0, end - CHUNK_OVERLAP)
  }

  return chunks
}

function buildChunksFromMarkdown(markdown: string): RawSection[] {
  const sections = splitMarkdownSections(markdown)
  return sections.flatMap((section) => chunkSection(section)).filter((section) => hasMeaningfulContent(section.content))
}

function hashChunk(title: string, source: string, content: string): string {
  return createHash('sha256').update(`${title}\n${source}\n${content}`).digest('hex')
}

function extractTokens(text: string): string[] {
  const matches = text.toLowerCase().match(/[a-z0-9][a-z0-9+.#/-]*|[\u4e00-\u9fff]+/g) ?? []
  const tokens: string[] = []

  for (const part of matches) {
    if (/^[\u4e00-\u9fff]+$/.test(part) && part.length > 2) {
      for (let i = 0; i < part.length - 1; i += 1) {
        tokens.push(part.slice(i, i + 2))
      }
    } else {
      tokens.push(part)
    }
  }

  return tokens.filter((token) => token.length > 0 && !STOPWORDS.has(token))
}

function keywordOverlapScore(question: string, title: string, content: string): number {
  const querySet = new Set(extractTokens(question))
  if (querySet.size === 0) {
    return 0
  }

  const chunkSet = new Set(extractTokens(`${title}\n${content}`))
  if (chunkSet.size === 0) {
    return 0
  }

  let common = 0
  for (const token of querySet) {
    if (chunkSet.has(token)) {
      common += 1
    }
  }

  const coverage = common / querySet.size
  const density = common / Math.max(6, chunkSet.size)
  return Math.min(1, coverage * 0.85 + density * 0.15)
}

function detectQueryIntents(question: string): Set<IntentCategory> {
  const q = question.toLowerCase()
  const intents = new Set<IntentCategory>()

  if (q.includes('项目') || q.includes('project') || q.includes('作品')) {
    intents.add('project')
  }
  if (q.includes('工作') || q.includes('实习') || q.includes('co-op') || q.includes('experience')) {
    intents.add('work')
  }
  if (q.includes('教育') || q.includes('学校') || q.includes('学历') || q.includes('education')) {
    intents.add('education')
  }
  if (q.includes('技能') || q.includes('tech') || q.includes('stack') || q.includes('skills')) {
    intents.add('skills')
  }
  if (q.includes('联系') || q.includes('邮箱') || q.includes('linkedin') || q.includes('contact')) {
    intents.add('contact')
  }
  if (q.includes('地点') || q.includes('城市') || q.includes('map') || q.includes('location')) {
    intents.add('location')
  }

  return intents
}

function detectChunkCategories(title: string, source: string): Set<IntentCategory> {
  const text = `${title}\n${source}`.toLowerCase()
  const categories = new Set<IntentCategory>()

  if (text.includes('project')) {
    categories.add('project')
  }
  if (text.includes('role') || text.includes('work experience') || text.includes('experience')) {
    categories.add('work')
  }
  if (text.includes('school') || text.includes('education')) {
    categories.add('education')
  }
  if (text.includes('skills')) {
    categories.add('skills')
  }
  if (text.includes('contact') || text.includes('links')) {
    categories.add('contact')
  }
  if (text.includes('location') || text.includes('map')) {
    categories.add('location')
  }

  return categories
}

function intentScoreAdjustment(queryIntents: Set<IntentCategory>, chunkCategories: Set<IntentCategory>): number {
  if (queryIntents.size === 0 || chunkCategories.size === 0) {
    return 0
  }

  let hasMatch = false
  for (const intent of queryIntents) {
    if (chunkCategories.has(intent)) {
      hasMatch = true
      break
    }
  }

  return hasMatch ? INTENT_MATCH_BOOST : -INTENT_MISMATCH_PENALTY
}

function hasCategoryIntersection(left: Set<IntentCategory>, right: Set<IntentCategory>): boolean {
  for (const item of left) {
    if (right.has(item)) {
      return true
    }
  }
  return false
}

function detectPrimaryIntent(question: string): IntentCategory | null {
  const intents = detectQueryIntents(question)
  const order: IntentCategory[] = ['project', 'work', 'education', 'skills', 'contact', 'location']
  for (const item of order) {
    if (intents.has(item)) {
      return item
    }
  }
  return null
}

function isHardIntentMatch(intent: IntentCategory, title: string, source: string, content: string): boolean {
  const text = `${title}\n${source}\n${content}`.toLowerCase()

  if (intent === 'project') {
    return (
      text.includes('project') ||
      text.includes('项目') ||
      text.includes('作品') ||
      text.includes('url / repo') ||
      text.includes('tech stack')
    )
  }
  if (intent === 'work') {
    return (
      text.includes('work experience') ||
      text.includes('role') ||
      text.includes('company') ||
      text.includes('实习') ||
      text.includes('co-op')
    )
  }
  if (intent === 'education') {
    return text.includes('education') || text.includes('school') || text.includes('学历') || text.includes('学校')
  }
  if (intent === 'skills') {
    return text.includes('skills') || text.includes('programming languages') || text.includes('frameworks')
  }
  if (intent === 'contact') {
    return text.includes('contact') || text.includes('email') || text.includes('linkedin') || text.includes('links')
  }
  return text.includes('location') || text.includes('map') || text.includes('latitude') || text.includes('longitude')
}

function parseTopic(topic: AskRequest['topic']): IntentCategory | null {
  if (
    topic === 'project' ||
    topic === 'work' ||
    topic === 'education' ||
    topic === 'skills' ||
    topic === 'contact' ||
    topic === 'location'
  ) {
    return topic
  }
  return null
}

function persistVectorIndex(chunks: VectorChunk[]): void {
  const folder = path.dirname(VECTOR_INDEX_PATH)
  if (!fs.existsSync(folder)) {
    fs.mkdirSync(folder, { recursive: true })
  }
  fs.writeFileSync(VECTOR_INDEX_PATH, JSON.stringify(chunks, null, 2), 'utf-8')
}

export async function askWithRag(config: AppConfig, request: AskRequest): Promise<AskResponse> {
  const chunks = readLocalVectorIndex()

  if (chunks.length === 0) {
    return {
      answer: '知识库还没有建立向量索引。请先调用 /api/reindex 生成索引。',
      provider: config.llmProvider,
      sources: []
    }
  }

  const questionEmbedding = await generateEmbedding(config, request.question)
  const forcedTopic = parseTopic(request.topic)
  const queryIntents = forcedTopic ? new Set<IntentCategory>([forcedTopic]) : detectQueryIntents(request.question)
  const primaryIntent = forcedTopic ?? detectPrimaryIntent(request.question)

  const ranked = chunks
    .map((item) => {
      const semanticScore = cosineSimilarity(questionEmbedding.embedding, item.embedding)
      const keywordScore = keywordOverlapScore(request.question, item.title, item.content)
      const chunkCategories = detectChunkCategories(item.title, item.source)
      const intentAdjustment = intentScoreAdjustment(queryIntents, chunkCategories)
      const score = semanticScore * SEMANTIC_WEIGHT + keywordScore * KEYWORD_WEIGHT + intentAdjustment

      return {
        item,
        score,
        semanticScore,
        keywordScore,
        intentAdjustment,
        chunkCategories
      }
    })
    .sort((left, right) => right.score - left.score)

  const preferredCategories = queryIntents
  const intentFiltered = preferredCategories.size > 0
    ? ranked.filter((entry) => hasCategoryIntersection(entry.chunkCategories, preferredCategories))
    : ranked
  const softIntentPool = intentFiltered.length > 0 ? intentFiltered : ranked
  const hardIntentPool =
    primaryIntent === null
      ? []
      : ranked.filter((entry) => isHardIntentMatch(primaryIntent, entry.item.title, entry.item.source, entry.item.content))
  const rankingPool = hardIntentPool.length > 0 ? hardIntentPool : softIntentPool

  const topK = Math.max(1, request.topK ?? DEFAULT_TOP_K)
  const matchedAboveThreshold = rankingPool.filter((entry) => entry.score >= MIN_RETRIEVAL_SCORE)
  const topChunks = (matchedAboveThreshold.length > 0 ? matchedAboveThreshold : rankingPool).slice(0, topK)

  if (topChunks.length === 0) {
    return {
      answer: '知识库暂无可用片段。请先执行 /api/reindex 或补充 MY_INFO.md 内容。',
      provider: config.llmProvider,
      sources: []
    }
  }

  if (topChunks[0].score < MIN_CONFIDENCE_SCORE) {
    return {
      answer: '我在当前知识库中没有找到足够相关的信息。请换一种更具体的问法（例如项目名、公司名、时间段）。',
      provider: config.llmProvider,
      sources: topChunks.map((item) => ({
        id: item.item.id,
        title: item.item.title,
        source: item.item.source,
        score: Number(item.score.toFixed(4))
      }))
    }
  }

  const llmResult = await generateAnswer(config, {
    question: request.question,
    contextChunks: topChunks.map((item) => item.item.content),
    lang: request.lang ?? 'auto'
  })

  return {
    answer: llmResult.answer,
    provider: config.llmProvider,
    sources: topChunks.map((item) => ({
      id: item.item.id,
      title: item.item.title,
      source: item.item.source,
      score: Number(item.score.toFixed(4))
    }))
  }
}

export async function reindexKnowledge(config: AppConfig): Promise<ReindexResponse> {
  if (!fs.existsSync(KNOWLEDGE_FILE_PATH)) {
    throw new Error('MY_INFO.md not found. Please create the knowledge file first.')
  }

  const markdown = fs.readFileSync(KNOWLEDGE_FILE_PATH, 'utf-8')
  const rawChunks = buildChunksFromMarkdown(markdown)
  const previousChunks = readLocalVectorIndex()

  if (rawChunks.length === 0) {
    throw new Error('No valid chunks were generated from MY_INFO.md')
  }

  const now = new Date().toISOString()
  const vectorChunks: VectorChunk[] = []
  const previousByHash = new Map(previousChunks.map((item) => [item.hash, item]))
  const currentHashes = new Set<string>()
  let reusedCount = 0
  let newCount = 0

  for (let index = 0; index < rawChunks.length; index += 1) {
    const chunk = rawChunks[index]
    const hash = hashChunk(chunk.title, chunk.source, chunk.content)
    currentHashes.add(hash)
    const reused = previousByHash.get(hash)

    if (reused) {
      reusedCount += 1
      vectorChunks.push({
        ...reused,
        id: reused.id,
        title: chunk.title,
        content: chunk.content,
        source: chunk.source,
        hash
      })
      continue
    }

    const embedding = await generateEmbedding(config, chunk.content)
    newCount += 1

    vectorChunks.push({
      id: `chunk-${index + 1}`,
      title: chunk.title,
      content: chunk.content,
      embedding: embedding.embedding,
      source: chunk.source,
      hash,
      updatedAt: now
    })
  }

  const removedCount = previousChunks.filter((item) => !currentHashes.has(item.hash)).length
  persistVectorIndex(vectorChunks)

  return {
    ok: true,
    chunkCount: vectorChunks.length,
    newCount,
    reusedCount,
    removedCount,
    message: `Reindex completed. total=${vectorChunks.length}, new=${newCount}, reused=${reusedCount}, removed=${removedCount}.`
  }
}
