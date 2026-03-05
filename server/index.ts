import { loadConfig } from './config.js'
import app from './app.js'

const config = loadConfig()

app.listen(config.port, () => {
  console.log(`[rag-server] running on http://localhost:${config.port}`)
})
