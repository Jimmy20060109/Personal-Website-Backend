import { loadConfig } from './config'
import app from './app'

const config = loadConfig()

app.listen(config.port, () => {
  console.log(`[rag-server] running on http://localhost:${config.port}`)
})
