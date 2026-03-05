export default async function handler(req: any, res: any) {
  try {
    const mod = await import('../server/app')
    const app = mod.default
    return app(req, res)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown bootstrap error'
    const stack = error instanceof Error ? error.stack : undefined

    console.error('[api/bootstrap] failed to load app:', error)

    return res.status(500).json({
      ok: false,
      stage: 'bootstrap',
      message,
      stack
    })
  }
}
