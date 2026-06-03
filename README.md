# Backend

## Run locally

1. Copy env file:

```bash
cp .env.example .env
```

2. Set your OpenAI API key in `.env`:

```bash
OPENAI_API_KEY=your_openai_api_key
LLM_PROVIDER=openai
EMBED_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
EMBED_MODEL=text-embedding-3-small
```

3. Install dependencies:

```bash
npm install
```

4. Start dev server:

```bash
npm run dev
```

## Vercel entrypoint

- `api/index.ts`
