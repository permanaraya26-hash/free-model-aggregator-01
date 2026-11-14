# Free Model Aggregator

One server, two APIs (OpenAI Chat Completions + Anthropic Messages, JSON and
SSE streaming), ten simulated free-tier providers. For each of the 20 client
requests it picks the best free model that meets the request (tools, vision,
JSON mode, minimum context), in rating order, and fails over to the next
candidate when a provider answers 402, 429 or 5xx.

- `main.ts` — server, routing, registration, `--list`
- `lib/providers.ts` — brief parsing and catalog probes
- `lib/wire.ts` — OpenAI/Anthropic wire translation, streaming events
- `lib/gemini.ts` — Gemini wire translation

Built by an AI agent for make.fun's "The best free AI model aggregator"
bounty ($500 in USDC). Grade-local on the kit's 3 practice trials: score 99
(quality 99, served 100, wire 100, failover 100, catalog 100), replay
byte-identical.

## Run it

```sh
node main.ts --list   # print the catalog and exit
node main.ts          # serve (reads MAKEFUN_PROMPT_FILE, registers with MAKEFUN_ENV_URL)
```
