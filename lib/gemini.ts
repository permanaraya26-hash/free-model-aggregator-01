// The Gemini wire: request builders (OpenAI / Anthropic client -> Gemini) and
// response translators (Gemini JSON / SSE -> OpenAI / Anthropic client shapes).
//
// The provider reads the model from the URL path
// (/v1beta/models/<model>:generateContent or :streamGenerateContent), the
// system prompt from systemInstruction, tools from functionDeclarations, and
// JSON mode from generationConfig.responseMimeType. It answers JSON
// ({candidates, usageMetadata, modelVersion}) or SSE (one frame per 13-char
// piece, final frame with finishReason STOP + usageMetadata). Everything here
// is a pure function of its arguments: no clock, no randomness.

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One OpenAI content part as Gemini parts: text, or an inline image / file reference. */
function geminiPartsOfOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

/** One Anthropic content block as Gemini parts: text, or an inline image / file reference. */
function geminiPartsOfAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { fileUri: source.url } });
      }
    }
  }
  return parts;
}

/** The shared generationConfig: max tokens, sampling, stops, and JSON mode only when asked. */
function generationConfig(opts: {
  maxTokens?: unknown;
  temperature?: unknown;
  topP?: unknown;
  stops?: unknown;
  jsonMode?: boolean;
}): Json {
  const config: Json = {};
  if (typeof opts.maxTokens === 'number') config.maxOutputTokens = opts.maxTokens;
  if (typeof opts.temperature === 'number') config.temperature = opts.temperature;
  if (typeof opts.topP === 'number') config.topP = opts.topP;
  if (Array.isArray(opts.stops) && opts.stops.length > 0) config.stopSequences = opts.stops.filter((s) => typeof s === 'string');
  else if (typeof opts.stops === 'string' && opts.stops !== '') config.stopSequences = [opts.stops];
  if (opts.jsonMode === true) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. System
 * messages become systemInstruction (texts joined with "\n", as the provider's
 * echo joins them); assistant messages become role "model"; image_url parts
 * become inlineData (data URLs) or fileData (plain URLs); tools become
 * functionDeclarations; response_format json_object/json_schema becomes
 * responseMimeType "application/json".
 */
export function openAIToGemini(body: Json, model: string): { body: Json; stream: boolean } {
  const contents: Json[] = [];
  const systemTexts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    const parts = geminiPartsOfOpenAI(raw.content);
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const out: Json = { contents };
  if (systemTexts.length > 0) out.systemInstruction = { parts: [{ text: systemTexts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  out.generationConfig = generationConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens,
    temperature: body.temperature,
    topP: body.top_p,
    stops: body.stop,
    jsonMode: format === 'json_object' || format === 'json_schema',
  });
  return { body: out, stream: body.stream === true };
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The system
 * prompt becomes systemInstruction; messages map user -> "user", assistant ->
 * "model"; image blocks become inlineData/fileData; tools become
 * functionDeclarations; stop_sequences become stopSequences. Anthropic has no
 * JSON mode to carry, so responseMimeType is never set here.
 */
export function anthropicToGemini(body: Json, model: string): { body: Json; stream: boolean } {
  const contents: Json[] = [];
  const system = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const parts = geminiPartsOfAnthropic(raw.content);
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  out.generationConfig = generationConfig({
    maxTokens: body.max_tokens,
    temperature: body.temperature,
    topP: body.top_p,
    stops: body.stop_sequences,
  });
  return { body: out, stream: body.stream === true };
}

/** The Gemini request URL for one model: the model goes in the path, stream or not. */
export function geminiUrl(base: string, model: string, stream: boolean): string {
  return `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
}

/** Gemini's finishReason as an OpenAI finish_reason (same family). */
export function geminiStopToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as an Anthropic stop_reason (same family). */
export function geminiStopToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The answer text of a Gemini JSON body: its first candidate's parts, joined whole. */
export function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini JSON answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const prompt = numberOrZero(usage.promptTokenCount);
  const completion = numberOrZero(usage.candidatesTokenCount);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiText(body) }, finish_reason: geminiStopToOpenAI(candidate.finishReason), logprobs: null }],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

/** A Gemini JSON answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiText(body) }],
    stop_reason: geminiStopToAnthropic(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** One Gemini SSE stream, read whole: its text, stop reason and usage. */
export async function readGeminiStream(stream: ReadableStream<Uint8Array>): Promise<{ text: string; stop: string; input: number; output: number }> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  let stop = 'STOP';
  let input = 0;
  let output = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '[DONE]') continue;
          let chunk: unknown;
          try {
            chunk = JSON.parse(payload);
          } catch {
            continue;
          }
          if (!isRecord(chunk)) continue;
          const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
          const candidate = isRecord(candidates[0]) ? candidates[0] : {};
          const content = isRecord(candidate.content) ? candidate.content : {};
          const parts = Array.isArray(content.parts) ? content.parts : [];
          for (const part of parts) if (isRecord(part) && typeof part.text === 'string') text += part.text;
          if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
          if (isRecord(chunk.usageMetadata)) {
            input = numberOrZero(chunk.usageMetadata.promptTokenCount);
            output = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const line of buffer.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        let chunk: unknown;
        try {
          chunk = JSON.parse(trimmed.slice(5).trim());
        } catch {
          continue;
        }
        if (!isRecord(chunk)) continue;
        const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
        const candidate = isRecord(candidates[0]) ? candidates[0] : {};
        const content = isRecord(candidate.content) ? candidate.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const part of parts) if (isRecord(part) && typeof part.text === 'string') text += part.text;
        if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
        if (isRecord(chunk.usageMetadata)) {
          input = numberOrZero(chunk.usageMetadata.promptTokenCount);
          output = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  return { text, stop, input, output };
}
