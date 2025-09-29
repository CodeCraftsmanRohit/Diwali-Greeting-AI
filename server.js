// server.js (Gemini / Google GenAI SDK - CommonJS)
require('dotenv').config();
const express = require('express');
const cors = require('cors');

// Import the Google Gen AI SDK
// Official SDK package: @google/genai
const { GoogleGenAI } = require('@google/genai');

const PORT = process.env.PORT || 4000;
const MODEL = process.env.MODEL || 'gemini-2.5-flash'; // change if you want another model

// Initialize Google GenAI client.
// If GEMINI_API_KEY env var is set, the SDK will use it.
// For Vertex AI usage (GCP), you can initialize differently - see notes below.
const apiKey = process.env.GEMINI_API_KEY || null;

let ai;
if (apiKey) {
  // Use Developer API key path
  ai = new GoogleGenAI({ apiKey });
} else if (process.env.GOOGLE_CLOUD_PROJECT) {
  // Vertex AI path (uses application default credentials / GCP service account)
  ai = new GoogleGenAI({
    vertexai: true,
    project: process.env.GOOGLE_CLOUD_PROJECT,
    location: process.env.GOOGLE_CLOUD_LOCATION || 'global',
  });
} else {
  console.warn('No GEMINI_API_KEY and no GOOGLE_CLOUD_PROJECT set. Set GEMINI_API_KEY for Gemini Developer API or set GOOGLE_CLOUD_PROJECT for Vertex AI.');
  // still create client — SDK might pick env var automatically, but warn users.
  ai = new GoogleGenAI();
}

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

if (!apiKey && !process.env.GOOGLE_CLOUD_PROJECT) {
  console.error('ERROR: No GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT configured in environment.');
  // keep server up for local debugging, but requests will likely error out.
}

app.post('/api/generate', async (req, res) => {
  try {
    const prompt = (req.body.prompt || '').trim();
    if (!prompt) return res.status(400).json({ error: 'Prompt is required' });

    // System instruction (behaviour) and the user prompt.
    const systemInstruction = "You are a friendly assistant that writes warm, short Diwali greetings.";

    // Build `contents` for the SDK generateContent call.
    // The SDK accepts either a single string or an array of content parts.
    const contents = [
      `System: ${systemInstruction}`,
      `User: Create a short Diwali greeting for: ${prompt}`
    ].join('\n');

    // Call the SDK: generateContent (or models.generate_content) depending on SDK version.
    // The SDK returns an object with .text (string) or structured response.
    const response = await ai.models.generateContent({
      model: MODEL,
      contents: contents,
      // optional configuration:
      // config: {
      //   temperature: 0.8,
      //   maxOutputTokens: 160,
      // }
    });

    // The SDK often exposes the generated text as response.text
    // but shape may differ; we try a few fallbacks.
    let output = null;
    if (!response) {
      return res.status(502).json({ error: 'No response from Gemini SDK', raw: response });
    }

    if (typeof response.text === 'string') {
      output = response.text;
    } else if (response?.candidates && Array.isArray(response.candidates) && response.candidates[0]?.content) {
      // some SDK shapes put the generated content inside candidates[].content
      output = response.candidates[0].content;
    } else if (response?.output?.[0]?.content) {
      output = response.output[0].content;
    } else {
      // fallback: stringify the response so developer can inspect it
      output = null;
    }

    // Log raw response for debugging on server console (trimmed)
    try {
      console.log('Gemini raw response:', JSON.stringify(response, null, 2).slice(0, 3000));
    } catch (e) {
      console.log('Gemini raw response (unstringifiable)');
    }

    if (!output) {
      return res.status(502).json({ error: 'No output in Gemini response', raw: response });
    }

    return res.json({ output: String(output).trim(), raw: response });
  } catch (err) {
    console.error('Server error calling Gemini:', err);
    // Don't leak secrets in error messages
    return res.status(500).json({ error: 'Internal server error', details: err?.message || String(err) });
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT} (model=${MODEL})`);
});
