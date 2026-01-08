# Mistral

Access Mistral AI models including Mistral Small, Medium, Large, Codestral, and Pixtral.

## Configuration

```bash
MISTRAL_API_KEY=...
```

## Supported Models

| Model | Description | Context | Vision |
|-------|-------------|---------|--------|
| `mistral-small-latest` | Fast, efficient model | 32K | No |
| `mistral-medium-latest` | Balanced performance | 32K | No |
| `mistral-large-latest` | Most capable model | 128K | No |
| `codestral-latest` | Optimized for code | 256K | No |
| `pixtral-large-latest` | Vision-enabled model | 128K | Yes |

## Provider Options

Passed via `:provider_options` keyword:

### `random_seed`
- **Type**: Integer
- **Purpose**: Deterministic sampling for reproducible outputs
- **Note**: ReqLLM auto-translates `seed` to `random_seed` for Mistral
- **Example**: `provider_options: [random_seed: 42]`

### `safe_prompt`
- **Type**: Boolean
- **Purpose**: Inject Mistral's safety prompt before all conversations
- **Example**: `provider_options: [safe_prompt: true]`

### `prediction`
- **Type**: Map
- **Purpose**: Enable speculative decoding with expected content
- **Example**:
  ```elixir
  provider_options: [
    prediction: %{
      type: "content",
      content: "Expected output pattern..."
    }
  ]
  ```

### `parallel_tool_calls`
- **Type**: Boolean
- **Purpose**: Enable parallel function calling during tool use
- **Example**: `provider_options: [parallel_tool_calls: true]`

### `prompt_mode`
- **Type**: Atom (`:reasoning`)
- **Purpose**: Toggle reasoning mode for reasoning models
- **Example**: `provider_options: [prompt_mode: :reasoning]`

### `metadata`
- **Type**: Map
- **Purpose**: Arbitrary key-value metadata for the request
- **Example**: `provider_options: [metadata: %{"user_id" => "abc123"}]`

## Seed Translation

ReqLLM automatically translates the standard `seed` option to Mistral's `random_seed`:

```elixir
# Both of these are equivalent for Mistral:
ReqLLM.generate_text("mistral:mistral-large-latest", "Hello", seed: 42)
ReqLLM.generate_text("mistral:mistral-large-latest", "Hello", 
  provider_options: [random_seed: 42])
```

## Tool Calling

Mistral supports function calling with automatic normalization:

```elixir
weather_tool = ReqLLM.Tool.new!(
  name: "get_weather",
  description: "Get current weather for a location",
  parameter_schema: [
    location: [type: :string, required: true, doc: "City name"]
  ],
  callback: fn %{"location" => loc} -> {:ok, "Sunny in #{loc}"} end
)

{:ok, response} = ReqLLM.generate_text(
  "mistral:mistral-large-latest",
  "What's the weather in Paris?",
  tools: [weather_tool]
)
```

## Streaming

Mistral supports streaming with full SSE compatibility:

```elixir
{:ok, stream_response} = ReqLLM.stream_text(
  "mistral:mistral-large-latest",
  "Write a haiku about coding"
)

stream_response
|> ReqLLM.StreamResponse.text_stream()
|> Enum.each(&IO.write/1)
```

## Object Generation

Generate structured outputs using Mistral's tool calling:

```elixir
schema = [
  name: [type: :string, required: true],
  age: [type: :integer, required: true],
  occupation: [type: :string]
]

{:ok, person} = ReqLLM.generate_object(
  "mistral:mistral-large-latest",
  "Generate a software engineer profile",
  schema,
  temperature: 0
)
# => %{"name" => "Alice Chen", "age" => 32, "occupation" => "Software Engineer"}
```

## Vision (Pixtral)

Pixtral models support image inputs:

```elixir
{:ok, response} = ReqLLM.generate_text(
  "mistral:pixtral-large-latest",
  [
    ReqLLM.Message.ContentPart.text("What's in this image?"),
    ReqLLM.Message.ContentPart.image_url("https://example.com/photo.jpg")
  ]
)
```

## Resources

- [Mistral API Documentation](https://docs.mistral.ai/api/)
- [Model Overview](https://docs.mistral.ai/getting-started/models/)
- [Pricing](https://mistral.ai/pricing/)

