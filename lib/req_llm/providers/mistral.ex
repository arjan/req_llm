defmodule ReqLLM.Providers.Mistral do
  @moduledoc """
  Mistral AI provider – OpenAI-compatible chat API.

  ## Implementation

  Uses built-in OpenAI-style encoding/decoding defaults with custom handling for:
  - Tool calls that may omit the "type" field (defaults to "function")
  - Arguments that may be returned as a map OR a JSON string

  ## Mistral-Specific Parameters

  Mistral supports additional parameters via `:provider_options`:
  - `:random_seed` - Deterministic sampling (Mistral uses this instead of OpenAI's `seed`)
  - `:safe_prompt` - Inject a safety prompt before all conversations
  - `:prediction` - Enable speculative decoding with expected content
  - `:parallel_tool_calls` - Enable parallel function calling during tool use
  - `:prompt_mode` - Reasoning mode for reasoning models (`:reasoning`)
  - `:metadata` - Arbitrary key-value metadata for the request

  ## Configuration

      # Add to .env file (automatically loaded)
      MISTRAL_API_KEY=...
  """

  use ReqLLM.Provider,
    id: :mistral,
    default_base_url: "https://api.mistral.ai/v1",
    default_env_key: "MISTRAL_API_KEY"

  use ReqLLM.Provider.Defaults

  import ReqLLM.Provider.Utils, only: [maybe_put: 3]

  alias ReqLLM.Providers.OpenAI.AdapterHelpers

  @provider_schema [
    random_seed: [
      type: :integer,
      doc: "Deterministic sampling for reproducible outputs"
    ],
    safe_prompt: [
      type: :boolean,
      doc: "Inject Mistral's safety prompt before all conversations"
    ],
    prediction: [
      type: :map,
      doc: "Enable speculative decoding with expected content"
    ],
    parallel_tool_calls: [
      type: :boolean,
      doc: "Enable parallel function calling during tool use"
    ],
    prompt_mode: [
      type: {:in, [:reasoning]},
      doc: "Allows toggling reasoning mode for reasoning models"
    ],
    metadata: [
      type: :map,
      doc: "Arbitrary key-value metadata for the request"
    ],
    mistral_structured_output_mode: [
      type: {:in, [:auto, :json_schema, :tool_strict]},
      default: :auto,
      doc: """
      Strategy for structured output generation:
      - `:auto` - Use json_schema when supported (default)
      - `:json_schema` - Force response_format with json_schema
      - `:tool_strict` - Force strict: true on function tools
      """
    ],
    response_format: [
      type: :map,
      doc: "Response format configuration (e.g., json_object or json_schema)"
    ]
  ]

  @impl ReqLLM.Provider
  def prepare_request(:object, model_spec, prompt, opts) do
    compiled_schema = Keyword.fetch!(opts, :compiled_schema)
    {:ok, model} = ReqLLM.model(model_spec)
    mode = determine_output_mode(model, opts)

    case mode do
      :json_schema ->
        prepare_json_schema_request(model_spec, prompt, compiled_schema, opts)

      :tool_strict ->
        prepare_tool_strict_request(model_spec, prompt, compiled_schema, opts)
    end
  end

  def prepare_request(operation, model_spec, input, opts) do
    ReqLLM.Provider.Defaults.prepare_request(__MODULE__, operation, model_spec, input, opts)
  end

  @impl ReqLLM.Provider
  def encode_body(request) do
    request = ReqLLM.Provider.Defaults.default_encode_body(request)
    body = Jason.decode!(request.body)

    provider_opts = request.options[:provider_options] || []

    body =
      body
      |> translate_tool_choice_format()
      |> maybe_put(:random_seed, provider_opts[:random_seed])
      |> maybe_put(:safe_prompt, provider_opts[:safe_prompt])
      |> maybe_put(:prediction, provider_opts[:prediction])
      |> maybe_put(:parallel_tool_calls, provider_opts[:parallel_tool_calls])
      |> maybe_put(:prompt_mode, provider_opts[:prompt_mode])
      |> maybe_put(:metadata, provider_opts[:metadata])
      |> Map.delete("seed")
      |> AdapterHelpers.add_response_format(provider_opts)

    encoded_body = Jason.encode!(body)
    Map.put(request, :body, encoded_body)
  end

  defp translate_tool_choice_format(body) do
    {tool_choice, body_key} =
      cond do
        Map.has_key?(body, :tool_choice) -> {Map.get(body, :tool_choice), :tool_choice}
        Map.has_key?(body, "tool_choice") -> {Map.get(body, "tool_choice"), "tool_choice"}
        true -> {nil, nil}
      end

    {type, name} =
      if is_map(tool_choice) do
        {Map.get(tool_choice, :type) || Map.get(tool_choice, "type"),
         Map.get(tool_choice, :name) || Map.get(tool_choice, "name")}
      else
        {nil, nil}
      end

    if type == "tool" && name do
      replacement =
        if is_map_key(tool_choice, :type) do
          %{type: "function", function: %{name: name}}
        else
          %{"type" => "function", "function" => %{"name" => name}}
        end

      Map.put(body, body_key, replacement)
    else
      body
    end
  end

  @impl ReqLLM.Provider
  def translate_options(_operation, _model, opts) do
    {seed, opts} = Keyword.pop(opts, :seed)

    opts =
      if seed do
        provider_opts = Keyword.get(opts, :provider_options, [])
        provider_opts = Keyword.put(provider_opts, :random_seed, seed)
        Keyword.put(opts, :provider_options, provider_opts)
      else
        opts
      end

    {opts, []}
  end

  @doc """
  Custom attach_stream that ensures translate_options is called for streaming requests.

  This is necessary because the default streaming path doesn't call translate_options,
  which means seed -> random_seed translation wouldn't be applied to streaming requests.
  """
  @impl ReqLLM.Provider
  def attach_stream(model, context, opts, finch_name) do
    {translated_opts, _warnings} = translate_options(:chat, model, opts)
    base_url = ReqLLM.Provider.Options.effective_base_url(__MODULE__, model, translated_opts)
    opts_with_base_url = Keyword.put(translated_opts, :base_url, base_url)

    ReqLLM.Provider.Defaults.default_attach_stream(
      __MODULE__,
      model,
      context,
      opts_with_base_url,
      finch_name
    )
  end

  @impl ReqLLM.Provider
  def decode_stream_event(%{data: data} = event, model) when is_map(data) do
    normalized_event = %{event | data: normalize_streaming_tool_calls(data)}
    ReqLLM.Provider.Defaults.default_decode_stream_event(normalized_event, model)
  end

  def decode_stream_event(event, model) do
    ReqLLM.Provider.Defaults.default_decode_stream_event(event, model)
  end

  defp normalize_streaming_tool_calls(%{"choices" => choices} = data) when is_list(choices) do
    normalized_choices =
      Enum.map(choices, fn
        %{"delta" => %{"tool_calls" => tool_calls} = delta} = choice when is_list(tool_calls) ->
          normalized_tool_calls = Enum.map(tool_calls, &normalize_streaming_tool_call/1)
          %{choice | "delta" => %{delta | "tool_calls" => normalized_tool_calls}}

        choice ->
          choice
      end)

    %{data | "choices" => normalized_choices}
  end

  defp normalize_streaming_tool_calls(data), do: data

  defp normalize_streaming_tool_call(%{"function" => _} = tc) do
    Map.put_new(tc, "type", "function")
  end

  defp normalize_streaming_tool_call(other), do: other

  @impl ReqLLM.Provider
  def decode_response({req, resp}) do
    case resp.status do
      200 ->
        body = ensure_parsed_body(resp.body)

        normalized_body =
          body
          |> normalize_mistral_tool_calls()
          |> normalize_mistral_finish_reason()

        ReqLLM.Provider.Defaults.default_decode_response({req, %{resp | body: normalized_body}})

      _ ->
        ReqLLM.Provider.Defaults.default_decode_response({req, resp})
    end
  end

  defp normalize_mistral_finish_reason(%{"choices" => choices} = body) when is_list(choices) do
    normalized_choices =
      Enum.map(choices, fn
        %{"finish_reason" => "model_length"} = choice ->
          %{choice | "finish_reason" => "length"}

        choice ->
          choice
      end)

    %{body | "choices" => normalized_choices}
  end

  defp normalize_mistral_finish_reason(body), do: body

  # Mistral's tool_calls may differ from OpenAI in two ways:
  # 1. They don't include "type": "function" - we add it
  # 2. Arguments may be a map (already parsed) instead of a JSON string - we stringify it
  defp normalize_mistral_tool_calls(body) when is_map(body) do
    case body do
      %{"choices" => choices} when is_list(choices) ->
        normalized_choices =
          Enum.map(choices, fn choice ->
            case choice do
              %{"message" => %{"tool_calls" => tool_calls} = message} when is_list(tool_calls) ->
                normalized_tool_calls = Enum.map(tool_calls, &normalize_tool_call/1)
                %{choice | "message" => %{message | "tool_calls" => normalized_tool_calls}}

              _ ->
                choice
            end
          end)

        %{body | "choices" => normalized_choices}

      _ ->
        body
    end
  end

  defp normalize_mistral_tool_calls(body), do: body

  defp normalize_tool_call(%{"id" => _, "function" => function} = tc) do
    # Add type: "function" if not present
    tc = Map.put_new(tc, "type", "function")

    # Normalize arguments: if it's a map, encode to JSON string for consistency
    case function do
      %{"arguments" => args} when is_map(args) ->
        normalized_function = %{function | "arguments" => Jason.encode!(args)}
        %{tc | "function" => normalized_function}

      _ ->
        tc
    end
  end

  defp normalize_tool_call(other), do: other

  defp ensure_parsed_body(body) when is_binary(body) do
    case Jason.decode(body) do
      {:ok, parsed} -> parsed
      {:error, _} -> body
    end
  end

  defp ensure_parsed_body(body), do: body

  @dialyzer {:nowarn_function, prepare_json_schema_request: 4}
  defp prepare_json_schema_request(model_spec, prompt, compiled_schema, opts) do
    schema_name = Map.get(compiled_schema, :name, "output_schema")
    json_schema = ReqLLM.Schema.to_json(compiled_schema.schema)

    opts_with_format =
      opts
      |> Keyword.update(
        :provider_options,
        [
          response_format: %{
            type: "json_schema",
            json_schema: %{
              name: schema_name,
              strict: true,
              schema: json_schema
            }
          }
        ],
        fn provider_opts ->
          Keyword.put(provider_opts, :response_format, %{
            type: "json_schema",
            json_schema: %{
              name: schema_name,
              strict: true,
              schema: json_schema
            }
          })
        end
      )
      |> Keyword.delete(:tools)
      |> Keyword.delete(:tool_choice)
      |> Keyword.put(:operation, :object)

    prepare_request(:chat, model_spec, prompt, opts_with_format)
  end

  @dialyzer {:nowarn_function, prepare_tool_strict_request: 4}
  defp prepare_tool_strict_request(model_spec, prompt, compiled_schema, opts) do
    structured_output_tool =
      ReqLLM.Tool.new!(
        name: "structured_output",
        description: "Generate structured output matching the provided schema",
        parameter_schema: compiled_schema.schema,
        strict: true,
        callback: fn _args -> {:ok, "structured output generated"} end
      )

    opts_with_tool =
      opts
      |> Keyword.update(:tools, [structured_output_tool], &[structured_output_tool | &1])
      |> Keyword.put(:tool_choice, %{
        type: "function",
        function: %{name: "structured_output"}
      })
      |> Keyword.put(:operation, :object)

    prepare_request(:chat, model_spec, prompt, opts_with_tool)
  end

  @doc """
  Determine the structured output mode for a model and options.

  ## Mode Selection Logic
  1. If explicit `mistral_structured_output_mode` is set, use it
  2. If `response_format` with `json_schema` is present in options, force `:json_schema`
  3. If `:auto`: Use `:json_schema` when no other tools present, else `:tool_strict`
  """
  @spec determine_output_mode(LLMDB.Model.t(), keyword()) :: :json_schema | :tool_strict
  def determine_output_mode(model, opts) do
    explicit_mode =
      opts
      |> Keyword.get(:provider_options, [])
      |> Keyword.get(:mistral_structured_output_mode)

    has_response_format_json_schema =
      opts
      |> Keyword.get(:response_format)
      |> case do
        %{json_schema: _} -> true
        _ -> false
      end

    cond do
      explicit_mode && explicit_mode != :auto ->
        explicit_mode

      has_response_format_json_schema ->
        :json_schema

      true ->
        auto_select_mode(model, opts)
    end
  end

  defp auto_select_mode(_model, opts) do
    if has_other_tools?(opts) do
      :tool_strict
    else
      :json_schema
    end
  end

  defp has_other_tools?(opts) do
    tools = Keyword.get(opts, :tools, [])

    Enum.any?(tools, fn tool ->
      name =
        case tool do
          %{name: n} -> n
          _ -> nil
        end

      name != "structured_output"
    end)
  end

  @doc """
  Check if Mistral models support native structured outputs via response_format with json_schema.

  All recent Mistral models support JSON schema response format.
  """
  @spec supports_native_structured_outputs?() :: boolean()
  def supports_native_structured_outputs?, do: true
end
