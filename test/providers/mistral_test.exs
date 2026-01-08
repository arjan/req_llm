defmodule ReqLLM.Providers.MistralTest do
  @moduledoc """
  Provider-level tests for Mistral implementation.

  Tests the provider contract directly without going through Generation layer.
  Focus: prepare_request -> attach -> request -> decode pipeline.
  """

  use ReqLLM.ProviderCase, provider: ReqLLM.Providers.Mistral

  alias ReqLLM.Context
  alias ReqLLM.Providers.Mistral

  describe "provider contract" do
    test "provider identity and configuration" do
      assert Mistral.provider_id() == :mistral
      assert Mistral.base_url() == "https://api.mistral.ai/v1"
      assert Mistral.default_env_key() == "MISTRAL_API_KEY"
    end

    test "provider schema separation from core options" do
      schema_keys = Mistral.provider_schema().schema |> Keyword.keys()
      core_keys = ReqLLM.Provider.Options.generation_schema().schema |> Keyword.keys()

      # Provider-specific keys should not overlap with core generation keys
      overlap = MapSet.intersection(MapSet.new(schema_keys), MapSet.new(core_keys))

      assert MapSet.size(overlap) == 0,
             "Schema overlap detected: #{inspect(MapSet.to_list(overlap))}"
    end

    test "provider schema combined with generation schema includes all core keys" do
      full_schema = Mistral.provider_extended_generation_schema()
      full_keys = Keyword.keys(full_schema.schema)
      core_keys = ReqLLM.Provider.Options.all_generation_keys()

      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))
      missing = core_without_meta -- full_keys
      assert missing == [], "Missing core generation keys in extended schema: #{inspect(missing)}"
    end

    test "provider_extended_generation_schema includes both base and provider options" do
      extended_schema = Mistral.provider_extended_generation_schema()
      extended_keys = extended_schema.schema |> Keyword.keys()

      # Should include all core generation keys
      core_keys = ReqLLM.Provider.Options.all_generation_keys()
      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))

      for core_key <- core_without_meta do
        assert core_key in extended_keys,
               "Extended schema missing core key: #{core_key}"
      end

      # Should include provider-specific keys
      provider_keys = Mistral.provider_schema().schema |> Keyword.keys()

      for provider_key <- provider_keys do
        assert provider_key in extended_keys,
               "Extended schema missing provider key: #{provider_key}"
      end
    end
  end

  describe "request preparation & pipeline wiring" do
    test "prepare_request creates configured request" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()
      opts = [temperature: 0.7, max_tokens: 100]

      {:ok, request} = Mistral.prepare_request(:chat, model, context, opts)

      assert %Req.Request{} = request
      assert request.url.path == "/chat/completions"
      assert request.method == :post
    end

    test "attach configures authentication and pipeline" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      opts = [temperature: 0.5, max_tokens: 50]

      request = Req.new() |> Mistral.attach(model, opts)

      # Verify authentication
      auth_header = Enum.find(request.headers, fn {name, _} -> name == "authorization" end)
      assert auth_header != nil
      {_, [auth_value]} = auth_header
      assert String.starts_with?(auth_value, "Bearer ")

      # Verify pipeline steps
      request_steps = Keyword.keys(request.request_steps)
      response_steps = Keyword.keys(request.response_steps)

      assert :llm_encode_body in request_steps
      assert :llm_decode_response in response_steps
    end

    test "error handling for invalid configurations" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      # Unsupported operation
      {:error, error} = Mistral.prepare_request(:unsupported, model, context, [])
      assert %ReqLLM.Error.Invalid.Parameter{} = error

      # Provider mismatch
      {:ok, wrong_model} = ReqLLM.model("xai:grok-3")

      assert_raise ReqLLM.Error.Invalid.Provider, fn ->
        Req.new() |> Mistral.attach(wrong_model, [])
      end
    end
  end

  describe "body encoding & context translation" do
    test "encode_body without tools" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false
        ]
      }

      updated_request = Mistral.encode_body(mock_request)

      assert is_binary(updated_request.body)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["model"] == "mistral-large-latest"
      assert is_list(decoded["messages"])
      assert length(decoded["messages"]) == 2
      assert decoded["stream"] == false
      refute Map.has_key?(decoded, "tools")

      [system_msg, user_msg] = decoded["messages"]
      assert system_msg["role"] == "system"
      assert user_msg["role"] == "user"
    end

    test "encode_body with tools" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test tool",
          parameter_schema: [
            name: [type: :string, required: true, doc: "A name parameter"]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          tools: [tool]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["tools"])
      assert length(decoded["tools"]) == 1

      [encoded_tool] = decoded["tools"]
      assert encoded_tool["function"]["name"] == "test_tool"
    end

    test "encode_body handles standard OpenAI options" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      test_cases = [
        {[temperature: 0.2, max_tokens: 55, top_p: 0.9],
         fn json ->
           assert json["temperature"] == 0.2
           assert json["max_tokens"] == 55
           assert json["top_p"] == 0.9
         end},
        {[presence_penalty: 0.2, user: "test_user"],
         fn json ->
           assert json["presence_penalty"] == 0.2
           assert json["user"] == "test_user"
         end}
      ]

      for {options, assertion} <- test_cases do
        full_options = [context: context, model: model.model, stream: false] ++ options
        mock_request = %Req.Request{options: full_options}
        updated_request = Mistral.encode_body(mock_request)
        decoded = Jason.decode!(updated_request.body)
        assertion.(decoded)
      end
    end
  end

  describe "response decoding & normalization" do
    test "decode_response handles non-streaming responses" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      mock_json_response = openai_format_json_fixture(model: "mistral-large-latest")

      mock_resp = %Req.Response{
        status: 200,
        body: mock_json_response
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, stream: false, id: "mistral:mistral-large-latest"],
        private: %{req_llm_model: model}
      }

      {req, resp} = Mistral.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Response{} = resp.body

      response = resp.body
      assert is_binary(response.id)
      assert response.model == model.model
      assert response.stream? == false

      # Verify message normalization
      assert response.message.role == :assistant
      text = ReqLLM.Response.text(response)
      assert is_binary(text)
      assert String.length(text) > 0
      assert response.finish_reason in [:stop, :length]

      # Verify usage normalization
      assert is_integer(response.usage.input_tokens)
      assert is_integer(response.usage.output_tokens)
      assert is_integer(response.usage.total_tokens)

      # Verify context advancement (original + assistant)
      assert length(response.context.messages) == 3
      assert List.last(response.context.messages).role == :assistant
    end

    test "decode_response handles API errors with non-200 status" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")

      error_body = %{
        "error" => %{
          "message" => "Invalid API key",
          "type" => "authentication_error",
          "code" => "invalid_api_key"
        }
      }

      mock_resp = %Req.Response{
        status: 401,
        body: error_body
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, id: "mistral:mistral-large-latest"],
        private: %{req_llm_model: model}
      }

      {req, error} = Mistral.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Error.API.Response{} = error
      assert error.status == 401
      assert error.reason =~ " API error"
      assert error.response_body == error_body
    end
  end

  describe "usage extraction" do
    test "extract_usage with valid usage data" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")

      body_with_usage = %{
        "usage" => %{
          "prompt_tokens" => 10,
          "completion_tokens" => 20,
          "total_tokens" => 30
        }
      }

      {:ok, usage} = Mistral.extract_usage(body_with_usage, model)
      assert usage["prompt_tokens"] == 10
      assert usage["completion_tokens"] == 20
      assert usage["total_tokens"] == 30
    end

    test "extract_usage with missing usage data" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      body_without_usage = %{"choices" => []}

      {:error, :no_usage_found} = Mistral.extract_usage(body_without_usage, model)
    end

    test "extract_usage with invalid body type" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")

      {:error, :invalid_body} = Mistral.extract_usage("invalid", model)
      {:error, :invalid_body} = Mistral.extract_usage(nil, model)
      {:error, :invalid_body} = Mistral.extract_usage(123, model)
    end
  end

  describe "error handling & robustness" do
    test "context validation" do
      # Multiple system messages should fail
      invalid_context =
        Context.new([
          Context.system("System 1"),
          Context.system("System 2"),
          Context.user("Hello")
        ])

      assert_raise ReqLLM.Error.Validation.Error,
                   ~r/Context should have at most one system message/,
                   fn ->
                     Context.validate!(invalid_context)
                   end
    end
  end

  describe "option translation" do
    test "provider implements translate_options/3" do
      assert function_exported?(Mistral, :translate_options, 3)
    end

    test "translate_options converts seed to random_seed in provider_options" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")

      opts = [temperature: 0.7, seed: 42]
      {translated_opts, warnings} = Mistral.translate_options(:chat, model, opts)

      refute Keyword.has_key?(translated_opts, :seed)
      provider_opts = Keyword.get(translated_opts, :provider_options, [])
      assert Keyword.get(provider_opts, :random_seed) == 42
      assert warnings == []
    end

    test "translate_options preserves other options unchanged" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")

      opts = [temperature: 0.7, max_tokens: 100, top_p: 0.9]
      {translated_opts, warnings} = Mistral.translate_options(:chat, model, opts)

      assert Keyword.get(translated_opts, :temperature) == 0.7
      assert Keyword.get(translated_opts, :max_tokens) == 100
      assert Keyword.get(translated_opts, :top_p) == 0.9
      assert warnings == []
    end

    test "translate_options without seed passes through unchanged" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")

      opts = [temperature: 0.5]
      {translated_opts, warnings} = Mistral.translate_options(:chat, model, opts)

      assert Keyword.get(translated_opts, :temperature) == 0.5
      refute Keyword.has_key?(translated_opts, :provider_options)
      assert warnings == []
    end
  end

  describe "Mistral-specific features" do
    test "encode_body includes random_seed from provider_options" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          provider_options: [random_seed: 12_345]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["random_seed"] == 12_345
    end

    test "encode_body includes safe_prompt from provider_options" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          provider_options: [safe_prompt: true]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["safe_prompt"] == true
    end

    test "encode_body includes prediction from provider_options" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      prediction = %{type: "content", content: "Expected output..."}

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          provider_options: [prediction: prediction]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["prediction"] == %{"type" => "content", "content" => "Expected output..."}
    end

    test "encode_body includes all Mistral-specific options together" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          provider_options: [
            random_seed: 42,
            safe_prompt: true,
            prediction: %{type: "content", content: "test"}
          ]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["random_seed"] == 42
      assert decoded["safe_prompt"] == true
      assert decoded["prediction"] == %{"type" => "content", "content" => "test"}
    end

    test "encode_body includes parallel_tool_calls from provider_options" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          provider_options: [parallel_tool_calls: true]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["parallel_tool_calls"] == true
    end

    test "encode_body includes prompt_mode from provider_options" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          provider_options: [prompt_mode: :reasoning]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["prompt_mode"] == "reasoning"
    end

    test "encode_body includes metadata from provider_options" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      metadata = %{"user_id" => "abc123", "session" => "xyz789"}

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          provider_options: [metadata: metadata]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["metadata"] == %{"user_id" => "abc123", "session" => "xyz789"}
    end

    test "encode_body removes seed key (uses random_seed instead)" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          seed: 12_345,
          provider_options: [random_seed: 42]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      refute Map.has_key?(decoded, "seed")
      assert decoded["random_seed"] == 42
    end

    test "encode_body includes all new Mistral-specific options together" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")
      context = context_fixture()

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          provider_options: [
            random_seed: 42,
            safe_prompt: true,
            prediction: %{type: "content", content: "test"},
            parallel_tool_calls: false,
            prompt_mode: :reasoning,
            metadata: %{"key" => "value"}
          ]
        ]
      }

      updated_request = Mistral.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["random_seed"] == 42
      assert decoded["safe_prompt"] == true
      assert decoded["prediction"] == %{"type" => "content", "content" => "test"}
      assert decoded["parallel_tool_calls"] == false
      assert decoded["prompt_mode"] == "reasoning"
      assert decoded["metadata"] == %{"key" => "value"}
    end
  end

  describe "Mistral finish reason normalization" do
    test "decode_response normalizes model_length to length" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")

      mock_json_response = %{
        "id" => "chatcmpl-test123",
        "object" => "chat.completion",
        "created" => 1_234_567_890,
        "model" => "mistral-large-latest",
        "choices" => [
          %{
            "index" => 0,
            "message" => %{
              "role" => "assistant",
              "content" => "This response was truncated..."
            },
            "finish_reason" => "model_length"
          }
        ],
        "usage" => %{
          "prompt_tokens" => 10,
          "completion_tokens" => 8,
          "total_tokens" => 18
        }
      }

      mock_resp = %Req.Response{
        status: 200,
        body: mock_json_response
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, stream: false],
        private: %{req_llm_model: model}
      }

      {_req, resp} = Mistral.decode_response({mock_req, mock_resp})

      assert %ReqLLM.Response{} = resp.body
      assert resp.body.finish_reason == :length
    end

    test "decode_response preserves standard finish reasons" do
      {:ok, model} = ReqLLM.model("mistral:mistral-large-latest")

      for {api_reason, expected_reason} <- [
            {"stop", :stop},
            {"length", :length},
            {"tool_calls", :tool_calls}
          ] do
        mock_json_response = %{
          "id" => "chatcmpl-test123",
          "object" => "chat.completion",
          "created" => 1_234_567_890,
          "model" => "mistral-large-latest",
          "choices" => [
            %{
              "index" => 0,
              "message" => %{"role" => "assistant", "content" => "Response"},
              "finish_reason" => api_reason
            }
          ],
          "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15}
        }

        mock_resp = %Req.Response{status: 200, body: mock_json_response}
        context = context_fixture()

        mock_req = %Req.Request{
          options: [context: context, stream: false],
          private: %{req_llm_model: model}
        }

        {_req, resp} = Mistral.decode_response({mock_req, mock_resp})

        assert resp.body.finish_reason == expected_reason,
               "Expected #{inspect(expected_reason)} for API reason #{inspect(api_reason)}, got #{inspect(resp.body.finish_reason)}"
      end
    end
  end
end
