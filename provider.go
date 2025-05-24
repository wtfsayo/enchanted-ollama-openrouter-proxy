package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"strings"
	"time"
	"net/http"

	"github.com/sashabaranov/go-openai"
)

type OpenrouterProvider struct {
	client     *openai.Client
	modelNames []string // Shared storage for model names
}

func NewOpenrouterProvider(apiKey string) *OpenrouterProvider {
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://openrouter.ai/api/v1/"
	
	// Create HTTP client with custom headers for OpenRouter
	httpClient := &http.Client{
		Transport: &headerTransport{
			Transport: http.DefaultTransport,
			Headers: map[string]string{
				"HTTP-Referer": "http://localhost:11434",
				"X-Title":      "Ollama Proxy",
			},
		},
	}
	config.HTTPClient = httpClient
	
	return &OpenrouterProvider{
		client:     openai.NewClientWithConfig(config),
		modelNames: []string{},
	}
}

// headerTransport adds custom headers to HTTP requests
type headerTransport struct {
	Transport http.RoundTripper
	Headers   map[string]string
}

func (h *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	for key, value := range h.Headers {
		req.Header.Set(key, value)
	}
	return h.Transport.RoundTrip(req)
}

func (o *OpenrouterProvider) Chat(messages []openai.ChatCompletionMessage, modelName string) (openai.ChatCompletionResponse, error) {
	// Create a chat completion request
	req := openai.ChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   false,
	}

	// Call the OpenAI API to get a complete response
	resp, err := o.client.CreateChatCompletion(context.Background(), req)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}

	// Return the complete response
	return resp, nil
}

func (o *OpenrouterProvider) ChatStream(messages []openai.ChatCompletionMessage, modelName string) (*openai.ChatCompletionStream, error) {
	// Create a chat completion request
	req := openai.ChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   true,
	}

	// Call the OpenAI API to get a streaming response
	stream, err := o.client.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		return nil, err
	}

	// Return the stream for further processing
	return stream, nil
}

type ModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type Model struct {
	Name       string       `json:"name"`
	Model      string       `json:"model,omitempty"`
	ModifiedAt string       `json:"modified_at,omitempty"`
	Size       int64        `json:"size,omitempty"`
	Digest     string       `json:"digest,omitempty"`
	Details    ModelDetails `json:"details,omitempty"`
	ContextLength int64 `json:"-"`
}

func (o *OpenrouterProvider) GetModels() ([]Model, error) {
	currentTime := time.Now().Format(time.RFC3339)

	// Fetch models from the OpenAI API
	modelsResponse, err := o.client.ListModels(context.Background())
	if err != nil {
		return nil, err
	}

	// Clear shared model storage
	o.modelNames = []string{}

	var models []Model
	for _, apiModel := range modelsResponse.Models {
		// Split model name
		parts := strings.Split(apiModel.ID, "/")
		name := parts[len(parts)-1]

		// Store name in shared storage
		o.modelNames = append(o.modelNames, apiModel.ID)

		// Estimate parameter size based on model name patterns
		parameterSize := "7B"
		nameToCheck := strings.ToLower(apiModel.ID + " " + name)
		
		if strings.Contains(nameToCheck, "70b") {
			parameterSize = "70B"
		} else if strings.Contains(nameToCheck, "13b") {
			parameterSize = "13B"
		} else if strings.Contains(nameToCheck, "3b") {
			parameterSize = "3B"
		} else if strings.Contains(nameToCheck, "1b") {
			parameterSize = "1B"
		}

		// Determine architecture based on model name
		family := "transformer"
		if strings.Contains(nameToCheck, "llama") {
			family = "llama"
		} else if strings.Contains(nameToCheck, "mistral") {
			family = "mistral"
		} else if strings.Contains(nameToCheck, "gemma") {
			family = "gemma"
		} else if strings.Contains(nameToCheck, "claude") {
			family = "claude"
		} else if strings.Contains(nameToCheck, "gpt") {
			family = "gpt"
		}

		// Generate a unique digest based on model name
		digest := fmt.Sprintf("%x", sha256.Sum256([]byte(apiModel.ID)))

		// Create model struct
		model := Model{
			Name:       name,
			Model:      name,
			ModifiedAt: currentTime,
			Size:       270898672, // Realistic size for GGUF models
			Digest:     digest,
			Details: ModelDetails{
				ParentModel:       "",
				Format:            "gguf",
				Family:            family,
				Families:          []string{family},
				ParameterSize:     parameterSize,
				QuantizationLevel: "Q4_K_M",
			},
			ContextLength: 200000,
		}
		models = append(models, model)
	}

	return models, nil
}

func (o *OpenrouterProvider) GetModelDetails(modelName string) (map[string]interface{}, error) {
	// Get the full model name first
	fullModelName, err := o.GetFullModelName(modelName)
	if err != nil {
		return nil, fmt.Errorf("model not found: %s", modelName)
	}

	// Try to get model info from OpenRouter
	models, err := o.GetModels()
	if err != nil {
		return nil, fmt.Errorf("failed to fetch model details: %w", err)
	}

	// Find the specific model - try both full name and short name
	var modelInfo *Model
	for _, model := range models {
		// Try matching against the short name (what's displayed in /api/tags)
		if model.Name == modelName || model.Name == fullModelName {
			modelInfo = &model
			break
		}
	}

	// If not found by short name, try finding by full model ID
	if modelInfo == nil {
		for i, fullID := range o.modelNames {
			if fullID == fullModelName || fullID == modelName {
				// Get the corresponding model from the models array
				if i < len(models) {
					modelInfo = &models[i]
					break
				}
			}
		}
	}

	if modelInfo == nil {
		return nil, fmt.Errorf("model not found: %s", modelName)
	}

	// Create Ollama-compatible response
	currentTime := time.Now().Format(time.RFC3339)
	
	// Estimate parameter size based on model name patterns
	parameterSize := "7B"
	parameterCount := int64(7_000_000_000)
	
	// Check both the short name and full name for size indicators
	nameToCheck := strings.ToLower(fullModelName + " " + modelInfo.Name)
	
	if strings.Contains(nameToCheck, "70b") {
		parameterSize = "70B"
		parameterCount = 70_000_000_000
	} else if strings.Contains(nameToCheck, "13b") {
		parameterSize = "13B"
		parameterCount = 13_000_000_000
	} else if strings.Contains(nameToCheck, "3b") {
		parameterSize = "3B"
		parameterCount = 3_000_000_000
	} else if strings.Contains(nameToCheck, "1b") {
		parameterSize = "1B"
		parameterCount = 1_000_000_000
	}

	// Determine architecture based on model name
	architecture := "transformer"
	if strings.Contains(nameToCheck, "llama") {
		architecture = "llama"
	} else if strings.Contains(nameToCheck, "mistral") {
		architecture = "mistral"
	} else if strings.Contains(nameToCheck, "gemma") {
		architecture = "gemma"
	} else if strings.Contains(nameToCheck, "claude") {
		architecture = "claude"
	} else if strings.Contains(nameToCheck, "gpt") {
		architecture = "gpt"
	}

	return map[string]interface{}{
		"modelfile": fmt.Sprintf("# Modelfile generated for %s\nFROM %s", modelName, fullModelName),
		"parameters": "",
		"template": "{{ if .System }}{{ .System }}\n{{ end }}{{ if .Prompt }}{{ .Prompt }}{{ end }}",
		"system": "",
		"details": map[string]interface{}{
			"parent_model":       "",
			"format":             "gguf",
			"family":             architecture,
			"families":           []string{architecture},
			"parameter_size":     parameterSize,
			"quantization_level": "Q4_K_M",
		},
		"model_info": map[string]interface{}{
			"general.architecture":    architecture,
			"general.file_type":       1,
			"general.parameter_count": parameterCount,
			"general.quantization_version": 2,
			"llama.context_length":    modelInfo.ContextLength,
			"llama.embedding_length":  4096,
			"llama.block_count":       32,
		},
		"modified_at": currentTime,
	}, nil
}

func (o *OpenrouterProvider) GetFullModelName(alias string) (string, error) {
	// If modelNames is empty or not populated yet, try to get models first
	if len(o.modelNames) == 0 {
		_, err := o.GetModels()
		if err != nil {
			return "", fmt.Errorf("failed to get models: %w", err)
		}
	}

	// First try exact match
	for _, fullName := range o.modelNames {
		if fullName == alias {
			return fullName, nil
		}
	}

	// Then try suffix match
	for _, fullName := range o.modelNames {
		if strings.HasSuffix(fullName, alias) {
			return fullName, nil
		}
	}

	// If no match found, just use the alias as is
	// This allows direct use of model names that might not be in the list
	return alias, nil
}
