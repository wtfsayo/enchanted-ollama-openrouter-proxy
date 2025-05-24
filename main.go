package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	openai "github.com/sashabaranov/go-openai"
)

var modelFilter map[string]struct{}

func loadModelFilter(path string) (map[string]struct{}, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	filter := make(map[string]struct{})

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			filter[line] = struct{}{}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return filter, nil
}

func main() {
	r := gin.Default()
	
	// Add CORS middleware
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, HEAD")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	})
	
	// Load the API key from environment variables or command-line arguments.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		if len(os.Args) > 1 {
			apiKey = os.Args[1]
		} else {
			slog.Error("OPENAI_API_KEY environment variable or command-line argument not set.")
			return
		}
	}

	provider := NewOpenrouterProvider(apiKey)

	filter, err := loadModelFilter("models-filter")
	if err != nil {
		if os.IsNotExist(err) {
			slog.Info("models-filter file not found. Skipping model filtering.")
			modelFilter = make(map[string]struct{})
		} else {
			slog.Error("Error loading models filter", "Error", err)
			return
		}
	} else {
		modelFilter = filter
		slog.Info("Loaded models from filter:")
		for model := range modelFilter {
			slog.Info(" - " + model)
		}
	}

	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Ollama is running")
	})
	r.HEAD("/", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.GET("/api/tags", func(c *gin.Context) {
		models, err := provider.GetModels()
		if err != nil {
			slog.Error("Error getting models", "Error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		filter := modelFilter
		// Construct a new array of model objects with extra fields
		newModels := make([]map[string]interface{}, 0, len(models))
		for _, m := range models {
			// Если фильтр пустой, значит пропускаем проверку и берём все модели
			if len(filter) > 0 {
				if _, ok := filter[m.Model]; !ok {
					continue
				}
			}
			newModels = append(newModels, map[string]interface{}{
				"name":        m.Name,
				"model":       m.Model,
				"modified_at": m.ModifiedAt,
				"size":        270898672,
				"digest":      "9077fe9d2ae1a4a41a868836b56b8163731a8fe16621397028c2c76f838c6907",
				"details":     m.Details,
			})
		}

		c.JSON(http.StatusOK, gin.H{"models": newModels})
	})
	r.HEAD("/api/tags", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.POST("/api/show", func(c *gin.Context) {
		var request map[string]string
		if err := c.BindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		modelName := request["model"] // Fixed: was "name", should be "model"
		if modelName == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model name is required"})
			return
		}

		details, err := provider.GetModelDetails(modelName)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, details)
	})
	r.HEAD("/api/show", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.POST("/api/generate", func(c *gin.Context) {
		var request struct {
			Model    string                 `json:"model"`
			Prompt   string                 `json:"prompt"`
			Stream   *bool                  `json:"stream"`
			Format   interface{}            `json:"format"`
			Options  map[string]interface{} `json:"options"`
			System   string                 `json:"system"`
			Template string                 `json:"template"`
			Raw      bool                   `json:"raw"`
			Context  []int                  `json:"context"`
		}

		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		// Convert prompt to messages format
		messages := []openai.ChatCompletionMessage{}
		if request.System != "" {
			messages = append(messages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleSystem,
				Content: request.System,
			})
		}
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: request.Prompt,
		})

		// Get full model name
		fullModelName, err := provider.GetFullModelName(request.Model)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Model not found: " + err.Error()})
			return
		}

		// Determine streaming (default true for /api/generate)
		streamRequested := true
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		if !streamRequested {
			// Non-streaming response
			response, err := provider.Chat(messages, fullModelName)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			responseContent := ""
			if len(response.Choices) > 0 {
				responseContent = response.Choices[0].Message.Content
			}

			c.JSON(http.StatusOK, gin.H{
				"model":              fullModelName,
				"created_at":         time.Now().Format(time.RFC3339),
				"response":           responseContent,
				"done":               true,
				"context":            []int{1, 2, 3}, // Stub context
				"total_duration":     0,
				"load_duration":      0,
				"prompt_eval_count":  0,
				"prompt_eval_duration": 0,
				"eval_count":         0,
				"eval_duration":      0,
			})
			return
		}

		// Streaming response
		stream, err := provider.ChatStream(messages, fullModelName)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer stream.Close()

		c.Header("Content-Type", "application/x-ndjson")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")

		w := c.Writer
		flusher, ok := w.(http.Flusher)
		if !ok {
			slog.Error("Expected http.ResponseWriter to be an http.Flusher")
			return
		}

		var lastFinishReason string

		for {
			response, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				slog.Error("Backend stream error", "Error", err)
				errorMsg := map[string]string{"error": "Stream error: " + err.Error()}
				errorJson, _ := json.Marshal(errorMsg)
				fmt.Fprintf(w, "%s\n", string(errorJson))
				flusher.Flush()
				return
			}

			if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
				lastFinishReason = string(response.Choices[0].FinishReason)
			}

			responseJSON := map[string]interface{}{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"response":   response.Choices[0].Delta.Content,
				"done":       false,
			}

			jsonData, err := json.Marshal(responseJSON)
			if err != nil {
				slog.Error("Error marshaling response JSON", "Error", err)
				return
			}

			fmt.Fprintf(w, "%s\n", string(jsonData))
			flusher.Flush()
		}

		// Final response
		if lastFinishReason == "" {
			lastFinishReason = "stop"
		}

		finalResponse := map[string]interface{}{
			"model":              fullModelName,
			"created_at":         time.Now().Format(time.RFC3339),
			"response":           "",
			"done":               true,
			"context":            []int{1, 2, 3}, // Stub context
			"total_duration":     0,
			"load_duration":      0,
			"prompt_eval_count":  0,
			"prompt_eval_duration": 0,
			"eval_count":         0,
			"eval_duration":      0,
		}

		finalJsonData, err := json.Marshal(finalResponse)
		if err != nil {
			slog.Error("Error marshaling final response JSON", "Error", err)
			return
		}

		fmt.Fprintf(w, "%s\n", string(finalJsonData))
		flusher.Flush()
	})
	r.HEAD("/api/generate", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.GET("/api/version", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"version": "0.1.0",
		})
	})
	r.HEAD("/api/version", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.GET("/api/ps", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"models": []interface{}{},
		})
	})
	r.HEAD("/api/ps", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.POST("/api/pull", func(c *gin.Context) {
		var request map[string]interface{}
		if err := c.BindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		modelName, ok := request["model"].(string)
		if !ok || modelName == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model name is required"})
			return
		}

		// Check if streaming is requested (default true for pull)
		streamRequested := true
		if stream, exists := request["stream"]; exists {
			if streamBool, ok := stream.(bool); ok {
				streamRequested = streamBool
			}
		}

		// Validate model exists in OpenRouter
		_, err := provider.GetFullModelName(modelName)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "Model not found: " + err.Error()})
			return
		}

		if !streamRequested {
			// Non-streaming response
			c.JSON(http.StatusOK, gin.H{
				"status": "success",
			})
			return
		}

		// Streaming response - simulate download progress
		c.Header("Content-Type", "application/x-ndjson")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")

		w := c.Writer
		flusher, ok := w.(http.Flusher)
		if !ok {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Streaming not supported"})
			return
		}

		// Simulate realistic pull progress for a large model
		// Step 1: Pulling manifest
		manifestStep := map[string]interface{}{"status": "pulling manifest"}
		jsonData, _ := json.Marshal(manifestStep)
		fmt.Fprintf(w, "%s\n", string(jsonData))
		flusher.Flush()
		time.Sleep(2 * time.Second)

		// Step 2: Multiple layers with realistic sizes (simulating a 7B model ~4GB)
		layers := []struct {
			digest string
			size   int64
		}{
			{"sha256:6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa", 1073741824}, // 1GB
			{"sha256:4fa551d4f938f68b8c1e6aedf7c5d35c5c5f5c5c5c5c5c5c5c5c5c5c5c5c5c5", 2147483648}, // 2GB  
			{"sha256:8ab4849b038cf2ad1f725cd8ad45c7b4f3c3c3c3c3c3c3c3c3c3c3c3c3c3c3", 536870912},  // 512MB
			{"sha256:577073ffcc6ce95b3ac4c4c4c4c4c4c4c4c4c4c4c4c4c4c4c4c4c4c4c4c4c4", 268435456},  // 256MB
			{"sha256:3f8eb4da87fa62c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2", 134217728},  // 128MB
		}

		for _, layer := range layers {
			// Simulate downloading each layer with progress updates
			chunkSize := layer.size / 20 // 20 progress updates per layer
			for completed := int64(0); completed < layer.size; completed += chunkSize {
				if completed+chunkSize > layer.size {
					completed = layer.size
				}
				
				progressStep := map[string]interface{}{
					"status":    "downloading",
					"digest":    layer.digest,
					"total":     layer.size,
					"completed": completed,
				}
				jsonData, _ := json.Marshal(progressStep)
				fmt.Fprintf(w, "%s\n", string(jsonData))
				flusher.Flush()
				
				// Slower download simulation - 500ms per chunk
				time.Sleep(500 * time.Millisecond)
			}
		}

		// Step 3: Verifying digests (slower)
		verifyStep := map[string]interface{}{"status": "verifying sha256 digest"}
		jsonData, _ = json.Marshal(verifyStep)
		fmt.Fprintf(w, "%s\n", string(jsonData))
		flusher.Flush()
		time.Sleep(3 * time.Second)

		// Step 4: Writing manifest
		writeStep := map[string]interface{}{"status": "writing manifest"}
		jsonData, _ = json.Marshal(writeStep)
		fmt.Fprintf(w, "%s\n", string(jsonData))
		flusher.Flush()
		time.Sleep(2 * time.Second)

		// Step 5: Removing unused layers
		removeStep := map[string]interface{}{"status": "removing any unused layers"}
		jsonData, _ = json.Marshal(removeStep)
		fmt.Fprintf(w, "%s\n", string(jsonData))
		flusher.Flush()
		time.Sleep(1 * time.Second)

		// Step 6: Success
		successStep := map[string]interface{}{"status": "success"}
		jsonData, _ = json.Marshal(successStep)
		fmt.Fprintf(w, "%s\n", string(jsonData))
		flusher.Flush()
	})
	r.HEAD("/api/pull", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.POST("/api/copy", func(c *gin.Context) {
		var request map[string]string
		if err := c.BindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		if request["source"] == "" || request["destination"] == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Source and destination are required"})
			return
		}

		// Stub response - just return success
		c.Status(http.StatusOK)
	})
	r.HEAD("/api/copy", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.DELETE("/api/delete", func(c *gin.Context) {
		var request map[string]string
		if err := c.BindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		if request["model"] == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model name is required"})
			return
		}

		// Stub response - just return success
		c.Status(http.StatusOK)
	})
	r.HEAD("/api/delete", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.POST("/api/chat", func(c *gin.Context) {
		var request struct {
			Model    string                         `json:"model"`
			Messages []openai.ChatCompletionMessage `json:"messages"`
			Stream   *bool                          `json:"stream"`
			Format   interface{}                    `json:"format"`
			Options  map[string]interface{}         `json:"options"`
			Tools    []interface{}                  `json:"tools"`
		}

		// Parse the JSON request
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		// Determine streaming (default true for /api/chat)
		streamRequested := true
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		// Handle empty messages array (load model)
		if len(request.Messages) == 0 {
			if streamRequested {
				c.JSON(http.StatusOK, gin.H{
					"model":      request.Model,
					"created_at": time.Now().Format(time.RFC3339),
					"message": gin.H{
						"role":    "assistant",
						"content": "",
					},
					"done":        true,
					"done_reason": "load",
				})
			} else {
				c.JSON(http.StatusOK, gin.H{
					"model":      request.Model,
					"created_at": time.Now().Format(time.RFC3339),
					"message": gin.H{
						"role":    "assistant",
						"content": "",
					},
					"done":        true,
					"done_reason": "load",
				})
			}
			return
		}

		// Get full model name
		fullModelName, err := provider.GetFullModelName(request.Model)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Model not found: " + err.Error()})
			return
		}

		// Handle non-streaming response
		if !streamRequested {
			response, err := provider.Chat(request.Messages, fullModelName)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			responseContent := ""
			if len(response.Choices) > 0 {
				responseContent = response.Choices[0].Message.Content
			}

			c.JSON(http.StatusOK, gin.H{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"message": gin.H{
					"role":    "assistant",
					"content": responseContent,
				},
				"done":               true,
				"total_duration":     0,
				"load_duration":      0,
				"prompt_eval_count":  0,
				"prompt_eval_duration": 0,
				"eval_count":         0,
				"eval_duration":      0,
			})
			return
		}

		slog.Info("Requested model", "model", request.Model)
		fullModelName, err = provider.GetFullModelName(request.Model)
		if err != nil {
			slog.Error("Error getting full model name", "Error", err, "model", request.Model)
			// Ollama возвращает 404 на неправильное имя модели
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}
		slog.Info("Using model", "fullModelName", fullModelName)

		// Call ChatStream to get the stream
		stream, err := provider.ChatStream(request.Messages, fullModelName)
		if err != nil {
			slog.Error("Failed to create stream", "Error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer stream.Close() // Ensure stream closure

		// --- ИСПРАВЛЕНИЯ для NDJSON (Ollama-style) ---

		// Set headers CORRECTLY for Newline Delimited JSON
		c.Writer.Header().Set("Content-Type", "application/x-ndjson") // <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
		c.Writer.Header().Set("Cache-Control", "no-cache")
		c.Writer.Header().Set("Connection", "keep-alive")
		// Transfer-Encoding: chunked устанавливается Gin автоматически

		w := c.Writer // Получаем ResponseWriter
		flusher, ok := w.(http.Flusher)
		if !ok {
			slog.Error("Expected http.ResponseWriter to be an http.Flusher")
			// Отправить ошибку клиенту уже сложно, т.к. заголовки могли уйти
			return
		}

		var lastFinishReason string

		// Stream responses back to the client
		for {
			response, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				// End of stream from the backend provider
				break
			}
			if err != nil {
				slog.Error("Backend stream error", "Error", err)
				// Попытка отправить ошибку в формате NDJSON
				// Ollama обычно просто обрывает соединение или шлет 500 перед этим
				errorMsg := map[string]string{"error": "Stream error: " + err.Error()}
				errorJson, _ := json.Marshal(errorMsg)
				fmt.Fprintf(w, "%s\n", string(errorJson)) // Отправляем ошибку + \n
				flusher.Flush()
				return
			}

			// Сохраняем причину остановки, если она есть в чанке
			if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
				lastFinishReason = string(response.Choices[0].FinishReason)
			}

			// Build JSON response structure for intermediate chunks (Ollama chat format)
			responseJSON := map[string]interface{}{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"message": map[string]string{
					"role":    "assistant",
					"content": response.Choices[0].Delta.Content, // Может быть ""
				},
				"done": false, // Всегда false для промежуточных чанков
			}

			// Marshal JSON
			jsonData, err := json.Marshal(responseJSON)
			if err != nil {
				slog.Error("Error marshaling intermediate response JSON", "Error", err)
				return // Прерываем, так как не можем отправить данные
			}

			// Send JSON object followed by a newline
			fmt.Fprintf(w, "%s\n", string(jsonData)) // <--- ИЗМЕНЕНО: Формат NDJSON (JSON + \n)

			// Flush data to send it immediately
			flusher.Flush()
		}

		// --- Отправка финального сообщения (done: true) в стиле Ollama ---

		// Определяем причину остановки (если бэкенд не дал, ставим 'stop')
		// Ollama использует 'stop', 'length', 'content_filter', 'tool_calls'
		if lastFinishReason == "" {
			lastFinishReason = "stop"
		}

		// ВАЖНО: Замените nil на 0 для числовых полей статистики
		finalResponse := map[string]interface{}{
			"model":             fullModelName,
			"created_at":        time.Now().Format(time.RFC3339),
			"message": map[string]string{
				"role":    "assistant",
				"content": "", // Пустой контент для финального сообщения
			},
			"done":              true,
			"finish_reason":     lastFinishReason, // Необязательно для /api/chat Ollama, но не вредит
			"total_duration":    0,
			"load_duration":     0,
			"prompt_eval_count":  0, // <--- ИЗМЕНЕНО: nil заменен на 0
			"eval_count":        0, // <--- ИЗМЕНЕНО: nil заменен на 0
			"eval_duration":     0,
		}

		finalJsonData, err := json.Marshal(finalResponse)
		if err != nil {
			slog.Error("Error marshaling final response JSON", "Error", err)
			return
		}

		// Отправляем финальный JSON-объект + newline
		fmt.Fprintf(w, "%s\n", string(finalJsonData)) // <--- ИЗМЕНЕНО: Формат NDJSON
		flusher.Flush()

		// ВАЖНО: Для NDJSON НЕТ 'data: [DONE]' маркера.
		// Клиент понимает конец потока по получению объекта с "done": true
		// и/или по закрытию соединения сервером (что Gin сделает автоматически после выхода из хендлера).

		// --- Конец исправлений ---
	})
	r.HEAD("/api/chat", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	r.Run(":11434")
}
