import { apiClient } from './config'

export interface LLMConfigDTO {
  provider: string
  has_api_key: boolean
  api_key_masked: string
  base_url?: string | null
  model?: string | null
  timeout?: number | null
}

export interface LLMConfigUpdatePayload {
  provider: string
  api_key?: string | null
  base_url?: string | null
  model?: string | null
  timeout?: number | null
  persist_to_env?: boolean
}

export const llmConfigApi = {
  get: () => apiClient.get<LLMConfigDTO>('/system/llm-config') as Promise<LLMConfigDTO>,
  update: (payload: LLMConfigUpdatePayload) =>
    apiClient.put<LLMConfigDTO>('/system/llm-config', payload) as Promise<LLMConfigDTO>,
}
