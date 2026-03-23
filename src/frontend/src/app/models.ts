export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources: string[];
}

export interface SearchResult {
  rank: number;
  text: string;
}

export interface SearchResponse {
  query: string;
  strategy: string;
  results: SearchResult[];
}

export interface ChatResponse {
  answer: string;
  sources: string[];
  history: ChatMessage[];
}

export interface DocumentInfo {
  name: string;
  type: 'file' | 'url';
  size: number;
  modified: string;
}

export interface HealthStatus {
  status: string;
  has_database: boolean;
  document_count: number;
}
