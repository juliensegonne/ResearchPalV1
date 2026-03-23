import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ChatMessage, ChatResponse, DocumentInfo, HealthStatus, SearchResponse } from './models';
import { environment } from '../environments/environment';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly base = environment.apiBase;

  health(): Observable<HealthStatus> {
    return this.http.get<HealthStatus>(`${this.base}/health`);
  }

  // --- Documents ---
  listDocuments(): Observable<DocumentInfo[]> {
    return this.http.get<DocumentInfo[]>(`${this.base}/documents`);
  }

  uploadFile(file: File): Observable<{ message: string; filename: string }> {
    const fd = new FormData();
    fd.append('file', file);
    return this.http.post<{ message: string; filename: string }>(`${this.base}/documents/upload`, fd);
  }

  addUrl(url: string): Observable<{ message: string; chunks: number }> {
    const fd = new FormData();
    fd.append('url', url);
    return this.http.post<{ message: string; chunks: number }>(`${this.base}/documents/add-url`, fd);
  }

  ingestDocuments(): Observable<{ message: string }> {
    return this.http.post<{ message: string }>(`${this.base}/documents/ingest`, {});
  }

  clearDatabase(): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(`${this.base}/documents/clear`);
  }

  // --- Chat ---
  chat(query: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.base}/chat`, { query });
  }

  search(query: string): Observable<SearchResponse> {
    return this.http.post<SearchResponse>(`${this.base}/search`, { query });
  }

  // --- History ---
  getHistory(): Observable<ChatMessage[]> {
    return this.http.get<ChatMessage[]>(`${this.base}/history`);
  }

  clearHistory(): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(`${this.base}/history`);
  }
}
