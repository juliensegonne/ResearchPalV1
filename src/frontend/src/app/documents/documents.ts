import { Component, inject, signal, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatePipe } from '@angular/common';
import { ApiService } from '../api.service';
import { DocumentInfo } from '../models';

@Component({
  selector: 'app-documents',
  standalone: true,
  imports: [FormsModule, DatePipe],
  templateUrl: './documents.html',
  styleUrl: './documents.scss',
})
export class DocumentsComponent implements OnInit {
  private readonly api = inject(ApiService);

  documents = signal<DocumentInfo[]>([]);
  url = signal('');
  uploading = signal(false);
  ingesting = signal(false);
  clearing = signal(false);
  statusMessage = signal('');
  statusType = signal<'success' | 'error' | ''>('');

  ngOnInit() {
    this.refreshDocuments();
  }

  refreshDocuments() {
    this.api.listDocuments().subscribe({
      next: (docs) => this.documents.set(docs),
      error: () => this.showStatus('Impossible de charger la liste des documents.', 'error'),
    });
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;

    const file = input.files[0];
    this.uploading.set(true);
    this.api.uploadFile(file).subscribe({
      next: (res) => {
        this.showStatus(res.message, 'success');
        this.uploading.set(false);
        this.refreshDocuments();
        input.value = '';
      },
      error: (err) => {
        this.showStatus(err.error?.detail || 'Erreur lors du téléversement.', 'error');
        this.uploading.set(false);
      },
    });
  }

  addUrl() {
    const u = this.url().trim();
    if (!u) return;

    this.uploading.set(true);
    this.api.addUrl(u).subscribe({
      next: (res) => {
        this.showStatus(res.message, 'success');
        this.url.set('');
        this.uploading.set(false);
        this.refreshDocuments();
      },
      error: (err) => {
        this.showStatus(err.error?.detail || 'Erreur lors de l\'ajout.', 'error');
        this.uploading.set(false);
      },
    });
  }

  ingestAll() {
    this.ingesting.set(true);
    this.api.ingestDocuments().subscribe({
      next: (res) => {
        this.showStatus(res.message, 'success');
        this.ingesting.set(false);
      },
      error: (err) => {
        this.showStatus(err.error?.detail || 'Erreur lors de l\'ingestion.', 'error');
        this.ingesting.set(false);
      },
    });
  }

  clearDatabase() {
    this.clearing.set(true);
    this.api.clearDatabase().subscribe({
      next: (res) => {
        this.showStatus(res.message, 'success');
        this.clearing.set(false);
        this.refreshDocuments();
      },
      error: (err) => {
        this.showStatus(err.error?.detail || 'Erreur lors de la suppression.', 'error');
        this.clearing.set(false);
      },
    });
  }

  formatSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  private showStatus(msg: string, type: 'success' | 'error') {
    this.statusMessage.set(msg);
    this.statusType.set(type);
    setTimeout(() => {
      this.statusMessage.set('');
      this.statusType.set('');
    }, 5000);
  }
}
