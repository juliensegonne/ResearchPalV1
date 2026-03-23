import { Component, inject, signal, ViewChild, ElementRef, AfterViewChecked, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../api.service';
import { ChatMessage } from '../models';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './chat.html',
  styleUrl: './chat.scss',
})
export class ChatComponent implements AfterViewChecked, OnInit {
  private readonly api = inject(ApiService);

  query = signal('');
  messages = signal<ChatMessage[]>([]);
  loading = signal(false);
  expandedSources = signal<Set<number>>(new Set());

  @ViewChild('messagesEnd') private messagesEnd!: ElementRef;
  private shouldScroll = false;

  ngOnInit() {
    this.api.getHistory().subscribe({
      next: (history) => {
        if (history.length) {
          this.messages.set(history);
          this.shouldScroll = true;
        }
      },
    });
  }

  ngAfterViewChecked() {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  sendMessage() {
    const q = this.query().trim();
    if (!q || this.loading()) return;

    this.messages.update(msgs => [...msgs, { role: 'user', content: q, sources: [] }]);
    this.query.set('');
    this.loading.set(true);
    this.shouldScroll = true;

    this.api.chat(q).subscribe({
      next: (res) => {
        this.messages.update(msgs => [
          ...msgs,
          { role: 'assistant', content: res.answer, sources: res.sources },
        ]);
        this.loading.set(false);
        this.shouldScroll = true;
      },
      error: (err) => {
        const errMsg = err.error?.detail || 'Erreur de connexion au serveur.';
        this.messages.update(msgs => [
          ...msgs,
          { role: 'assistant', content: `⚠️ ${errMsg}`, sources: [] },
        ]);
        this.loading.set(false);
        this.shouldScroll = true;
      },
    });
  }

  toggleSources(index: number) {
    this.expandedSources.update(set => {
      const newSet = new Set(set);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  }

  isSourceExpanded(index: number): boolean {
    return this.expandedSources().has(index);
  }

  clearHistory() {
    this.api.clearHistory().subscribe(() => {
      this.messages.set([]);
      this.expandedSources.set(new Set());
    });
  }

  onKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  private scrollToBottom() {
    try {
      this.messagesEnd?.nativeElement?.scrollIntoView({ behavior: 'smooth' });
    } catch (_) {}
  }
}
