import { Routes } from '@angular/router';
import { ChatComponent } from './chat/chat';
import { DocumentsComponent } from './documents/documents';

export const routes: Routes = [
  { path: '', component: ChatComponent },
  { path: 'documents', component: DocumentsComponent },
  { path: '**', redirectTo: '' },
];
