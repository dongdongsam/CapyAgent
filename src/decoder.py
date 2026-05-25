import torch
import torch.nn as nn
import string

class CapyGenerativeDecoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=256):
        super().__init__()
        # Vocabulary: lowercase letters + space + common punctuation
        self.chars = " " + string.ascii_lowercase + ".,!?'\"-()[]"
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}

        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        self.char_embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.context_projection = nn.Linear(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(embedding_dim, self.vocab_size)
        
        # Proper initialization to avoid gibberish
        nn.init.xavier_uniform_(self.context_projection.weight)
        nn.init.xavier_uniform_(self.out.weight)
        
    def forward_train(self, motor_states, target_indices):
        """
        motor_states: (Num_Motor, Dim)
        target_indices: (Batch=1, Seq_Len)
        """
        device = motor_states.device
        seq_len = target_indices.size(1)
        
        context = motor_states.view(1, -1)
        hidden_init = torch.tanh(self.context_projection(context))
        hidden = hidden_init.unsqueeze(0).repeat(2, 1, 1)
        
        # Teacher forcing: provide the actual target character at each step
        embedded = self.char_embedding(target_indices) # (1, Seq_Len, Dim)
        output, _ = self.gru(embedded, hidden)
        logits = self.out(output) # (1, Seq_Len, vocab_size)
        
        return logits

    def generate(self, motor_states, max_len=50, temperature=0.7):
        # The previous 'forward' logic moved here for clarity
        device = motor_states.device
        context = motor_states.view(1, -1)
        hidden_init = torch.tanh(self.context_projection(context))
        hidden = hidden_init.unsqueeze(0).repeat(2, 1, 1)
        
        curr_char = torch.tensor([[self.char_to_idx[" "]]]).to(device)
        generated_text = ""
        
        for _ in range(max_len):
            embedded = self.char_embedding(curr_char)
            output, hidden = self.gru(embedded, hidden)
            logits = self.out(output.squeeze(1))
            
            probs = torch.softmax(logits / temperature, dim=1)
            char_idx = torch.multinomial(probs, 1).item()
            
            char = self.idx_to_char[char_idx]
            if char == " " and generated_text.endswith(" "): continue
            
            generated_text += char
            curr_char = torch.tensor([[char_idx]]).to(device)
            if generated_text.endswith("..."): break
                
        return generated_text.strip() + "..."
