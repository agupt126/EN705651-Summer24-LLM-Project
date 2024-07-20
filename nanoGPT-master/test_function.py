def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)# forward the GPT model itself
    tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
    pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)

    logits = self.lm_head(x)  # raw logitsif targets isnotNone:
    # Apply sigmoid activation for binary cross-entropy loss
    probs = torch.sigmoid(logits)
    loss = F.binary_cross_entropy(probs.view(-1), targets.view(-1).float(), reduction='mean')
    else:
    # inference-time mini-optimization: only forward the lm_head on the very last position
    logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
    probs = torch.sigmoid(logits)
    loss = None
return probs, loss