def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():  # Disable gradient computation for evaluation (saves memory and computation)
        for i, (x, y) in enumerate(val_loader):
            # Stop evaluation after specified number of steps to limit eval time
            if i >= config.eval_steps:
                break

            # Move input sequences (x) and target sequences (y) to GPU/device
            x, y = x.to(device), y.to(device)

            # Use automatic mixed precision if enabled (faster training with minimal accuracy loss)
            with autocast(enabled=config.use_amp):
                # Forward pass: get model predictions (logits) for input sequence
                logits = model(x)

                # Calculate cross-entropy loss between predictions and targets
                # Reshape to (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
                # for proper cross-entropy computation across all token positions
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            # Accumulate total loss weighted by number of tokens in this batch
            total_loss += loss.item() * y.numel()
            # Keep track of total number of tokens processed
            total_tokens += y.numel()

            # Get predicted token IDs by taking argmax over vocabulary dimension
            predictions = logits.argmax(dim=-1)
            # Count correct predictions for accuracy calculation
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}
