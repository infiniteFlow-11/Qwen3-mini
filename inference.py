def load_trained_model(model_path: str = "final_model.pt"):
    """Load a trained model from checkpoint"""
    print(f" Loading model from {model_path}")

    # Add ModelConfig to safe globals for PyTorch 2.6+
    from torch.serialization import add_safe_globals
    add_safe_globals([ModelConfig])

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
    except Exception as e:
        print(f"⚠️ Error loading with weights_only=True, trying with weights_only=False...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

    # Create model with same config
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {device}")

    return model, config