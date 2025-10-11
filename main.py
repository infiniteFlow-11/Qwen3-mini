if __name__ == "__main__":
    # Check if we have a trained model
    import os

    if os.path.exists("final_model.pt"):
        print("🎉 Found trained model! Running demo...")
        demo_inference("final_model.pt")

        # Optionally run interactive session
        response = input("\n🤖 Would you like to try interactive inference? (y/n): ")
        if response.lower() in ['y', 'yes']:
            interactive_inference("final_model.pt")
    else:
        print("⚠️ No trained model found. Please run the training cells first.")
        print("💡 Look for 'final_model.pt' or 'best_model.pt' in your directory.")