def plot_training_history(history, version, experiment_dir, run):
    """
    Rysuje wykresy treningu (accuracy i loss), zapisuje je lokalnie
    oraz jako artefakt w Neptune.
    """
    plt.figure(figsize=(10, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title(f'Model {version} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title(f'Model {version} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plot_path = f"{experiment_dir}/plots/model_{version}_training_results.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    # Wysyłamy wykres do Neptune jako artefakt
    run[f"model_{version}/plots/training_results"].upload(plot_path)