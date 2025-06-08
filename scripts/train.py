import glob
import os
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


# Defines one full training epoch.
# model: our AutoVC model
# emotion_classifier: auxiliary classifier used for the emotion loss
# dataloader: yields batches of (source, target, emotion)
# optimizer: for AutoVC
# optimizer_cls: for the emotion classifier
# device: usually "cuda" or "cpu"
# lambda_ce: weight for the emotion classification loss
# lambda_spk: weight for the speaker loss

def extract_epoch_num(filename):
    try:
        return int(filename.split("epoch")[-1].split(".")[0])
    except:
        return -1  # fallback

def train_one_epoch(model, emotion_classifier, dataloader, optimizer, optimizer_cls,
                    device, lambda_ce=0.5, lambda_spk=0.5):
    # Puts both models into training mode.
    model.train()
    emotion_classifier.train()
    # Initialize accumulators to track average losses across all batches.
    total_recon_loss = 0
    total_emotion_loss = 0
    total_speaker_cs_loss = 0
    total_speaker_ce_loss = 0
    total_speaker_loss = 0

    for source_mel, target_mel, emotion_label, target_speaker_id in tqdm(dataloader):
        source_mel = source_mel.to(device)  # (B, T, 80)
        target_mel = target_mel.to(device)  # (B, T, 80)
        emotion_label = emotion_label.to(device)  # (B,)
        target_speaker_id = target_speaker_id.to(device)  # (B,)

        # === Forward pass ===
        # Predict the mel-spectrogram from source + speaker + emotion
        mel_pred, spk_logits = model(source_mel, target_mel, emotion_label)  # (B, T, 80)

        # === Loss: Reconstruction ===
        recon_loss = F.l1_loss(mel_pred, target_mel)

        # === Loss: Emotion classification ===
        logits = emotion_classifier(mel_pred)  # (B, num_emotions)
        # Compares it to the ground truth emotion label using cross-entropy loss
        ce_loss = F.cross_entropy(logits, emotion_label)

        # === Speaker Consistency Loss ===
        with torch.no_grad():
            target_speaker_emb = model.speaker_encoder(target_mel)  # (B, D)
        pred_speaker_emb = model.speaker_encoder(mel_pred.detach())  # (B, D)

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(pred_speaker_emb, target_speaker_emb, dim=-1)  # (B,)
        speaker_cs_loss = 1.0 - cos_sim.mean()  # (scalar)
        speaker_ce_loss = F.cross_entropy(spk_logits, target_speaker_id)
        speaker_loss = speaker_cs_loss + speaker_ce_loss

        # === Combine losses ===
        total_loss = recon_loss + lambda_ce * ce_loss + lambda_spk * speaker_loss

        # === Backprop ===
        optimizer.zero_grad()
        optimizer_cls.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer_cls.step()

        total_recon_loss += recon_loss.item()
        total_emotion_loss += ce_loss.item()
        total_speaker_cs_loss += speaker_cs_loss.item()
        total_speaker_ce_loss += speaker_ce_loss.item()
        total_speaker_loss += speaker_cs_loss.item() + speaker_ce_loss.item()

    avg_recon = total_recon_loss / max(len(dataloader), 1)
    avg_ce = total_emotion_loss / max(len(dataloader), 1)
    avg_cs_spk = total_speaker_cs_loss / max(len(dataloader), 1)
    avg_ce_spk = total_speaker_ce_loss / max(len(dataloader), 1)
    avg_spk = total_speaker_loss / max(len(dataloader), 1)

    print(
        f"Avg Recon: {avg_recon:.4f} | Avg CE: {avg_ce:.4f} | Avg CS_Spk: {avg_cs_spk:.4f}"
        f" | Avg CE_Spk: {avg_ce_spk:.4f} | Avg Spk: {avg_spk:.4f}")

    return avg_recon, avg_ce, avg_cs_spk, avg_ce_spk, avg_spk


def train(model, emotion_classifier, dataloader,
          optimizer, optimizer_cls, device,
          num_epochs=100, lambda_ce=0.5, lambda_spk=0.5,
          checkpoint_dir="checkpoints"):
    """
    Trains the model over multiple epochs.
    Args:
        model: AutoVC model
        emotion_classifier: auxiliary emotion classifier
        dataloader: PyTorch DataLoader
        optimizer: optimizer for the AutoVC model
        optimizer_cls: optimizer for the classifier
        device: "cuda" or "cpu"
        num_epochs: number of epochs to train
        lambda_ce: weight for emotion classification loss
        lambda_spk: weight for the speaker loss
        checkpoint_dir: directory to save model checkpoints
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path = os.path.join(checkpoint_dir, "train_history.pt")

    if os.path.exists(history_path):
        print(f"Loading existing training history from {history_path}")
        history = torch.load(history_path)
    else:
        history = {"recon": [], "emotion": [], "speaker_ce": [], "speaker_cos": [], "speaker_total": []}

    # === Try to resume training ===
    cls_ckpts = sorted(
    glob.glob(os.path.join(checkpoint_dir, "emotion_cls_epoch*.pt")),
    key=extract_epoch_num
    )
    autovc_ckpts = sorted(
    glob.glob(os.path.join(checkpoint_dir, "autovc_epoch*.pt")),
    key=extract_epoch_num
    )

    start_epoch = 1
    if autovc_ckpts and cls_ckpts:
        last_autovc = autovc_ckpts[-1]
        last_cls = cls_ckpts[-1]
        print(f"Resuming from checkpoint: {last_autovc} and {last_cls}")
        model.load_state_dict(torch.load(last_autovc, map_location=device))
        emotion_classifier.load_state_dict(torch.load(last_cls, map_location=device))
        # Extract epoch number from filename
        start_epoch = int(last_autovc.split("epoch")[-1].split(".")[0]) + 1

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch}/{start_epoch + num_epochs - 1}")

        avg_recon, avg_ce, avg_cs_spk, avg_ce_spk, avg_spk = train_one_epoch(
            model, emotion_classifier, dataloader,
            optimizer, optimizer_cls, device,
            lambda_ce, lambda_spk
        )

        print(f"Recon: {avg_recon:.4f} | Emotion: {avg_ce:.4f} | Speaker: {avg_spk:.4f}")

        # Save model checkpoints
        autovc_path = os.path.join(checkpoint_dir, f"autovc_epoch{epoch}.pt")
        emotion_cls_path = os.path.join(checkpoint_dir, f"emotion_cls_epoch{epoch}.pt")
        combined_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")

        torch.save(model.state_dict(), autovc_path)
        torch.save(emotion_classifier.state_dict(), emotion_cls_path)

        # Save combined checkpoint with mappings
        torch.save({
            "model_state": model.state_dict(),
            "emotion_classifier_state": emotion_classifier.state_dict(),
            "speaker2idx": dataloader.dataset.speaker2idx,
            "emo2idx": dataloader.dataset.emo2idx,
        }, combined_path)

        # Store loss history
        history["recon"].append(avg_recon)
        history["emotion"].append(avg_ce)
        history["speaker_total"].append(avg_spk)
        history["speaker_ce"].append(avg_ce_spk)
        history["speaker_cos"].append(avg_cs_spk)

    torch.save(history, os.path.join(checkpoint_dir, "train_history.pt"))

    # === Plot training losses ===
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 5, 1)
    plt.plot(history["recon"])
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, 3)
    plt.grid(True)

    plt.subplot(1, 5, 2)
    plt.plot(history["emotion"])
    plt.title("Emotion Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, 3)
    plt.grid(True)

    plt.subplot(1, 5, 3)
    plt.plot(history["speaker_ce"])
    plt.title("Speaker CE Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, 3)
    plt.grid(True)

    plt.subplot(1, 5, 4)
    plt.plot(history["speaker_cos"])
    plt.title("Speaker Cosine Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, 3)
    plt.grid(True)

    plt.subplot(1, 5, 5)
    plt.plot(history["speaker_total"])
    plt.title("Total Speaker Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, 3)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "loss_plot.png"))
    print(f"Training plot saved to {os.path.join(checkpoint_dir, 'loss_plot.png')}")
    plt.show()

    return history
