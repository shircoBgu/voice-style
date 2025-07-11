# -*- coding: utf-8 -*-
import glob
import os
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.emotion_utils import extract_emotion_embedding


# Defines one full training epoch.
# model: our AutoVC model
# emotion_classifier: auxiliary classifier used for the emotion loss
# dataloader: yields batches of (source, target, emotion)
# optimizer: for AutoVC
# optimizer_cls: for the emotion classifier
# device: usually "cuda" or "cpu"
# lambda_ce: weight for the emotion classification loss
# lambda_spk: weight for the speaker loss

# =======================================test=========================================
def plot_mel_comparison(source_mel, target_mel, predicted_mel, step, save_dir, title_prefix=""):
    os.makedirs(save_dir, exist_ok=True)

    if torch.is_tensor(source_mel):
        source_mel = source_mel.detach().cpu().numpy()
    if torch.is_tensor(target_mel):
        target_mel = target_mel.detach().cpu().numpy()
    if torch.is_tensor(predicted_mel):
        predicted_mel = predicted_mel.detach().cpu().numpy()

    source_mel = source_mel.T
    target_mel = target_mel.T
    predicted_mel = predicted_mel.T

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axs[0].imshow(source_mel, origin='lower', aspect='auto', cmap='magma')
    axs[0].set_title(f"{title_prefix}Source Mel")

    axs[1].imshow(target_mel, origin='lower', aspect='auto', cmap='magma')
    axs[1].set_title(f"{title_prefix}Target Mel")

    axs[2].imshow(predicted_mel, origin='lower', aspect='auto', cmap='magma')
    axs[2].set_title(f"{title_prefix}Predicted Mel (Output)")

    for ax in axs:
        ax.set_ylabel("Mel bins")
    axs[-1].set_xlabel("Time frames")

    plt.tight_layout()

    # Save the plot
    filename = os.path.join(save_dir, f"mel_plot_step_{step:05d}.png")
    plt.savefig(filename)
    plt.close(fig)  # Don't keep figures open


def extract_epoch_num(filename):
    try:
        return int(filename.split("epoch")[-1].split(".")[0])
    except:
        return -1  # fallback


def train_one_epoch(model, dataloader, optimizer,
                    device, lamda=1, mu=1, global_step=0):
    # Puts both models into training mode.
    model.train()
    # emotion_classifier.train()
    # Initialize accumulators to track average losses across all batches.
    total_recon_loss = 0
    # total_emotion_loss = 0
    total_recon_loss_post = 0
    total_content_loss = 0

    for source_mel, target_mel, target_emotion_id, target_speaker_id, src_wav_path, tgt_wav_path in tqdm(dataloader):
        source_mel = source_mel.to(device)  # (B, T, 80)
        target_mel = target_mel.to(device)  # (B, T, 80)
        emotion_embeds = torch.stack([
            extract_emotion_embedding(wav_path) for wav_path in src_wav_path
        ]).to(device)  # shape: (B, 64) or (B, 128) depending on emotion2vec model
        # emotion_label = emotion_label.to(device)  # (B,)
        target_speaker_id = target_speaker_id.to(device)  # (B,)

        # === Forward pass ===
        # Predict the mel-spectrogram from source + speaker + emotion
        mel_pred_post, mel_pred, content_emb, spk_logits, src_spk_emb, trg_spk_emb = model(
            source_mel, target_mel, emotion_embedding=emotion_embeds
        )
        emotion_pred = extract_emotion_embedding(predicted_wav_path)
        L_emo = F.l1_loss(emotion_pred, emotion_embed_batch)
        # =================================================test=====================================
        if global_step % 20 == 0:
            plot_mel_comparison(
                source_mel[0], target_mel[0], mel_pred[0],
                step=global_step,
                save_dir="/content/drive/MyDrive/voice_style_project/voice_style_project_new/scripts/mel_plot_new",
                title_prefix=f" Step {global_step}: "
            )
        global_step += 1
        # =============================================================================================

        # === Loss: Reconstruction ===
        recon_loss_post = F.mse_loss(mel_pred_post, source_mel)
        recon_loss = F.mse_loss(mel_pred, source_mel)
        print(f"recon loss:{recon_loss:.4f}")

        # conent embedding loss
        content_recon = model.content_encoder(mel_pred_post, src_spk_emb)
        content_recon = torch.cat(content_recon, dim=-1)
        content_loss = F.l1_loss(content_emb, content_recon)

        # === Loss: Emotion classification ===
        # logits = emotion_classifier(mel_pred)  # (B, num_emotions)
        # Compares it to the ground truth emotion label using cross-entropy loss
        # ce_loss = F.cross_entropy(logits, emotion_label)

        # === Combine losses ===
        total_loss = recon_loss + recon_loss_post * mu + content_loss * lamda

        # === Backprop ===
        optimizer.zero_grad()
        # optimizer_cls.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # optimizer_cls.step()

        total_recon_loss += recon_loss.item()
        # total_emotion_loss += ce_loss.item()
        total_recon_loss_post += recon_loss_post.item()
        total_content_loss += content_loss.item()

    avg_recon = total_recon_loss / max(len(dataloader), 1)
    # avg_ce = total_emotion_loss / max(len(dataloader), 1)
    avg_recon_post = total_recon_loss_post / max(len(dataloader), 1)
    avg_content_loss = total_content_loss / max(len(dataloader), 1)

    print(
        f"Avg Recon: {avg_recon:.4f} | Avg Recon Post: {recon_loss_post:.4f}"
        f" | Avg Content Recon: {avg_content_loss:.4f}")

    return avg_recon, avg_recon_post, avg_content_loss, global_step


def train(model, dataloader,
          optimizer, device,
          num_epochs=100, lamda=1, mu=1,
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
        history = {"recon": [], "recon_post": [], "content_recon": []}

    # === Try to resume training ===

    autovc_ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "autovc_epoch*.pt")),
        key=extract_epoch_num
    )

    start_epoch = 1
    if autovc_ckpts:
        last_autovc = autovc_ckpts[-1]
        # last_cls = cls_ckpts[-1]
        print(f"Resuming from checkpoint: {last_autovc} ")
        model.load_state_dict(torch.load(last_autovc, map_location=device))
        # emotion_classifier.load_state_dict(torch.load(last_cls, map_location=device))
        # Extract epoch number from filename
        start_epoch = int(last_autovc.split("epoch")[-1].split(".")[0]) + 1

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch}/{start_epoch + num_epochs - 1}")
        global_step = 0
        avg_recon, avg_recon_post, avg_content_loss, global_step = train_one_epoch(
            model, dataloader,
            optimizer, device,
            lamda, mu, global_step=global_step
        )

        print(f"Recon Post: {avg_recon_post:.4f}  | content: {avg_content_loss:.4f}")

        # Save model checkpoints
        autovc_path = os.path.join(checkpoint_dir, f"autovc_epoch{epoch}.pt")
        # emotion_cls_path = os.path.join(checkpoint_dir, f"emotion_cls_epoch{epoch}.pt")
        combined_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")

        torch.save(model.state_dict(), autovc_path)
        # torch.save(emotion_classifier.state_dict(), emotion_cls_path)

        # Save combined checkpoint with mappings
        torch.save({
            "model_state": model.state_dict(),
            # "emotion_classifier_state": emotion_classifier.state_dict(),
            "speaker2idx": dataloader.dataset.speaker2idx,
            # "emo2idx": dataloader.dataset.emo2idx,
        }, combined_path)

        # Store loss history
        history["recon"].append(avg_recon)
        # history["emotion"].append(avg_ce)
        history["recon_post"].append(avg_recon_post)
        history["content_recon"].append(avg_content_loss)
        # history["speaker_total"].append(avg_spk)
        # history["speaker_ce"].append(avg_ce_spk)
        # history["speaker_cos"].append(avg_cs_spk)

    torch.save(history, os.path.join(checkpoint_dir, "train_history.pt"))

    # === Plot training losses ===
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 5, 1)
    plt.plot(history["recon"])
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylim(0, 3)
    plt.grid(True)

    # plt.subplot(1, 5, 2)
    # plt.plot(history["emotion"])
    # plt.title("Emotion Loss")
    # plt.xlabel("Epoch")
    # plt.ylim(0, 3)
    # plt.grid(True)

    plt.subplot(1, 5, 2)
    plt.plot(history["recon_post"])
    plt.title("recon_post")
    plt.xlabel("Epoch")
    plt.ylim(0, 5)
    plt.grid(True)

    plt.subplot(1, 5, 3)
    plt.plot(history["content_recon"])
    plt.title("content_recon")
    plt.xlabel("Epoch")
    plt.ylim(0, 3)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "loss_plot.png"))
    print(f"Training plot saved to {os.path.join(checkpoint_dir, 'loss_plot.png')}")
    plt.show()

    return history
