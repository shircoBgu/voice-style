import os
import torch
from torch.nn import functional as F
from tqdm import tqdm
import torch.optim as optim


# Defines one full training epoch.
# model: our AutoVC model
# emotion_classifier_real: auxiliary classifier used for the emotion loss
# emotion_classifier_adv: adversarial classifer to separete the content embedding from the emotion 
# dataloader: yields batches of (source, target, emotion)
# optimizer: for AutoVC
# optimizer_cls: for the emotion classifier
# device: usually "cuda" or "cpu"
# lambda_ce: weight for the emotion classification loss




def train_one_epoch(model, emotion_classifier, dataloader, optimizer, optimizer_cls,
                    device, lambda_ce=0.5, lambda_spk=0.5):
    # Puts both models into training mode.
    model.train()
    emotion_classifier_real.train()

    # Setup for adversarial training
    content_encoder_real = ContentEncoder() #not sure we need that 
    emotion_classifier_real = EmotionClassifier() #not ruin the original emotion classifier we use
    content_encoder_adv = ContentEncoder()
    emotion_classifier_adv = EmotionClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer_encoder = torch.optim.Adam(content_encoder_adv.parameters(), lr=1e-4)
    optimizer_classifier = torch.optim.Adam(emotion_classifier_adv.parameters(), lr=1e-4)

    # Initialize accumulators to track average losses across all batches.
    total_recon_loss = 0
    total_emotion_loss = 0
    total_speaker_loss = 0

    for source_mel, target_mel, emotion_label in tqdm(dataloader):
        source_mel = source_mel.to(device)  # (B, T, 80)
        target_mel = target_mel.to(device)  # (B, T, 80)
        emotion_label = emotion_label.to(device)  # (B,)

#======Adversarial training for separating content from emotion======#
        # ========== STEP A: Train classifier ========== #
        #freeze encoder
        content_encoder_adv.eval()
        emotion_classifier_adv.train()

        with torch.no_grad():  # no gradient for encoder
              bottleneck = content_encoder_adv(source_mel)  # [B, T, D]
              pooled = bottleneck.mean(dim=1)    # [B, D]

        logits_adv = emotion_classifier_adv(pooled)  # [B, num_classes]
        clf_loss = loss_fn(logits_adv, emotion_label)

        optimizer_classifier.zero_grad()
        clf_loss.backward()
        optimizer_classifier.step()

        # ========== STEP B: Train encoder adversarially ========== #
        content_encoder_adv.train()
        emotion_classifier_adv.eval()  # freeze classifier

        bottleneck = content_encoder_adv(source_mel)  
        pooled = bottleneck.mean(dim=1)    # [B, D]

        logits_adv = emotion_classifier_adv(pooled)
        adv_loss = -loss_fn(logits_adv, emotion_label) #we want to 'fool' the classifier, we train the encoder to maximize the classifier’s error - as we increase the loss the encoder will 'hide' the emotion component better

        optimizer_encoder.zero_grad()
        adv_loss.backward()
        optimizer_encoder.step()
#============================================================================
        # === Forward pass ===
        # Predict the mel-spectrogram from source + speaker + emotion
        mel_pred = model(source_mel, target_mel, emotion_label)  # (B, T, 80)

        # === Loss: Reconstruction ===
        # Self-supervised trick — model tries to reconstruct mel when content = speaker = emotion
        recon_loss = F.l1_loss(mel_pred, target_mel)

        # === Loss: Emotion classification ===
        logits = emotion_classifier_real(mel_pred)  # (B, num_emotions)
        # Compares it to the ground truth emotion label using cross-entropy loss
        ce_loss = F.cross_entropy(logits, emotion_label)

        # === Speaker Consistency Loss ===
        with torch.no_grad():
            target_speaker_emb = model.speaker_encoder(target_mel)  # (B, D)
        pred_speaker_emb = model.speaker_encoder(mel_pred.detach())  # (B, D)

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(pred_speaker_emb, target_speaker_emb, dim=-1)  # (B,)
        speaker_loss = 1.0 - cos_sim.mean()  # (scalar)

        # === Combine losses ===
        total_loss = recon_loss + lambda_ce * ce_loss + lambda_spk * speaker_loss 

        # === Backprop ===
        optimizer.zero_grad()
        optimizer_cls.zero_grad()
        total_loss.backward()
        optimizer.step()
        optimizer_cls.step()

        total_recon_loss += recon_loss.item()
        total_emotion_loss += ce_loss.item()
        total_speaker_loss += speaker_loss.item()

    avg_recon = total_recon_loss / max(len(dataloader), 1)
    avg_ce = total_emotion_loss / max(len(dataloader), 1)
    avg_spk = total_speaker_loss / max(len(dataloader), 1)

    print(f"Avg Recon: {avg_recon:.4f} | Avg CE: {avg_ce:.4f} | Avg Spk: {avg_spk:.4f}")
    return avg_recon, avg_ce, avg_spk


def train(model, emotion_classifier_real, dataloader,
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
        checkpoint_dir: directory to save model checkpoints
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    history = {"recon": [], "emotion": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        avg_recon, avg_ce, avg_spk = train_one_epoch(
            model, emotion_classifier_real, dataloader,
            optimizer, optimizer_cls, device,
            lambda_ce, lambda_spk
        )

        print(f"Recon: {avg_recon:.4f} | Emotion: {avg_ce:.4f} | Speaker: {avg_spk:.4f}")

        # Save model checkpoints
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"autovc_epoch{epoch}.pt"))
        torch.save(emotion_classifier_real.state_dict(), os.path.join(checkpoint_dir, f"emotion_cls_epoch{epoch}.pt"))

        # Store loss history
        history["recon"].append(avg_recon)
        history["emotion"].append(avg_ce)
        history["speaker"].append(avg_spk)

    return history
