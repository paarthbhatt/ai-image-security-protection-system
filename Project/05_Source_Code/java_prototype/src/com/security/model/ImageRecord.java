package com.security.model;

import java.time.Instant;

/**
 * Represents an image entity managed by the Image Security and Protection System.
 * Encapsulates cryptographic metadata, forensic AI risk scores, and vault status.
 */
public class ImageRecord {
    private int id;
    private String filename;
    private String format;
    private long fileSizeBytes;
    private String sha256Digest;
    private double forensicRiskScore; // 0.0 to 1.0 (from 143 signal features)
    private double deepRiskScore;     // 0.0 to 1.0 (from ResNet-18 embeddings)
    private double hybridRiskScore;   // 0.0 to 1.0 (Model 4 Multimodal Fusion)
    private String verdict;           // "AUTHENTIC", "LOW_RISK", "TAMPERED / FORGED"
    private boolean encryptedInVault; // AES-256-GCM AEAD status
    private String vaultNonceHex;
    private String createdAtUtc;

    public ImageRecord(int id, String filename, String format, long fileSizeBytes, String sha256Digest,
                       double forensicRiskScore, double deepRiskScore, double hybridRiskScore,
                       String verdict, boolean encryptedInVault, String vaultNonceHex) {
        this.id = id;
        this.filename = filename;
        this.format = format;
        this.fileSizeBytes = fileSizeBytes;
        this.sha256Digest = sha256Digest;
        this.forensicRiskScore = forensicRiskScore;
        this.deepRiskScore = deepRiskScore;
        this.hybridRiskScore = hybridRiskScore;
        this.verdict = verdict;
        this.encryptedInVault = encryptedInVault;
        this.vaultNonceHex = vaultNonceHex;
        this.createdAtUtc = Instant.now().toString();
    }

    public int getId() { return id; }
    public String getFilename() { return filename; }
    public String getFormat() { return format; }
    public long getFileSizeBytes() { return fileSizeBytes; }
    public String getSha256Digest() { return sha256Digest; }
    public double getForensicRiskScore() { return forensicRiskScore; }
    public double getDeepRiskScore() { return deepRiskScore; }
    public double getHybridRiskScore() { return hybridRiskScore; }
    public String getVerdict() { return verdict; }
    public boolean isEncryptedInVault() { return encryptedInVault; }
    public String getVaultNonceHex() { return vaultNonceHex; }
    public String getCreatedAtUtc() { return createdAtUtc; }

    public void setVerdict(String verdict) { this.verdict = verdict; }
    public void setEncryptedInVault(boolean encryptedInVault) { this.encryptedInVault = encryptedInVault; }
    public void setVaultNonceHex(String vaultNonceHex) { this.vaultNonceHex = vaultNonceHex; }

    @Override
    public String toString() {
        return String.format("[%d] %-24s | %-4s | %6d B | SHA: %s... | M2: %5.1f%% | M4: %5.1f%% | Verdict: %-18s | Vault: %s",
                id, filename, format.toUpperCase(), fileSizeBytes,
                sha256Digest != null && sha256Digest.length() >= 8 ? sha256Digest.substring(0, 8) : "N/A",
                forensicRiskScore * 100, hybridRiskScore * 100, verdict,
                encryptedInVault ? "AES-GCM (LOCKED)" : "UNENCRYPTED");
    }
}

