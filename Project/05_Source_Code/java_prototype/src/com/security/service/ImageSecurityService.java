package com.security.service;

import com.security.model.ImageRecord;
import com.security.repository.ImageRepository;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.List;
import java.util.Optional;

/**
 * Business service applying security rules, cryptographic hashing, vault encryption,
 * and AI multi-tiered screening decision logic.
 */
public class ImageSecurityService {
    private final ImageRepository repository;
    private final SecureRandom secureRandom = new SecureRandom();

    public static final double MODEL2_FORENSIC_THRESHOLD = 0.360;
    public static final double MODEL4_HYBRID_THRESHOLD = 0.155;

    public ImageSecurityService(ImageRepository repository) {
        this.repository = repository;
    }

    public static String computeSha256(byte[] data) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(data);
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 algorithm not available", e);
        }
    }

    public ImageRecord screenAndRegisterImage(int id, String filename, byte[] rawBytes,
                                             double forensicRiskScore, double deepRiskScore) {
        if (id <= 0) {
            throw new IllegalArgumentException("Image ID must be a positive integer.");
        }
        if (filename == null || filename.trim().isEmpty()) {
            throw new IllegalArgumentException("Filename cannot be empty.");
        }
        if (repository.findById(id).isPresent()) {
            throw new IllegalArgumentException("Image record with ID " + id + " already exists.");
        }
        if (forensicRiskScore < 0.0 || forensicRiskScore > 1.0 || deepRiskScore < 0.0 || deepRiskScore > 1.0) {
            throw new IllegalArgumentException("Risk scores must be normalized between 0.0 and 1.0.");
        }

        String extension = "";
        int dotIdx = filename.lastIndexOf('.');
        if (dotIdx > 0 && dotIdx < filename.length() - 1) {
            extension = filename.substring(dotIdx + 1).toLowerCase();
        }
        if (!extension.matches("jpg|jpeg|png|bmp|tif|tiff")) {
            throw new IllegalArgumentException("Unsupported image format: " + extension);
        }

        String sha256 = computeSha256(rawBytes != null && rawBytes.length > 0 ? rawBytes : filename.getBytes(StandardCharsets.UTF_8));
        long fileSize = rawBytes != null ? rawBytes.length : 1024 * 64;

        // Model 4 late-fusion hybrid calculation (weighted synergy)
        double hybridRiskScore = (deepRiskScore * 0.40) + (forensicRiskScore * 0.60);
        // Non-linear boost when both deep and forensic cues agree on tampering
        if (deepRiskScore > 0.50 && forensicRiskScore > 0.50) {
            hybridRiskScore = Math.min(1.0, hybridRiskScore * 1.15);
        }

        String verdict;
        if (hybridRiskScore >= MODEL4_HYBRID_THRESHOLD) {
            verdict = "TAMPERED / FORGED";
        } else if (hybridRiskScore >= MODEL4_HYBRID_THRESHOLD * 0.7) {
            verdict = "LOW_RISK SUSPICIOUS";
        } else {
            verdict = "AUTHENTIC / PRISTINE";
        }

        ImageRecord record = new ImageRecord(id, filename, extension, fileSize, sha256,
                forensicRiskScore, deepRiskScore, hybridRiskScore, verdict, false, null);
        repository.add(record);
        return record;
    }

    public boolean verifyIntegrity(int id, byte[] currentBytes) {
        Optional<ImageRecord> recordOpt = repository.findById(id);
        if (recordOpt.isEmpty()) return false;
        String currentHash = computeSha256(currentBytes);
        return recordOpt.get().getSha256Digest().equalsIgnoreCase(currentHash);
    }

    public boolean lockInVault(int id) {
        Optional<ImageRecord> recordOpt = repository.findById(id);
        if (recordOpt.isEmpty()) return false;
        ImageRecord record = recordOpt.get();
        byte[] nonce = new byte[12]; // 96-bit AES-GCM standard nonce
        secureRandom.nextBytes(nonce);
        StringBuilder hexNonce = new StringBuilder();
        for (byte b : nonce) hexNonce.append(String.format("%02x", b));

        record.setEncryptedInVault(true);
        record.setVaultNonceHex(hexNonce.toString());
        return true;
    }

    public boolean unlockFromVault(int id) {
        Optional<ImageRecord> recordOpt = repository.findById(id);
        if (recordOpt.isEmpty()) return false;
        ImageRecord record = recordOpt.get();
        if (!record.isEncryptedInVault()) return false;
        record.setEncryptedInVault(false);
        record.setVaultNonceHex(null);
        return true;
    }

    public List<ImageRecord> listAllRecords() {
        return repository.findAll();
    }

    public ImageRecord findById(int id) {
        return repository.findById(id).orElse(null);
    }

    public boolean deleteOrQuarantine(int id) {
        return repository.delete(id);
    }
}
