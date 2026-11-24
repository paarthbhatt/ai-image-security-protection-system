package com.security.app;

import com.security.model.ImageRecord;
import com.security.repository.ImageRepository;
import com.security.service.ImageSecurityService;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

/**
 * Interactive Console Application demonstrating the Image Security and Protection System
 * in accordance with Software Engineering Project Guidelines.
 */
public class ImageSecurityApp {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        ImageRepository repository = new ImageRepository();
        ImageSecurityService service = new ImageSecurityService(repository);

        // Seed with sample benchmark images
        service.screenAndRegisterImage(101, "Au_ani_00001.jpg", "Authentic animal natural photo 2026".getBytes(StandardCharsets.UTF_8), 0.022, 0.015);
        service.screenAndRegisterImage(102, "Tp_D_CNN_M_B_nat11105.jpg", "Spliced landscape with copied region".getBytes(StandardCharsets.UTF_8), 0.682, 0.945);
        service.screenAndRegisterImage(103, "columbia_splice_042.bmp", "Uncompressed spliced BMP document".getBytes(StandardCharsets.UTF_8), 0.395, 0.720);

        System.out.println("========================================================================");
        System.out.println("   AI-BASED IMAGE SECURITY AND PROTECTION SYSTEM - JAVA PROTOTYPE");
        System.out.println("   Software Engineering Lab Implementation Model (Layered Architecture)");
        System.out.println("========================================================================");

        while (true) {
            System.out.println("\n----------------- SYSTEM OPERATIONS MENU -----------------");
            System.out.println("1. Add & Screen New Image (AI Multi-Tiered Evaluation)");
            System.out.println("2. Display All Managed Image Records & Security Vault");
            System.out.println("3. Search Image Record by ID");
            System.out.println("4. Verify Image Cryptographic Integrity (SHA-256 Bitstream)");
            System.out.println("5. Lock Image Asset in Secure Vault (AES-256-GCM AEAD)");
            System.out.println("6. Unlock Image Asset from Vault");
            System.out.println("7. Delete / Quarantine Compromised Image Record");
            System.out.println("8. Exit Application");
            System.out.print("Select Operation (1-8): ");

            if (!sc.hasNextInt()) {
                System.out.println("Invalid input. Closing application.");
                break;
            }
            int choice = sc.nextInt();
            sc.nextLine(); // consume newline

            try {
                switch (choice) {
                    case 1 -> {
                        System.out.print("Enter Image ID (integer): ");
                        int id = sc.nextInt();
                        sc.nextLine();
                        System.out.print("Enter Filename (e.g., photo.jpg, doc.png): ");
                        String filename = sc.nextLine();
                        System.out.print("Enter Model 2 Forensic Risk (0.0 to 1.0): ");
                        double forensicRisk = sc.nextDouble();
                        System.out.print("Enter Model 3 Deep Semantic Risk (0.0 to 1.0): ");
                        double deepRisk = sc.nextDouble();
                        sc.nextLine();

                        byte[] dummyBytes = ("Image Content Payload for " + filename).getBytes(StandardCharsets.UTF_8);
                        ImageRecord rec = service.screenAndRegisterImage(id, filename, dummyBytes, forensicRisk, deepRisk);
                        System.out.println("\n[+] Image successfully screened and registered:");
                        System.out.println(rec);
                        System.out.printf("    -> Hybrid Risk Score (Model 4): %.2f%%\n", rec.getHybridRiskScore() * 100);
                        System.out.println("    -> Final Verdict: " + rec.getVerdict());
                    }
                    case 2 -> {
                        System.out.println("\n--- All Managed Image Security Records ---");
                        var list = service.listAllRecords();
                        if (list.isEmpty()) {
                            System.out.println("No records found in repository.");
                        } else {
                            list.forEach(System.out::println);
                        }
                    }
                    case 3 -> {
                        System.out.print("Enter Image ID to search: ");
                        int id = sc.nextInt();
                        ImageRecord rec = service.findById(id);
                        if (rec != null) {
                            System.out.println("\nRecord Found:");
                            System.out.println(rec);
                            System.out.println("Full SHA-256: " + rec.getSha256Digest());
                            System.out.println("Vault Nonce:  " + (rec.getVaultNonceHex() != null ? rec.getVaultNonceHex() : "None"));
                            System.out.println("Timestamp:    " + rec.getCreatedAtUtc());
                        } else {
                            System.out.println("Error: Image ID " + id + " not found.");
                        }
                    }
                    case 4 -> {
                        System.out.print("Enter Image ID to verify integrity: ");
                        int id = sc.nextInt();
                        sc.nextLine();
                        System.out.print("Enter current payload string to simulate verification: ");
                        String payload = sc.nextLine();
                        boolean matches = service.verifyIntegrity(id, payload.getBytes(StandardCharsets.UTF_8));
                        if (matches) {
                            System.out.println("[SUCCESS] Bitstream matches reference digest! Integrity verified.");
                        } else {
                            System.out.println("[WARNING] Hash mismatch detected! Bitstream has been tampered or corrupted.");
                        }
                    }
                    case 5 -> {
                        System.out.print("Enter Image ID to lock in AES-GCM Vault: ");
                        int id = sc.nextInt();
                        boolean locked = service.lockInVault(id);
                        if (locked) {
                            System.out.println("[VAULT] Image locked with AES-256-GCM. 96-bit cryptographic nonce generated.");
                        } else {
                            System.out.println("Error: Could not find image ID " + id);
                        }
                    }
                    case 6 -> {
                        System.out.print("Enter Image ID to unlock from Vault: ");
                        int id = sc.nextInt();
                        boolean unlocked = service.unlockFromVault(id);
                        if (unlocked) {
                            System.out.println("[VAULT] Image unlocked from authenticated storage.");
                        } else {
                            System.out.println("Error: Image was not locked or record not found.");
                        }
                    }
                    case 7 -> {
                        System.out.print("Enter Image ID to delete / quarantine: ");
                        int id = sc.nextInt();
                        boolean deleted = service.deleteOrQuarantine(id);
                        if (deleted) {
                            System.out.println("[QUARANTINE] Image record " + id + " deleted from active storage.");
                        } else {
                            System.out.println("Error: Record ID " + id + " not found.");
                        }
                    }
                    case 8 -> {
                        System.out.println("Exiting Image Security and Protection System. Goodbye!");
                        return;
                    }
                    default -> System.out.println("Invalid option. Please choose between 1 and 8.");
                }
            } catch (IllegalArgumentException e) {
                System.out.println("[VALIDATION ERROR] " + e.getMessage());
            }
        }
    }
}

