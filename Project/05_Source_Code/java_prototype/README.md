# Java Prototype: Image Security & Protection System
## Software Engineering Lab Implementation Model (Layered Architecture)

This Java prototype satisfies Section 7 of the Software Engineering Project Guidelines (*Small Java Implementation Model*). It demonstrates a clean, decoupled Object-Oriented Architecture for image asset security, metadata persistence, cryptographic verification, and AI risk decision logic.

---

## 1. Layered Architecture

| Layer | Component Class | Architectural Responsibility |
| :--- | :--- | :--- |
| **Model** | `com.security.model.ImageRecord` | Encapsulates image entity data, SHA-256 digest, Model 2 & Model 4 risk scores, verdict, and AES-GCM vault metadata. |
| **Repository** | `com.security.repository.ImageRepository` | In-memory thread-safe data store supporting CRUD operations (`add`, `findAll`, `findById`, `findByHash`, `delete`). |
| **Service** | `com.security.service.ImageSecurityService` | Business rule engine computing SHA-256 digests, multi-tiered risk fusion, validation, and 96-bit AES-GCM nonce generation. |
| **Presentation** | `com.security.app.ImageSecurityApp` | Console-based interactive menu allowing analysts to screen images, verify integrity, lock assets in vault, and quarantine files. |

---

## 2. Compilation & Execution Instructions

From this directory (`Project/05_Source_Code/java_prototype/`):

```bash
# 1. Create output binary directory
mkdir -p bin

# 2. Compile all source files
javac -d bin src/com/security/model/*.java src/com/security/repository/*.java src/com/security/service/*.java src/com/security/app/*.java

# 3. Execute the interactive application
java -cp bin com.security.app.ImageSecurityApp
```

---

## 3. Sample Console Output

```text
========================================================================
   AI-BASED IMAGE SECURITY AND PROTECTION SYSTEM - JAVA PROTOTYPE
   Software Engineering Lab Implementation Model (Layered Architecture)
========================================================================

----------------- SYSTEM OPERATIONS MENU -----------------
1. Add & Screen New Image (AI Multi-Tiered Evaluation)
2. Display All Managed Image Records & Security Vault
3. Search Image Record by ID
4. Verify Image Cryptographic Integrity (SHA-256 Bitstream)
5. Lock Image Asset in Secure Vault (AES-256-GCM AEAD)
6. Unlock Image Asset from Vault
7. Delete / Quarantine Compromised Image Record
8. Exit Application
```
