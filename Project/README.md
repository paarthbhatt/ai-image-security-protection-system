# AI-Based Image Security and Protection System
## Software Engineering & Artificial Intelligence Academic Project Submission

This directory represents the formal, submission-ready project deliverable adhering to the **Software Engineering Project Documentation & Implementation Guideline (Version 1.0)**.

---

## 1. Project Directory Structure

```text
Project/
├── 01_Project_Proposal.pdf           # Problem definition, target users, scope, objectives, and work plan
├── 02_SRS.pdf                        # Software Requirements Specification (IEEE 830 compliant)
├── 03_SDD.pdf                        # Software Design Document (Architecture, UML, DB, Traceability Matrix)
├── 04_Wireframes.pdf                 # UI/UX wireframes, screen hierarchy, and review checklist
├── 05_Source_Code/                   # Complete source code (Java prototype + full Python AI pipeline)
│   ├── java_prototype/        # Section 7 Java Implementation Model (Layered OOP Architecture)
│   │   ├── src/com/security/
│   │   │   ├── model/         # ImageRecord entity
│   │   │   ├── repository/    # In-memory ImageRepository
│   │   │   ├── service/       # ImageSecurityService (SHA-256, AES-GCM, Rule Engine)
│   │   │   └── app/           # ImageSecurityApp (Console CLI application)
│   │   └── README.md          # Java compilation and execution instructions
│   └── python_system/         # Complete Python AI & Cryptographic Protection Core
│       ├── src/               # features.py, deep_detector.py, hybrid_detector.py, etc.
│       ├── tests/             # 79 automated unit tests
│       ├── demo.py            # Live CLI screener & visual ELA generator
│       └── requirements.txt   # Python dependency specifications
├── 06_Test_Cases.xlsx         # 25 detailed test cases with requirements traceability & execution metrics
├── 07_Test_Evidence/         # Concrete verification evidence
│   ├── 01_unit_test_suite_log.txt         # 79 unit tests passing run log
│   ├── 02_live_demo_screener_execution.txt # Dual authentic vs spliced live screening output
│   ├── 03_cryptographic_audit_record.json  # Real AES-GCM & SHA-256 audit trail
│   ├── 04_four_model_comparison.png       # Model 4 outperforming literature benchmark
│   ├── 05_roc_pr_curves.png               # ROC and PR curves with bootstrap CIs
│   ├── 06_segmentation_predictions.png    # U-Net pixel-level localization triplets
│   ├── 07_feature_importances.png         # SRM, ELA, and Wavelet feature weights
│   ├── 08_demo_ela_inspection.png         # Visual Error Level Analysis difference map
│   └── 09_java_prototype_console_log.txt  # Java prototype interactive console output
├── 08_Presentation.pptx       # 10-slide team PowerPoint deck for group evaluation
├── AI_ML_Final_Project_Report.docx # Formatted university project report in editable Word format
├── Final_Project_Report.pdf   # 8-page unified publication-grade master project report
└── README.md                  # This submission master document
```

---

## 2. Team Members & Roles

**Project Group No.:** Group 41 &nbsp;|&nbsp; **Academic Year:** 2026–2027 &nbsp;|&nbsp; **Programme:** B.Tech CSE  
**Faculty Mentors:** Dr. Shreyansh Sharma (Software Engineering) & Dr. Shweta Malwe (Artificial Intelligence) &nbsp;|&nbsp; **Institution:** Department of Computer Science & Engineering  

| Team Member Name | Group & Programme | Primary Role & Architectural Contributions | Evaluation Track |
| :--- | :--- | :--- | :--- |
| **PARTH BHATT** | Group 41 / B.Tech CSE | Project Lead & System Architect — Multimodal Late-Fusion (Model 4), ResNet-18 Backbone, Evaluation Framework | AI & Software Engineering |
| **SYED ALI AKBAR** | Group 41 / B.Tech CSE | Data Engineering Lead — CASIA v2.0, Columbia, COVERAGE Pipelines, Disjoint-Set Union (DSU) Graph Partitioning | Software Engineering & AI |
| **GARVIT SINGH** | Group 41 / B.Tech CSE | Cryptographic Security Lead — FIPS SHA-256 Digest Engine, AES-256-GCM AEAD Vault, U-Net Segmentation Integration | Software Engineering |
| **NIPUN BATRA** | Group 41 / B.Tech CSE | Quality Assurance Lead — 79 Automated Unit Tests, 25 System Test Cases, UI Wireframes, Requirements Traceability | Software Engineering |

---

## 3. Rubric & Deliverables Compliance Matrix

The submission fulfills all criteria of the **Evaluation Rubric** (Section 11):

| Evaluation Component | Marks | Deliverable Location | Evidence Summary |
| :--- | :---: | :--- | :--- |
| **Problem Definition & Scope** | Complete | `01_Project_Proposal.pdf` | Rigorous problem background, target user roles, explicit in-scope and out-of-scope boundaries, and work breakdown. |
| **Software Requirements Specification (SRS)** | Complete | `02_SRS.pdf` | Complete IEEE 830 structure: 10 Functional Requirements (FR-01 to FR-10), 6 Non-Functional Requirements, 4 Use Cases, Data Dictionary, and Acceptance Criteria. |
| **Software Design Document (SDD) & UML** | Complete | `03_SDD.pdf` | Layered architecture, 8 detailed module specifications, OOP class models, sequence flows, error handling, security design, and full Requirements Traceability Matrix (RTM). |
| **Wireframe / UI Design** | Complete | `04_Wireframes.pdf` | 6 complete low/medium-fidelity screens (Login, Dashboard, Screener, Triple-Viewport Inspection, Vault, Quarantine Modal), navigation hierarchy, and review checklist. |
| **Java Implementation Model** | Complete | `05_Source_Code/java_prototype/` | Clean object-oriented layered architecture (Model, Repository, Service, App) implementing CRUD, SHA-256 calculation, and AES-256-GCM vaulting. |
| **Testing & Traceability** | Complete | `06_Test_Cases.xlsx` & `07_Test_Evidence/` | 25 detailed test cases mapped to SRS requirements, 100% pass status, backed by 79 automated invariant unit tests and visual evidence. |
| **All Required Artifacts** | **15 Marks Total** | **Complete Project Package** | **Production-Ready & Fully Verified** |

---

## 4. Key Scientific & Engineering Results

### A. Four-Model Benchmark Comparison (Held-Out CASIA v2.0 Test Set)
Every model was evaluated on the exact same frozen, scene-disjoint split (`data/processed/benchmark_casia2.csv`, 0 scene leakage):

| Model / Benchmark | Architecture / Features | Accuracy | Precision | Recall | F1 Score | ROC-AUC | MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Model 1: Logistic Regression | 143 Classical Forensics | 80.21% | 0.7105 | 0.8269 | 0.7643 | 0.8744 | 0.6005 |
| Model 3: Deep ResNet-18 | 224×224 RGB Pixels (Fine-Tuned) | 76.18% | 0.6533 | 0.8229 | 0.7283 | 0.8221 | 0.5322 |
| Model 2: HistGBDT Forensics | 143 Signal Residuals | 87.18% | 0.7976 | 0.8975 | 0.8446 | 0.9430 | 0.7399 |
| **Yadav & Vishwakarma (June 2024)** | **Two-Branch ResNet-CNN** | **88.51%** | **0.8123** | **0.9415** | **0.8664** | **0.9596** | **0.7744** |
| **Model 4: Late-Fusion Hybrid** | **512 Deep + 143 Forensics (655D)** | **90.96%** | **0.8538** | **0.9254** | **0.8882** | **0.9725** | **0.8143** |
| **Margin over 2024 Literature** | — | **+2.45%** | **+4.15%** | *(Balanced)* | **+0.0218** | **+0.0129** | **+0.0399** |

### B. Pixel-Level Tampering Localization (U-Net)
* **Tampered Pixel Dice:** **0.2855** *(Up from 0.131 tiled baseline — more than 2x improvement)*
* **Tampered Pixel IoU:** **0.2026** *(Up from 0.082 tiled baseline — 2.5x improvement)*
* **Zero-Shot COVERAGE Transfer Dice:** **0.1929**

### C. Software Quality Assurance
* **Automated Unit Tests:** **79 / 79 PASSED** in 43.5s (`test_metric_contracts.py`, `test_copymove_group_is_alive.py`, `test_modified_casia.py`, `test_artifact_hygiene.py`, etc.).
* **Data Leakage Guarantee:** Transitive Disjoint-Set Union clustering verified 12,614 images into 6,523 independent scenes with **0.0% cross-fold contamination**.

---

## 5. Quick Execution Guide

### Live Practical Demonstration (Python CLI):
```bash
# Run dual screener on sample authentic and tampered images:
python demo.py

# Screen any custom image with visual ELA difference artifact:
python demo.py --image path/to/image.jpg

# Execute the 79 automated software engineering unit tests:
python -m unittest discover -s tests -t .
```

### Java Implementation Model:
```bash
cd Project/05_Source_Code/java_prototype
mkdir -p bin
javac -d bin src/com/security/model/*.java src/com/security/repository/*.java src/com/security/service/*.java src/com/security/app/*.java
java -cp bin com.security.app.ImageSecurityApp
```

