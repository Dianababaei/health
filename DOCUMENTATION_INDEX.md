# Documentation Index - Artemis Livestock Health Monitoring System

**Version:** 2.0
**Last Updated:** November 17, 2025

---

## Quick Start

**New to the system?** Read these documents in order:

1. [README.md](README.md) - System overview and installation
2. [QUICK_START.md](QUICK_START.md) - Get started in 5 minutes
3. [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Test all features

---

## Core Documentation

### System Architecture

| Document | Purpose | For Who |
|----------|---------|---------|
| [README.md](README.md) | System overview, installation, basic usage | Everyone |
| [APP_STRUCTURE.md](APP_STRUCTURE.md) | Directory structure and component organization | Developers |
| [DATABASE_SCHEMA_EXPLAINED.md](DATABASE_SCHEMA_EXPLAINED.md) | Database tables, columns, relationships | Developers, Data Analysts |
| [DATA_STORAGE_EXPLAINED.md](DATA_STORAGE_EXPLAINED.md) | File storage locations and formats | Developers, System Admins |

### User Guides

| Document | Purpose | For Who |
|----------|---------|---------|
| [QUICK_START.md](QUICK_START.md) | 5-minute getting started guide | New Users |
| [UPLOAD_WORKFLOW.md](UPLOAD_WORKFLOW.md) | How to upload and process sensor data | End Users |
| [END_TO_END_TEST_GUIDE.md](END_TO_END_TEST_GUIDE.md) | Complete workflow from data to dashboard | QA Testers, Users |

### Testing & Quality Assurance

| Document | Purpose | For Who |
|----------|---------|---------|
| [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) | **Complete testing scenarios and procedures** | QA Testers, Developers |
| [TEST_DATASETS_COMPARISON.md](TEST_DATASETS_COMPARISON.md) | Comparison of test data generators | Testers, Data Scientists |

### Data Generation

| Document | Purpose | For Who |
|----------|---------|---------|
| [DATASET_3_REPRODUCTIVE_FOCUS.md](DATASET_3_REPRODUCTIVE_FOCUS.md) | Reproductive health test data (estrus/pregnancy) | Data Scientists, Testers |
| [DATASET_4_RUMINATION_FOCUS.md](DATASET_4_RUMINATION_FOCUS.md) | Rumination-focused test data | Data Scientists, Testers |
| [TEST_DATA_SUMMARY.md](TEST_DATA_SUMMARY.md) | Summary of all test datasets | Testers |

### Production & Deployment

| Document | Purpose | For Who |
|----------|---------|---------|
| [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) | Production deployment checklist | System Admins, DevOps |
| [DATABASE_GUIDE.md](DATABASE_GUIDE.md) | Database setup and maintenance | DBAs, System Admins |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions | Support, System Admins |

### Recent Fixes & Updates

| Document | Purpose | For Who |
|----------|---------|---------|
| [ALERT_DASHBOARD_FIXES.md](ALERT_DASHBOARD_FIXES.md) | Alert pagination and timestamp fixes (Part 1) | Developers |
| [FINAL_DASHBOARD_FIXES.md](FINAL_DASHBOARD_FIXES.md) | **Complete fix summary (All parts)** | Developers, QA |

---

## Documentation by Role

### For End Users (Farmers, Veterinarians)

**Start here:**
1. [README.md](README.md) - What is this system?
2. [QUICK_START.md](QUICK_START.md) - How do I use it?
3. [UPLOAD_WORKFLOW.md](UPLOAD_WORKFLOW.md) - How do I upload data?

**When you need help:**
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

### For QA Testers

**Start here:**
1. [README.md](README.md) - System overview
2. [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - **Complete test procedures**

**Supporting documents:**
- [TEST_DATASETS_COMPARISON.md](TEST_DATASETS_COMPARISON.md) - Test data options
- [DATASET_3_REPRODUCTIVE_FOCUS.md](DATASET_3_REPRODUCTIVE_FOCUS.md) - Reproductive testing
- [DATASET_4_RUMINATION_FOCUS.md](DATASET_4_RUMINATION_FOCUS.md) - Rumination testing
- [END_TO_END_TEST_GUIDE.md](END_TO_END_TEST_GUIDE.md) - End-to-end workflow

### For Developers

**Start here:**
1. [README.md](README.md) - System overview
2. [APP_STRUCTURE.md](APP_STRUCTURE.md) - Code organization
3. [DATABASE_SCHEMA_EXPLAINED.md](DATABASE_SCHEMA_EXPLAINED.md) - Database design

**Code fixes:**
- [FINAL_DASHBOARD_FIXES.md](FINAL_DASHBOARD_FIXES.md) - **All recent fixes**
- [ALERT_DASHBOARD_FIXES.md](ALERT_DASHBOARD_FIXES.md) - Alert system fixes

**Testing:**
- [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Test all features

### For System Administrators

**Start here:**
1. [README.md](README.md) - System overview
2. [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) - **Production deployment**
3. [DATABASE_GUIDE.md](DATABASE_GUIDE.md) - Database setup

**Maintenance:**
- [DATA_STORAGE_EXPLAINED.md](DATA_STORAGE_EXPLAINED.md) - File locations
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Issue resolution

### For Data Scientists

**Start here:**
1. [DATABASE_SCHEMA_EXPLAINED.md](DATABASE_SCHEMA_EXPLAINED.md) - Data structure
2. [TEST_DATASETS_COMPARISON.md](TEST_DATASETS_COMPARISON.md) - Available datasets

**Data generation:**
- [DATASET_3_REPRODUCTIVE_FOCUS.md](DATASET_3_REPRODUCTIVE_FOCUS.md) - Reproductive patterns
- [DATASET_4_RUMINATION_FOCUS.md](DATASET_4_RUMINATION_FOCUS.md) - Rumination patterns
- [TEST_DATA_SUMMARY.md](TEST_DATA_SUMMARY.md) - Dataset overview

---

## File Organization

```
i:\livestock\health\
├── README.md                          # Main system documentation
├── DOCUMENTATION_INDEX.md              # This file
│
├── Getting Started
│   ├── QUICK_START.md                  # 5-minute quick start
│   ├── UPLOAD_WORKFLOW.md              # How to upload data
│   └── END_TO_END_TEST_GUIDE.md        # Complete workflow guide
│
├── Testing & QA
│   ├── COMPREHENSIVE_TESTING_GUIDE.md  # ⭐ Complete testing guide
│   ├── TEST_DATASETS_COMPARISON.md     # Dataset comparison
│   ├── DATASET_3_REPRODUCTIVE_FOCUS.md # Reproductive test data
│   ├── DATASET_4_RUMINATION_FOCUS.md   # Rumination test data
│   └── TEST_DATA_SUMMARY.md            # Dataset summary
│
├── System Architecture
│   ├── APP_STRUCTURE.md                # Directory structure
│   ├── DATABASE_SCHEMA_EXPLAINED.md    # Database design
│   └── DATA_STORAGE_EXPLAINED.md       # File storage
│
├── Production & Deployment
│   ├── PRODUCTION_GUIDE.md             # Production deployment
│   ├── DATABASE_GUIDE.md               # Database setup
│   └── TROUBLESHOOTING.md              # Common issues
│
└── Fixes & Updates
    ├── FINAL_DASHBOARD_FIXES.md        # ⭐ All recent fixes
    └── ALERT_DASHBOARD_FIXES.md        # Alert system fixes (Part 1)
```

---

## Key Features by Document

### Health Monitoring
- **Basic Monitoring:** [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Scenario 1
- **Health Score:** [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Scenario 5
- **Trend Analysis:** [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Scenario 6

### Alert System
- **Alert Detection:** [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Scenario 1
- **Alert Management:** [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Scenario 4
- **Alert Fixes:** [FINAL_DASHBOARD_FIXES.md](FINAL_DASHBOARD_FIXES.md)

### Reproductive Health
- **Estrus Detection:** [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Scenario 2
- **Pregnancy Detection:** [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Scenario 2
- **Test Data:** [DATASET_3_REPRODUCTIVE_FOCUS.md](DATASET_3_REPRODUCTIVE_FOCUS.md)

### Rumination
- **Detection:** [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Scenario 3
- **Test Data:** [DATASET_4_RUMINATION_FOCUS.md](DATASET_4_RUMINATION_FOCUS.md)
- **Display Fix:** [FINAL_DASHBOARD_FIXES.md](FINAL_DASHBOARD_FIXES.md) - Issue 7

---

## Recent Updates (November 17, 2025)

### Major Fixes Applied
1. ✅ Alert pagination (10 → 100 alerts visible)
2. ✅ Alert timestamps (show actual detection time)
3. ✅ Health score accuracy (reflects all alerts)
4. ✅ Rumination detection (recognizes all state variants)
5. ✅ Rumination display (shows actual percentage)
6. ✅ Alert status label clarity
7. ✅ Future date handling for simulation data

**See:** [FINAL_DASHBOARD_FIXES.md](FINAL_DASHBOARD_FIXES.md) for complete details

### New Documentation
- ✅ [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md) - Complete testing procedures
- ✅ [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - This index

---

## How to Use This Index

### Scenario 1: "I want to test the system"
→ Go to [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md)

### Scenario 2: "I want to deploy to production"
→ Go to [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)

### Scenario 3: "Something isn't working"
→ Go to [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
→ Or check [FINAL_DASHBOARD_FIXES.md](FINAL_DASHBOARD_FIXES.md) for recent fixes

### Scenario 4: "I need to understand the data structure"
→ Go to [DATABASE_SCHEMA_EXPLAINED.md](DATABASE_SCHEMA_EXPLAINED.md)

### Scenario 5: "I want to generate test data"
→ Go to [TEST_DATASETS_COMPARISON.md](TEST_DATASETS_COMPARISON.md)
→ Then [DATASET_3_REPRODUCTIVE_FOCUS.md](DATASET_3_REPRODUCTIVE_FOCUS.md) or [DATASET_4_RUMINATION_FOCUS.md](DATASET_4_RUMINATION_FOCUS.md)

### Scenario 6: "I'm new and don't know where to start"
→ Go to [README.md](README.md) → [QUICK_START.md](QUICK_START.md)

---

## Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| COMPREHENSIVE_TESTING_GUIDE.md | ✅ Current | Nov 17, 2025 |
| FINAL_DASHBOARD_FIXES.md | ✅ Current | Nov 17, 2025 |
| PRODUCTION_GUIDE.md | ✅ Current | Nov 15, 2025 |
| DATABASE_SCHEMA_EXPLAINED.md | ✅ Current | Nov 16, 2025 |
| DATASET_3_REPRODUCTIVE_FOCUS.md | ✅ Current | Nov 17, 2025 |
| DATASET_4_RUMINATION_FOCUS.md | ✅ Current | Nov 17, 2025 |
| All others | ✅ Current | Nov 2025 |

---

## Getting Help

1. **Check the documentation** using this index
2. **Run the tests** in [COMPREHENSIVE_TESTING_GUIDE.md](COMPREHENSIVE_TESTING_GUIDE.md)
3. **Review recent fixes** in [FINAL_DASHBOARD_FIXES.md](FINAL_DASHBOARD_FIXES.md)
4. **Check troubleshooting** in [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Maintained by:** Artemis Health Development Team
**Version:** 2.0
**Last Updated:** November 17, 2025
