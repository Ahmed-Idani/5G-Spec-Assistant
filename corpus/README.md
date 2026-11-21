# Corpus

Normative 3GPP 5G specifications used by the RAG chatbot.

| Spec | Title | Version | Release | Stage | File | Source |
|---|---|---|---|---|---|---|
| TS 23.501 | System architecture for the 5G System (5GS) | V18.12.0 | Rel-18 | Stage 2 | `raw/23501-ic0.docx` | https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/23501-ic0.zip |
| TS 24.501 | Non-Access-Stratum (NAS) protocol for 5G System (5GS) | V18.14.0 | Rel-18 | Stage 3 | `raw/24501-ie0.docx` | https://www.3gpp.org/ftp/Specs/archive/24_series/24.501/24501-ie0.zip |

- **Downloaded:** 2026-11-21, from the official 3GPP FTP archive.
- **Format:** original `.docx`. 3GPP publishes Word files only. The zip archives were extracted and then removed.
- **File naming:** `<spec><version>.docx`, where the version is coded as letters (`i` = 18, `c` = 12, `e` = 14). So `ic0` = V18.12.0 and `ie0` = V18.14.0.

## Why these two specs
- **TS 23.501** describes the architecture: network functions (AMF, SMF, UPF…), reference points, mobility, sessions, QoS and network slicing. It explains *why* procedures exist and how they work.
- **TS 24.501** specifies the UE↔network NAS signalling: 5GMM/5GSM messages, information elements, timers and cause codes. It defines *exactly how* procedures are carried out.

Together they support cross-spec questions, for example the registration procedure (23.501) and the NAS messages exchanged with the AMF (24.501).

## Layout
```
corpus/
├── README.md        this file
└── raw/             original specs, do not edit
    ├── 23501-ic0.docx
    └── 24501-ie0.docx
```
