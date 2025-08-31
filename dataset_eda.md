1. CICFLowMeter_out

🔹 Dataset Preview:
shape: (5, 84)
┌─────────────────────────────────┬──────────────┬──────────┬────────────────┬───┬──────────┬──────────┬──────────┬────────────────┐
│ Flow ID                         ┆ Src IP       ┆ Src Port ┆ Dst IP         ┆ … ┆ Idle Std ┆ Idle Max ┆ Idle Min ┆ Label          │
│ ---                             ┆ ---          ┆ ---      ┆ ---            ┆   ┆ ---      ┆ ---      ┆ ---      ┆ ---            │
│ str                             ┆ str          ┆ i64      ┆ str            ┆   ┆ f64      ┆ f64      ┆ f64      ┆ str            │
╞═════════════════════════════════╪══════════════╪══════════╪════════════════╪═══╪══════════╪══════════╪══════════╪════════════════╡
│ 175.45.176.2-149.171.126.16-23… ┆ 175.45.176.2 ┆ 23357    ┆ 149.171.126.16 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Exploits       │
│ 175.45.176.0-149.171.126.16-13… ┆ 175.45.176.0 ┆ 13284    ┆ 149.171.126.16 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Reconnaissance │
│ 175.45.176.2-149.171.126.16-13… ┆ 175.45.176.2 ┆ 13792    ┆ 149.171.126.16 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Exploits       │
│ 175.45.176.0-149.171.126.15-39… ┆ 175.45.176.0 ┆ 39500    ┆ 149.171.126.15 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ DoS            │
│ 175.45.176.0-149.171.126.14-29… ┆ 175.45.176.0 ┆ 29309    ┆ 149.171.126.14 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Generic        │
└─────────────────────────────────┴──────────────┴──────────┴────────────────┴───┴──────────┴──────────┴──────────┴────────────────┘

🔹 Number of features (columns): 84
🔹 Feature names: ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

🔹 Null values per column:
shape: (84, 2)
┌───────────┬──────────┐
│ column    ┆ column_0 │
│ ---       ┆ ---      │
│ str       ┆ u32      │
╞═══════════╪══════════╡
│ Flow ID   ┆ 0        │
│ Src IP    ┆ 0        │
│ Src Port  ┆ 0        │
│ Dst IP    ┆ 0        │
│ Dst Port  ┆ 0        │
│ …         ┆ …        │
│ Idle Mean ┆ 0        │
│ Idle Std  ┆ 0        │
│ Idle Max  ┆ 0        │
│ Idle Min  ┆ 0        │
│ Label     ┆ 0        │
└───────────┴──────────┘

🔹 Summary statistics:
shape: (9, 85)
┌────────────┬─────────────────────────────────┬─────────────┬──────────────┬───┬───────────────┬──────────────┬──────────────┬──────────┐
│ statistic  ┆ Flow ID                         ┆ Src IP      ┆ Src Port     ┆ … ┆ Idle Std      ┆ Idle Max     ┆ Idle Min     ┆ Label    │
│ ---        ┆ ---                             ┆ ---         ┆ ---          ┆   ┆ ---           ┆ ---          ┆ ---          ┆ ---      │
│ str        ┆ str                             ┆ str         ┆ f64          ┆   ┆ f64           ┆ f64          ┆ f64          ┆ str      │
╞════════════╪═════════════════════════════════╪═════════════╪══════════════╪═══╪═══════════════╪══════════════╪══════════════╪══════════╡
│ count      ┆ 3540241                         ┆ 3540241     ┆ 3.540241e6   ┆ … ┆ 3.540241e6    ┆ 3.540241e6   ┆ 3.540241e6   ┆ 3540241  │
│ null_count ┆ 0                               ┆ 0           ┆ 0.0          ┆ … ┆ 0.0           ┆ 0.0          ┆ 0.0          ┆ 0        │
│ mean       ┆ null                            ┆ null        ┆ 31875.725738 ┆ … ┆ 1825.652059   ┆ 48661.718928 ┆ 45528.574833 ┆ null     │
│ std        ┆ null                            ┆ null        ┆ 19489.018961 ┆ … ┆ 111509.706745 ┆ 1.3099e6     ┆ 1.2784e6     ┆ null     │
│ min        ┆ 10.40.182.1-10.40.198.10-33265… ┆ 10.40.182.1 ┆ 10.0         ┆ … ┆ 0.0           ┆ 0.0          ┆ 0.0          ┆ Analysis │
│ 25%        ┆ null                            ┆ null        ┆ 14729.0      ┆ … ┆ 0.0           ┆ 0.0          ┆ 0.0          ┆ null     │
│ 50%        ┆ null                            ┆ null        ┆ 31789.0      ┆ … ┆ 0.0           ┆ 0.0          ┆ 0.0          ┆ null     │
│ 75%        ┆ null                            ┆ null        ┆ 48846.0      ┆ … ┆ 0.0           ┆ 0.0          ┆ 0.0          ┆ null     │
│ max        ┆ 59.166.0.9-149.171.126.9-9998-… ┆ 59.166.0.9  ┆ 65535.0      ┆ … ┆ 7.2144e7      ┆ 1.19882844e8 ┆ 1.19882844e8 ┆ Worms    │
└────────────┴─────────────────────────────────┴─────────────┴──────────────┴───┴───────────────┴──────────────┴──────────────┴──────────┘

🔹 Example categorical counts (top 10 for each string column):

Column: Flow ID
shape: (10, 2)
┌─────────────────────────────────┬───────┐
│ Flow ID                         ┆ count │
│ ---                             ┆ ---   │
│ str                             ┆ u32   │
╞═════════════════════════════════╪═══════╡
│ 10.40.182.6-10.40.182.255-138-… ┆ 60    │
│ 175.45.176.1-149.171.126.18-47… ┆ 58    │
│ 175.45.176.1-149.171.126.18-10… ┆ 57    │
│ 149.171.126.18-175.45.176.1-10… ┆ 57    │
│ 149.171.126.18-175.45.176.1-47… ┆ 57    │
│ 175.45.176.0-149.171.126.10-10… ┆ 33    │
│ 175.45.176.0-149.171.126.10-47… ┆ 33    │
│ 175.45.176.3-149.171.126.15-47… ┆ 33    │
│ 149.171.126.15-175.45.176.3-47… ┆ 33    │
│ 175.45.176.3-149.171.126.15-10… ┆ 33    │
└─────────────────────────────────┴───────┘

Column: Src IP
shape: (10, 2)
┌────────────┬────────┐
│ Src IP     ┆ count  │
│ ---        ┆ ---    │
│ str        ┆ u32    │
╞════════════╪════════╡
│ 59.166.0.1 ┆ 328460 │
│ 59.166.0.4 ┆ 328435 │
│ 59.166.0.0 ┆ 328163 │
│ 59.166.0.2 ┆ 328048 │
│ 59.166.0.5 ┆ 327836 │
│ 59.166.0.3 ┆ 325670 │
│ 59.166.0.9 ┆ 315669 │
│ 59.166.0.8 ┆ 314465 │
│ 59.166.0.6 ┆ 314298 │
│ 59.166.0.7 ┆ 313893 │
└────────────┴────────┘

Column: Dst IP
shape: (10, 2)
┌───────────────┬────────┐
│ Dst IP        ┆ count  │
│ ---           ┆ ---    │
│ str           ┆ u32    │
╞═══════════════╪════════╡
│ 149.171.126.4 ┆ 328153 │
│ 149.171.126.3 ┆ 328153 │
│ 149.171.126.1 ┆ 328086 │
│ 149.171.126.2 ┆ 328072 │
│ 149.171.126.0 ┆ 326749 │
│ 149.171.126.5 ┆ 326574 │
│ 149.171.126.7 ┆ 316126 │
│ 149.171.126.9 ┆ 315782 │
│ 149.171.126.6 ┆ 315413 │
│ 149.171.126.8 ┆ 311829 │
└───────────────┴────────┘

Column: Timestamp
shape: (10, 2)
┌────────────────────────┬───────┐
│ Timestamp              ┆ count │
│ ---                    ┆ ---   │
│ str                    ┆ u32   │
╞════════════════════════╪═══════╡
│ 22/01/2015 07:50:43 AM ┆ 776   │
│ 17/02/2015 08:38:09 PM ┆ 396   │
│ 22/01/2015 07:50:47 AM ┆ 198   │
│ 17/02/2015 08:38:10 PM ┆ 191   │
│ 17/02/2015 08:38:08 PM ┆ 179   │
│ 22/01/2015 12:46:39 PM ┆ 177   │
│ 17/02/2015 08:38:12 PM ┆ 174   │
│ 17/02/2015 08:38:13 PM ┆ 173   │
│ 22/01/2015 12:53:08 PM ┆ 168   │
│ 22/01/2015 12:51:03 PM ┆ 165   │
└────────────────────────┴───────┘

Column: Label
shape: (10, 2)
┌────────────────┬─────────┐
│ Label          ┆ count   │
│ ---            ┆ ---     │
│ str            ┆ u32     │
╞════════════════╪═════════╡
│ Benign         ┆ 3450658 │
│ Exploits       ┆ 30951   │
│ Fuzzers        ┆ 29613   │
│ Reconnaissance ┆ 16735   │
│ Generic        ┆ 4632    │
│ DoS            ┆ 4467    │
│ Shellcode      ┆ 2102    │
│ Backdoor       ┆ 452     │
│ Analysis       ┆ 385     │
│ Worms          ┆ 246     │
└────────────────┴─────────┘

🔹 Chunked processing example (first 2 chunks):

Chunk 1: shape = (100000, 84)
shape: (3, 84)
┌─────────────────────────────────┬──────────────┬──────────┬────────────────┬───┬──────────┬──────────┬──────────┬────────────────┐
│ Flow ID                         ┆ Src IP       ┆ Src Port ┆ Dst IP         ┆ … ┆ Idle Std ┆ Idle Max ┆ Idle Min ┆ Label          │
│ ---                             ┆ ---          ┆ ---      ┆ ---            ┆   ┆ ---      ┆ ---      ┆ ---      ┆ ---            │
│ str                             ┆ str          ┆ i64      ┆ str            ┆   ┆ f64      ┆ f64      ┆ f64      ┆ str            │
╞═════════════════════════════════╪══════════════╪══════════╪════════════════╪═══╪══════════╪══════════╪══════════╪════════════════╡
│ 175.45.176.2-149.171.126.16-23… ┆ 175.45.176.2 ┆ 23357    ┆ 149.171.126.16 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Exploits       │
│ 175.45.176.0-149.171.126.16-13… ┆ 175.45.176.0 ┆ 13284    ┆ 149.171.126.16 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Reconnaissance │
│ 175.45.176.2-149.171.126.16-13… ┆ 175.45.176.2 ┆ 13792    ┆ 149.171.126.16 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Exploits       │
└─────────────────────────────────┴──────────────┴──────────┴────────────────┴───┴──────────┴──────────┴──────────┴────────────────┘

Chunk 2: shape = (100000, 84)
shape: (3, 84)
┌─────────────────────────────────┬────────────┬──────────┬───────────────┬───┬──────────┬──────────┬──────────┬────────┐
│ Flow ID                         ┆ Src IP     ┆ Src Port ┆ Dst IP        ┆ … ┆ Idle Std ┆ Idle Max ┆ Idle Min ┆ Label  │
│ ---                             ┆ ---        ┆ ---      ┆ ---           ┆   ┆ ---      ┆ ---      ┆ ---      ┆ ---    │
│ str                             ┆ str        ┆ i64      ┆ str           ┆   ┆ f64      ┆ f64      ┆ f64      ┆ str    │
╞═════════════════════════════════╪════════════╪══════════╪═══════════════╪═══╪══════════╪══════════╪══════════╪════════╡
│ 59.166.0.5-149.171.126.9-16578… ┆ 59.166.0.5 ┆ 16578    ┆ 149.171.126.9 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Benign │
│ 59.166.0.4-149.171.126.1-44046… ┆ 59.166.0.4 ┆ 44046    ┆ 149.171.126.1 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Benign │
│ 59.166.0.5-149.171.126.7-23881… ┆ 59.166.0.5 ┆ 23881    ┆ 149.171.126.7 ┆ … ┆ 0.0      ┆ 0.0      ┆ 0.0      ┆ Benign │
└─────────────────────────────────┴────────────┴──────────┴───────────────┴───┴──────────┴──────────┴──────────┴────────┘

2. Data
🔹 Dataset Preview:
shape: (5, 76)
┌───────────────┬──────────────────┬───────────────────┬────────────────────────────┬───┬───────────┬──────────┬──────────┬──────────┐
│ Flow Duration ┆ Total Fwd Packet ┆ Total Bwd packets ┆ Total Length of Fwd Packet ┆ … ┆ Idle Mean ┆ Idle Std ┆ Idle Max ┆ Idle Min │
│ ---           ┆ ---              ┆ ---               ┆ ---                        ┆   ┆ ---       ┆ ---      ┆ ---      ┆ ---      │
│ i64           ┆ i64              ┆ i64               ┆ f64                        ┆   ┆ f64       ┆ f64      ┆ f64      ┆ f64      │
╞═══════════════╪══════════════════╪═══════════════════╪════════════════════════════╪═══╪═══════════╪══════════╪══════════╪══════════╡
│ 214392        ┆ 9                ┆ 21                ┆ 388.0                      ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
│ 2376792       ┆ 9                ┆ 3                 ┆ 752.0                      ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
│ 131350        ┆ 10               ┆ 3                 ┆ 7564.0                     ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
│ 164796        ┆ 6                ┆ 3                 ┆ 770.0                      ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
│ 163418        ┆ 6                ┆ 3                 ┆ 400.0                      ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
└───────────────┴──────────────────┴───────────────────┴────────────────────────────┴───┴───────────┴──────────┴──────────┴──────────┘

🔹 Number of features (columns): 76
🔹 Feature names: ['Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

🔹 Null values per column:
shape: (76, 2)
┌────────────────────────────┬──────────┐
│ column                     ┆ column_0 │
│ ---                        ┆ ---      │
│ str                        ┆ u32      │
╞════════════════════════════╪══════════╡
│ Flow Duration              ┆ 0        │
│ Total Fwd Packet           ┆ 0        │
│ Total Bwd packets          ┆ 0        │
│ Total Length of Fwd Packet ┆ 0        │
│ Total Length of Bwd Packet ┆ 0        │
│ …                          ┆ …        │
│ Active Min                 ┆ 0        │
│ Idle Mean                  ┆ 0        │
│ Idle Std                   ┆ 0        │
│ Idle Max                   ┆ 0        │
│ Idle Min                   ┆ 0        │
└────────────────────────────┴──────────┘

🔹 Summary statistics:
shape: (9, 77)
┌────────────┬───────────────┬──────────────────┬───────────────────┬───┬──────────────┬───────────────┬───────────────┬──────────────┐
│ statistic  ┆ Flow Duration ┆ Total Fwd Packet ┆ Total Bwd packets ┆ … ┆ Idle Mean    ┆ Idle Std      ┆ Idle Max      ┆ Idle Min     │
│ ---        ┆ ---           ┆ ---              ┆ ---               ┆   ┆ ---          ┆ ---           ┆ ---           ┆ ---          │
│ str        ┆ f64           ┆ f64              ┆ f64               ┆   ┆ f64          ┆ f64           ┆ f64           ┆ f64          │
╞════════════╪═══════════════╪══════════════════╪═══════════════════╪═══╪══════════════╪═══════════════╪═══════════════╪══════════════╡
│ count      ┆ 447915.0      ┆ 447915.0         ┆ 447915.0          ┆ … ┆ 447915.0     ┆ 447915.0      ┆ 447915.0      ┆ 447915.0     │
│ null_count ┆ 0.0           ┆ 0.0              ┆ 0.0               ┆ … ┆ 0.0          ┆ 0.0           ┆ 0.0           ┆ 0.0          │
│ mean       ┆ 598300.316004 ┆ 22.598428        ┆ 27.238503         ┆ … ┆ 98592.835568 ┆ 3341.817362   ┆ 102622.830749 ┆ 95788.535673 │
│ std        ┆ 4.8788e6      ┆ 127.986936       ┆ 116.638803        ┆ … ┆ 2.2228e6     ┆ 220178.212879 ┆ 2.2610e6      ┆ 2.2073e6     │
│ min        ┆ 1.0           ┆ 1.0              ┆ 0.0               ┆ … ┆ 0.0          ┆ 0.0           ┆ 0.0           ┆ 0.0          │
│ 25%        ┆ 349.0         ┆ 1.0              ┆ 2.0               ┆ … ┆ 0.0          ┆ 0.0           ┆ 0.0           ┆ 0.0          │
│ 50%        ┆ 5788.0        ┆ 3.0              ┆ 2.0               ┆ … ┆ 0.0          ┆ 0.0           ┆ 0.0           ┆ 0.0          │
│ 75%        ┆ 179246.0      ┆ 17.0             ┆ 15.0              ┆ … ┆ 0.0          ┆ 0.0           ┆ 0.0           ┆ 0.0          │
│ max        ┆ 1.19997527e8  ┆ 20038.0          ┆ 11021.0           ┆ … ┆ 1.19192568e8 ┆ 7.2144e7      ┆ 1.19192568e8  ┆ 1.19192568e8 │
└────────────┴───────────────┴──────────────────┴───────────────────┴───┴──────────────┴───────────────┴───────────────┴──────────────┘

🔹 Example categorical counts (top 10 for each string column):

🔹 Chunked processing example (first 2 chunks):

Chunk 1: shape = (100000, 76)
shape: (3, 76)
┌───────────────┬──────────────────┬───────────────────┬────────────────────────────┬───┬───────────┬──────────┬──────────┬──────────┐
│ Flow Duration ┆ Total Fwd Packet ┆ Total Bwd packets ┆ Total Length of Fwd Packet ┆ … ┆ Idle Mean ┆ Idle Std ┆ Idle Max ┆ Idle Min │
│ ---           ┆ ---              ┆ ---               ┆ ---                        ┆   ┆ ---       ┆ ---      ┆ ---      ┆ ---      │
│ i64           ┆ i64              ┆ i64               ┆ f64                        ┆   ┆ f64       ┆ f64      ┆ f64      ┆ f64      │
╞═══════════════╪══════════════════╪═══════════════════╪════════════════════════════╪═══╪═══════════╪══════════╪══════════╪══════════╡
│ 214392        ┆ 9                ┆ 21                ┆ 388.0                      ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
│ 2376792       ┆ 9                ┆ 3                 ┆ 752.0                      ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
│ 131350        ┆ 10               ┆ 3                 ┆ 7564.0                     ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
└───────────────┴──────────────────┴───────────────────┴────────────────────────────┴───┴───────────┴──────────┴──────────┴──────────┘

Chunk 2: shape = (100000, 76)
shape: (3, 76)
┌───────────────┬──────────────────┬───────────────────┬────────────────────────────┬───┬───────────┬──────────┬──────────┬──────────┐
│ Flow Duration ┆ Total Fwd Packet ┆ Total Bwd packets ┆ Total Length of Fwd Packet ┆ … ┆ Idle Mean ┆ Idle Std ┆ Idle Max ┆ Idle Min │
│ ---           ┆ ---              ┆ ---               ┆ ---                        ┆   ┆ ---       ┆ ---      ┆ ---      ┆ ---      │
│ i64           ┆ i64              ┆ i64               ┆ f64                        ┆   ┆ f64       ┆ f64      ┆ f64      ┆ f64      │
╞═══════════════╪══════════════════╪═══════════════════╪════════════════════════════╪═══╪═══════════╪══════════╪══════════╪══════════╡
│ 10529         ┆ 32               ┆ 33                ┆ 454.0                      ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
│ 15543         ┆ 42               ┆ 43                ┆ 470.0                      ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
│ 18307         ┆ 34               ┆ 33                ┆ 454.0                      ┆ … ┆ 0.0       ┆ 0.0      ┆ 0.0      ┆ 0.0      │
└───────────────┴──────────────────┴───────────────────┴────────────────────────────┴───┴───────────┴──────────┴──────────┴──────────┘

