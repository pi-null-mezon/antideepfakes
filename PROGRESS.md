Finetune effnet_v2_s(IMNET) on crops from DF-Celebs-V2 + FaceForensics++ + Kaggle's dfdc: (APCER05+BPCER05)/2 around 0.11 

> 10.11.2023 23:00

SEQUENCE PROCESSOR TEST for sequence length: 10 encoder@effnet_v2_s@256x60x0.1.jit
 - EER: 0.0644 (score: 0.5750)
 - BPCER@0.1: 0.0234
 - BPCER@0.01: 0.4507
 - BPCER@0.001: 0.9475

NAIVE AVERAGING TEST for sequence length: 10 effnet_v2_s@256x60x0.1.jit
 - EER: 0.0852 (score: 0.4440)
 - BPCER@0.1: 0.0609
 - BPCER@0.01: 0.6140
 - BPCER@0.001: 0.9446

SEQUENCE PROCESSOR TEST for sequence length: 8 encoder@effnet_v2_s@224x90x0.2.jit
 - EER: 0.0880 (score: 0.5360)
 - BPCER@0.1: 0.0661
 - BPCER@0.01: 0.6521
 - BPCER@0.001: 0.8876

NAIVE AVERAGING TEST for sequence length: 8 effnet_v2_s@224x90x0.2.jit
 - EER: 0.1394 (score: 0.3120)
 - BPCER@0.1: 0.1970
 - BPCER@0.01: 0.7299
 - BPCER@0.001: 0.9002

> 11.11.2023 02:00

SEQUENCE PROCESSOR TEST for sequence length: 10 frames + encoder@effnet_v2_s@256x60x0.1.jit
 - EER: 0.0666 (score: 0.7630)
 - BPCER@0.1: 0.0139
 - BPCER@0.01: 0.4394
 - BPCER@0.001: 0.8305

NAIVE AVERAGING TEST for sequence length: 10 frames + effnet_v2_s@256x60x0.1.jit
 - EER: 0.0807 (score: 0.4530)
 - BPCER@0.1: 0.0533
 - BPCER@0.01: 0.6033
 - BPCER@0.001: 0.9397

SEQUENCE PROCESSOR TEST for sequence length: 10 frames + encoder@effnet_v2_s@224x90x0.2.jit
 - EER: 0.0819 (score: 0.7480)
 - BPCER@0.1: 0.0447
 - BPCER@0.01: 0.4917
 - BPCER@0.001: 0.8083

NAIVE AVERAGING TEST for sequence length: 10 frames + effnet_v2_s@224x90x0.2.jit
 - EER: 0.1368 (score: 0.3260)
 - BPCER@0.1: 0.1969
 - BPCER@0.01: 0.7503
 - BPCER@0.001: 0.8886

NAIVE AVERAGING TEST for seq.length 10 + resnext50.pth:
 - EER: 0.0979 (score: 0.3560)
 - BPCER@0.1: 0.0972
 - BPCER@0.01: 0.4877
 - BPCER@0.001: 0.9770

> 21.11.2023 21:30 (datasets were updated)

FIRST COMMIT VERSION SEQUENCE PROCESSOR TEST for precision: fp16, seq.length: 10 frames
 - EER: 0.1305 (score: 0.9430)
 - BPCER@0.1: 0.2299
 - BPCER@0.01: 0.8684
 - BPCER@0.001: 0.9674

--- 

NAIVE AVERAGING TEST for precision: fp16, seq.length: 10 frames, effnet_v2_s@256x60x0.1.pth
 - EER: 0.0359 (score: 0.3540)
 - BPCER@0.1: 0.0151
 - BPCER@0.01: 0.1203
 - BPCER@0.001: 0.9098

SEQUENCE PROCESSOR TEST for precision: fp16, seq.length: 10 frames, encoder@effnet_v2_s@256x60x0.1.jit
 - EER: 0.0298 (score: 0.3420)
 - BPCER@0.1: 0.0080
 - BPCER@0.01: 0.1160
 - BPCER@0.001: 0.8462

SEQUENCE PROCESSOR TEST for precision: fp16, seq.length: 10 frames, encoder@effnet_v2_s@256x60x0.1.jit
 - EER: 0.0266 (score: 0.2960)
 - BPCER@0.1: 0.0071
 - BPCER@0.01: 0.1004
 - BPCER@0.001: 0.9172

SEQUENCE PROCESSOR TEST for precision: fp16, seq.length: 10 frames, encoder@effnet_v2_s@256x60x0.1.jit
 - EER: 0.0275 (score: 0.4130)
 - BPCER@0.1: 0.0069
 - BPCER@0.01: 0.0928
 - BPCER@0.001: 0.8758

---

SEQUENCE PROCESSOR TEST for precision: fp16, seq.length: 10 frames, encoder@resnext50@256x60x0.1_v1.jit
 - EER: 0.0703 (score: 0.7650)
 - BPCER@0.1: 0.0514
 - BPCER@0.01: 0.3517
 - BPCER@0.001: 0.7723

SEQUENCE PROCESSOR TEST for precision: fp16, seq.length: 10 frames, encoder@resnext50@256x60x0.1_v2.jit
 - EER: 0.0708 (score: 0.6940)
 - BPCER@0.1: 0.0473
 - BPCER@0.01: 0.5973
 - BPCER@0.001: 0.9476

SEQUENCE PROCESSOR TEST for precision: fp16, seq.length: 10 frames, encoder@resnext50@256x60x0.1.jit
 - EER: 0.0547 (score: 0.6320)
 - BPCER@0.1: 0.0343
 - BPCER@0.01: 0.1300
 - BPCER@0.001: 0.7339
