# Security Labs Portfolio

Hands-on lab work from my B.Tech in Computer Science (Information Security) at VIT Vellore, 2022–2026. Each folder holds the lab reports (PDF, with screenshots) and a README explaining what was done and which tools were used.

| Area | What's covered | Tools |
|---|---|---|
| [Digital Forensics](digital-forensics/) | Disk image investigation, FTK Imager, packet analysis, ProDiscover, email header forensics | Autopsy, FTK Imager, ProDiscover, Wireshark, Aid4Mail, MXToolbox |
| [Malware Analysis](malware-analysis/) | Static + dynamic analysis of Zeus, IOC extraction from public samples, ANY.RUN behaviour analysis, IDA and x64dbg reversing | REMnux, FLARE VM, VirusTotal, PeStudio, FLOSS, capa, Cutter, INetSim, Procmon, IDA Free, x64dbg |
| [Penetration Testing](penetration-testing/) | Nmap scanning, Windows enumeration, MD5 cracking, ARP and DNS traffic analysis, blind SQL injection | Nmap, Kali Linux, Metasploitable2, Wireshark, Python, Burp-style web labs |
| [Steganography / Watermarking](steganography-watermarking/) | Team course report on LSB, DCT and DWT data hiding | Python |
| [Image Watermarking Robustness](image-watermarking-robustness/) | stegmark: working LSB / DCT / DWT tool that hides text in any image and recovers it after JPEG, noise and resizing; Reed-Solomon, AES-GCM, CLI, 56 tests, measured benchmark | Python, NumPy, SciPy, PyWavelets |

## Notes
- All work was done in isolated lab VMs against intentionally vulnerable or provided targets, for educational purposes only.
- No malware samples, disk images or packet captures are stored in this repo.
- These are course submissions written up from screenshots and notes. Where a lab was a group project, the README says so.
