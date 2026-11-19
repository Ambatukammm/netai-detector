"""
Live Network Traffic Capture & Feature Extractor
Runs in background — automatically captures and processes traffic windows
"""
import numpy as np
import time
from pathlib import Path
from collections import Counter
import json

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
except ImportError:
    print("[ERROR] scapy not installed — run: pip install scapy")
    exit(1)

# ─── SETTINGS ───
WINDOW_PACKETS = 500          # How many packets per sample
CAPTURE_DURATION = 300        # Seconds to capture for training (5 min)
SAMPLE_INTERVAL = 10          # Seconds between live monitoring samples
DATA_DIR = Path("data")
CAPTURE_DIR = Path("captures")
DATA_DIR.mkdir(exist_ok=True)
CAPTURE_DIR.mkdir(exist_ok=True)

# Your local subnet — UPDATE THIS IF NEEDED!
LOCAL_SUBNET = "192.168."
NUM_FEATURES = 80

# Global buffer for live capture
packet_buffer = []
window_count = 0

# ─── FEATURE EXTRACTION (80 features) ───
def extract_features(packets):
    if not packets:
        return None
    
    pkt_sizes = []
    src_ips = Counter()
    dst_ips = Counter()
    src_ports = Counter()
    dst_ports = Counter()
    protocols = Counter()
    tcp_flags = Counter()
    dir_stats = {'in_in': 0, 'in_out': 0, 'out_in': 0, 'out_out': 0}
    
    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
        ip = pkt[IP]
        pkt_len = len(pkt)
        pkt_sizes.append(pkt_len)
        
        src_ips[ip.src] += 1
        dst_ips[ip.dst] += 1
        
        src_internal = ip.src.startswith(LOCAL_SUBNET)
        dst_internal = ip.dst.startswith(LOCAL_SUBNET)
        
        if src_internal and dst_internal:
            dir_stats['in_in'] += 1
        elif src_internal and not dst_internal:
            dir_stats['in_out'] += 1
        elif not src_internal and dst_internal:
            dir_stats['out_in'] += 1
        else:
            dir_stats['out_out'] += 1
        
        if pkt.haslayer(TCP):
            protocols['TCP'] += 1
            tcp = pkt[TCP]
            src_ports[tcp.sport] += 1
            dst_ports[tcp.dport] += 1
            tcp_flags[str(tcp.flags)] += 1
        elif pkt.haslayer(UDP):
            protocols['UDP'] += 1
            udp = pkt[UDP]
            src_ports[udp.sport] += 1
            dst_ports[udp.dport] += 1
        elif pkt.haslayer(ICMP):
            protocols['ICMP'] += 1
        else:
            protocols['OTHER'] += 1
    
    n = len(pkt_sizes) or 1
    total_p = sum(protocols.values()) or 1
    
    f = []
    # 1-10: Packet size stats
    f.extend([np.mean(pkt_sizes), np.std(pkt_sizes), np.min(pkt_sizes), np.max(pkt_sizes),
              np.median(pkt_sizes), np.percentile(pkt_sizes, 25), np.percentile(pkt_sizes, 75),
              np.sum(pkt_sizes), n, np.var(pkt_sizes)])
    
    # 11-20: IP diversity
    f.extend([len(src_ips), len(dst_ips), len(src_ips)/n, len(dst_ips)/n,
              max(src_ips.values())/n if src_ips else 0,
              max(dst_ips.values())/n if dst_ips else 0,
              len(set(src_ips.keys()) | set(dst_ips.keys())),
              np.mean(list(src_ips.values())) if src_ips else 0,
              np.mean(list(dst_ips.values())) if dst_ips else 0,
              len(src_ips) + len(dst_ips)])
    
    # 21-30: Port diversity
    well_known = sum(1 for p in dst_ports if p <= 1024)
    f.extend([len(src_ports), len(dst_ips), well_known/max(1, len(dst_ports)),
              len([p for p in dst_ports if 1024 < p < 49152])/max(1, len(dst_ports)),
              len([p for p in dst_ports if p >= 49152])/max(1, len(dst_ports)),
              max(dst_ports.values())/n if dst_ports else 0,
              len(src_ports)/n * 1000, len(dst_ports)/n * 1000,
              sum(dst_ports.values()) / max(1, len(dst_ports)),
              len(set(list(src_ports.keys()) + list(dst_ports.keys())))])
    
    # 31-35: Protocol mix
    f.extend([protocols.get('TCP',0)/total_p, protocols.get('UDP',0)/total_p,
              protocols.get('ICMP',0)/total_p, protocols.get('OTHER',0)/total_p, total_p])
    
    # 36-45: Traffic direction
    f.extend([dir_stats['in_in']/n, dir_stats['in_out']/n, dir_stats['out_in']/n,
              dir_stats['out_out']/n, sum(dir_stats.values()),
              dir_stats['in_in'] / max(1, dir_stats['in_out'] + 0.001),
              (dir_stats['in_in'] + dir_stats['in_out'])/n,
              (dir_stats['out_in'] + dir_stats['out_out'])/n,
              dir_stats['in_in'] + dir_stats['in_out'],
              dir_stats['out_in'] + dir_stats['out_out']])
    
    # 46-55: TCP flags
    f.extend([tcp_flags.get('S',0)/total_p, tcp_flags.get('SA',0)/total_p,
              tcp_flags.get('A',0)/total_p, tcp_flags.get('PA',0)/total_p,
              tcp_flags.get('FA',0)/total_p, tcp_flags.get('R',0)/total_p,
              sum(v for k,v in tcp_flags.items() if 'S' in k and 'A' not in k)/total_p,
              sum(v for k,v in tcp_flags.items() if 'R' in k)/total_p,
              len(tcp_flags), total_p/n])
    
    # 56-80: Additional diversity features
    all_ip_pairs = Counter()
    for s in src_ips:
        for d in dst_ips:
            all_ip_pairs[(s,d)] += 1
    top_pair = max(all_ip_pairs.values()) if all_ip_pairs else 0
    f.extend([
        top_pair / n if n else 0, len(all_ip_pairs), len(all_ip_pairs)/n,
        np.mean(list(all_ip_pairs.values())) if all_ip_pairs else 0,
        np.std(list(all_ip_pairs.values())) if all_ip_pairs else 0,
        np.sum(pkt_sizes) / n, np.max(pkt_sizes) - np.min(pkt_sizes),
        np.std(pkt_sizes) / np.mean(pkt_sizes) if np.mean(pkt_sizes) else 0,
        len(src_ips) / n, len(dst_ips) / n,
        np.percentile(pkt_sizes, 10), np.percentile(pkt_sizes, 90),
        np.sum(pkt_sizes) / (n * 1500),  # vs MTU
        len([p for p in pkt_sizes if p < 64]) / n,  # runt packets
        len([p for p in pkt_sizes if p > 1400]) / n,  # full size packets
        np.mean([int(ip.dst.split('.')[-1]) for ip in packets if ip.haslayer(IP)])/255 if any(p.haslayer(IP) for p in packets) else 0.5,
        len(set(dst_ips.keys())) / max(1, len(dst_ips)),
        len(set(src_ips.keys())) / max(1, len(src_ips)),
        protocols.get('TCP',0)/max(1, protocols.get('UDP',0)),
        len(dst_ports) / max(1, len(dst_ips)),
        np.mean(list(dst_ports.values())) if dst_ports else 0,
        np.median(list(dst_ports.values())) if dst_ports else 0,
        len([p for p in dst_ports if p == 53]) / max(1, len(dst_ports)),   # DNS
        len([p for p in dst_ports if p == 80 or p == 443]) / max(1, len(dst_ports)),  # HTTP/HTTPS
        np.sum(pkt_sizes) / max(1, sum(protocols.values()))
    ])
    
    # Ensure exactly 80 features
    f = np.array(f[:NUM_FEATURES], dtype=np.float32)
    if len(f) < NUM_FEATURES:
        f = np.pad(f, (0, NUM_FEATURES - len(f)), mode='constant')
    
    return np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=-1.0)


def packet_callback(pkt):
    """Process every captured packet"""
    global packet_buffer
    packet_buffer.append(pkt)
    
    # When we have enough packets → extract a feature window
    if len(packet_buffer) >= WINDOW_PACKETS:
        feat = extract_features(packet_buffer)
        if feat is not None:
            save_live_sample(feat)
        packet_buffer = []


def save_live_sample(features):
    """Save one feature window to disk"""
    global window_count
    window_count += 1
    out_path = DATA_DIR / f"live_sample_{window_count:04d}.npy"
    np.save(out_path, features)
    print(f"[CAPTURE] Sample {window_count} saved → {out_path}")


def capture_training_data(duration=CAPTURE_DURATION):
    """Capture normal traffic for training"""
    print(f"\n{'='*60}")
    print(f"📡 CAPTURING NORMAL TRAFFIC — {duration} SECONDS")
    print(f"   Use your network normally — browsing, streaming, etc.")
    print(f"   ⏱️  Will auto-stop after {duration//60} minutes")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    all_features = []
    
    def training_callback(pkt):
        nonlocal all_features
        global packet_buffer
        packet_buffer.append(pkt)
        if len(packet_buffer) >= WINDOW_PACKETS:
            feat = extract_features(packet_buffer)
            if feat is not None:
                all_features.append(feat)
                print(f"   Captured: {len(all_features)} windows", end='\r')
            packet_buffer = []
    
    # Start capture
    sniff(prn=training_callback, timeout=duration)
    
    if len(all_features) == 0:
        print("\n[WARNING] No packets captured — check your network interface")
        return None
    
    X = np.array(all_features, dtype=np.float32)
    out_file = DATA_DIR / "normal_traffic.npy"
    np.save(out_file, X)
    print(f"\n✅ Saved {len(X)} samples → {out_file}")
    print(f"   Shape: {X.shape}")
    return X


def start_live_monitor(sample_interval=SAMPLE_INTERVAL):
    """Continuously capture and save samples for live inference"""
    print("\n🔴 LIVE MONITORING STARTED — Press Ctrl+C to stop")
    print(f"   Capturing new sample every {sample_interval} seconds\n")
    
    try:
        sniff(prn=packet_callback)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "train"
    
    if mode == "train":
        capture_training_data()
    elif mode == "monitor":
        start_live_monitor()
    else:
        print("Usage: python live_capture.py [train|monitor]")
        print("  train   → Capture normal traffic for training")
        print("  monitor → Continuous live capture")
