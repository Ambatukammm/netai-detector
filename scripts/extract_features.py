"""
Traffic Feature Extractor
Reads .pcap files → extracts 80 numerical features → saves as .npy for training
"""
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import socket

try:
    from scapy.all import rdpcap, IP, TCP, UDP, ICMP
except ImportError:
    print("[ERROR] scapy not installed → run: pip install scapy")
    exit(1)

# ─── CONFIG ───
NUM_FEATURES = 80
WINDOW_SIZE = 100  # packets per feature window

def extract_features_from_pcap(pcap_path, window_size=WINDOW_SIZE):
    """
    Read a pcap file, slide a window over it,
    compute 80 features for each window
    """
    print(f"[INFO] Reading: {pcap_path}")
    packets = rdpcap(str(pcap_path))
    print(f"[INFO] Total packets loaded: {len(packets)}")
    
    all_features = []
    
    # Process in sliding windows
    for i in range(0, len(packets) - window_size + 1, window_size // 2):
        window = packets[i:i+window_size]
        feat = compute_features(window)
        if feat is not None:
            all_features.append(feat)
    
    if len(all_features) == 0:
        print("[WARNING] No valid windows extracted")
        return None
    
    return np.array(all_features, dtype=np.float32)


def compute_features(packets):
    """
    Compute 80 numerical features from a list of packets
    """
    n = len(packets)
    if n == 0:
        return None
    
    # ─── BASIC STATS ───
    pkt_sizes = []
    src_ips = Counter()
    dst_ips = Counter()
    src_ports = Counter()
    dst_ports = Counter()
    protocols = Counter()
    tcp_flags = Counter()
    is_internal = []
    
    # Define your local subnet — UPDATE THIS TO MATCH YOURS!
    LOCAL_SUBNET = "192.168."
    
    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
        
        ip = pkt[IP]
        pkt_len = len(pkt)
        pkt_sizes.append(pkt_len)
        
        src_ip = ip.src
        dst_ip = ip.dst
        src_ips[src_ip] += 1
        dst_ips[dst_ip] += 1
        
        src_internal = src_ip.startswith(LOCAL_SUBNET)
        dst_internal = dst_ip.startswith(LOCAL_SUBNET)
        is_internal.append((src_internal, dst_internal))
        
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
    
    if not pkt_sizes:
        return None
    
    # ─── BUILD FEATURE VECTOR ───
    f = []
    
    # Packet size distribution (10 features)
    f.extend([
        np.mean(pkt_sizes), np.std(pkt_sizes), np.min(pkt_sizes), np.max(pkt_sizes),
        np.median(pkt_sizes), np.percentile(pkt_sizes, 25), np.percentile(pkt_sizes, 75),
        np.sum(pkt_sizes), len(pkt_sizes), np.var(pkt_sizes)
    ])
    
    # IP diversity (10 features)
    f.extend([
        len(src_ips), len(dst_ips), len(src_ips)/n, len(dst_ips)/n,
        max(src_ips.values())/n if src_ips else 0,
        max(dst_ips.values())/n if dst_ips else 0,
        len(set(src_ips.keys()) | set(dst_ips.keys())),
        np.mean(list(src_ips.values())) if src_ips else 0,
        np.mean(list(dst_ips.values())) if dst_ips else 0,
        len(src_ips) + len(dst_ips)
    ])
    
    # Port diversity (10 features)
    well_known = sum(1 for p in dst_ports if p <= 1024)
    f.extend([
        len(src_ports), len(dst_ports), well_known / max(1, len(dst_ports)),
        len([p for p in dst_ports if p > 1024 and p < 49152]) / max(1, len(dst_ports)),
        len([p for p in dst_ports if p >= 49152]) / max(1, len(dst_ports)),
        max(dst_ports.values())/n if dst_ports else 0,
        len(src_ports)/n * 1000, len(dst_ports)/n * 1000,
        sum(dst_ports.values()) / max(1, len(src_ports)),
        len(set(list(src_ports.keys()) + list(dst_ports.keys())))
    ])
    
    # Protocol mix (5 features)
    total_p = sum(protocols.values()) or 1
    f.extend([
        protocols.get('TCP', 0)/total_p, protocols.get('UDP', 0)/total_p,
        protocols.get('ICMP', 0)/total_p, protocols.get('OTHER', 0)/total_p,
        total_p
    ])
    
    # Traffic direction — internal vs external (10 features)
    in_in = sum(1 for s,d in is_internal if s and d)
    in_out = sum(1 for s,d in is_internal if s and not d)
    out_in = sum(1 for s,d in is_internal if not s and d)
    out_out = sum(1 for s,d in is_internal if not s and not d)
    f.extend([
        in_in/n, in_out/n, out_in/n, out_out/n,
        in_in + in_out + out_in + out_out,
        in_in / max(1, in_out + 0.001),
        (in_in + in_out) / n, (out_in + out_out) / n,
        len([1 for s,d in is_internal if s]),
        len([1 for s,d in is_internal if d])
    ])
    
    # TCP flag patterns (10 features)
    flag_counts = Counter({k:v for k,v in tcp_flags.items()})
    total_tcp = sum(tcp_flags.values()) or 1
    f.extend([
        tcp_flags.get('S', 0)/total_tcp, tcp_flags.get('SA', 0)/total_tcp,
        tcp_flags.get('A', 0)/total_tcp, tcp_flags.get('PA', 0)/total_tcp,
        tcp_flags.get('FA', 0)/total_tcp, tcp_flags.get('R', 0)/total_tcp,
        sum(v for k,v in tcp_flags.items() if 'S' in k and 'A' not in k)/total_tcp,
        sum(v for k,v in tcp_flags.items() if 'R' in k)/total_tcp,
        len(tcp_flags), total_tcp/n
    ])
    
    # Concentrate diversity metrics into remaining features (25 features → total 80)
    all_ip_pairs = Counter()
    for s in src_ips:
        for d in dst_ips:
            all_ip_pairs[(s,d)] += 1
    top_ip_pair_count = max(all_ip_pairs.values()) if all_ip_pairs else 0
    f.extend([
        top_ip_pair_count / n, len(all_ip_pairs), len(all_ip_pairs)/n,
        np.mean(list(all_ip_pairs.values())) if all_ip_pairs else 0,
        np.std(list(all_ip_pairs.values())) if all_ip_pairs else 0
    ])
    
    # Pad or truncate to exactly NUM_FEATURES
    if len(f) < NUM_FEATURES:
        f = np.pad(f, (0, NUM_FEATURES - len(f)), mode='constant')
    else:
        f = np.array(f[:NUM_FEATURES])
    
    # Replace inf/nan
    f = np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return f


def save_features(features, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    print(f"✅ Saved {len(features)} windows → {output_path}")
    print(f"   Feature shape: {features.shape}")


# ─── RUN FROM COMMAND LINE ───
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py path/to/traffic.pcap [output.npy]")
        print("\nExample:")
        print("  python extract_features.py captures/normal.pcap data/normal_traffic.npy")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "traffic_features.npy"
    
    if not Path(pcap_file).exists():
        print(f"[ERROR] File not found: {pcap_file}")
        sys.exit(1)
    
    features = extract_features_from_pcap(pcap_file)
    if features is not None:
        save_features(features, out_file)
