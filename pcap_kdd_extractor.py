#!/usr/bin/env python3
"""
PCAP -> KDD Cup 99 style feature extractor (best-effort from packet data)

Usage:
    python pcap_kdd_extractor.py input.pcap output.csv

Notes & Assumptions:
- Builds bidirectional TCP/UDP/ICMP flows keyed by 5-tuple (src, dst, sport, dport, proto).
- Approximates KDD "flag" (S0,S1,SF,REJ,RSTO,RSTR,S2,S3,SH) from TCP handshake/teardown heuristics.
- Computes 2-second time-based stats ("count", "srv_count", ...).
- Computes host-based stats within a sliding window of the last 100 connections *to the same dst host*
  as per KDD definition (best effort).
- Content features that require application semantics are set to 0: hot, num_failed_logins, logged_in,
  num_compromised, root_shell, su_attempted, num_root, num_file_creations, num_shells, num_access_files,
  num_outbound_cmds, is_host_login, is_guest_login.
- rerror/serror rates are inferred from presence of RST or ICMP errors when initiating/within a connection.
- Service is inferred from dport for TCP/UDP using a small port map; unknown -> "other".
"""
import sys
import csv
import socket
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Tuple, List, Dict, Deque, Optional, Iterable

try:
    import dpkt
    from dpkt import icmp6 as dpkt_icmp6
except Exception as exc:
    print("This script requires dpkt. Install with `pip install dpkt`.", file=sys.stderr)
    raise

ICMP6_MESSAGE_CLASSES: Tuple[type, ...] = tuple(
    cls
    for name in ("ICMP6", "ICMP6Packet")
    if isinstance(getattr(dpkt_icmp6, name, None), type)
    for cls in (getattr(dpkt_icmp6, name),)
)
ICMP6_DEST_UNREACH_CODES = tuple(
    code
    for code in (
        getattr(dpkt_icmp6, "ICMP6_DEST_UNREACH", None),
        getattr(dpkt_icmp6, "ICMP6_DST_UNREACH", None),
    )
    if isinstance(code, int)
)
if not ICMP6_DEST_UNREACH_CODES:
    ICMP6_DEST_UNREACH_CODES = (1,)

# --- Helpers ---

SERVICE_PORTS = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain", 67: "dhcp", 68: "dhcp", 69: "tftp",
    80: "http", 110: "pop3", 111: "rpc", 113: "auth", 119: "nntp",
    123: "ntp", 135: "msrpc", 137: "netbios_ns", 138: "netbios_dgm", 139: "netbios_ssn",
    143: "imap", 161: "snmp", 162: "snmptrap", 389: "ldap",
    443: "https", 445: "microsoft_ds", 993: "imaps", 995: "pop3s",
    1433: "ms_sql", 1521: "oracle", 2049: "nfs", 3306: "mysql",
    3389: "rdp", 5060: "sip", 8080: "http_alt", 8443: "https_alt"
}

def service_name(proto: str, dport: Optional[int]) -> str:
    if proto not in ("tcp", "udp"):
        return "other"
    if dport is None:
        return "other"
    return SERVICE_PORTS.get(dport, "other")

@dataclass
class Flow:
    src: str
    dst: str
    sport: Optional[int]
    dport: Optional[int]
    proto: str
    start: float
    end: float
    src_bytes: int = 0
    dst_bytes: int = 0
    land: int = 0
    wrong_fragment: int = 0
    urgent_src: int = 0
    urgent_dst: int = 0
    syn: bool = False
    syn_ack: bool = False
    ack_after_syn: bool = False
    fin: bool = False
    rst_from_srv: bool = False
    rst_from_cli: bool = False
    icmp_error: bool = False
    packets: int = 0

    def update(
        self,
        ts: float,
        length: int,
        direction_src_to_dst: bool,
        tcp_flags: Optional[int] = None,
        wrong_fragment: bool = False,
        icmp_error: bool = False,
    ) -> None:
        self.end = ts
        self.packets += 1
        if wrong_fragment:
            self.wrong_fragment += 1
        if icmp_error:
            self.icmp_error = True
        if self.proto == "tcp" and tcp_flags is not None:
            flags = tcp_flags
            if flags & 0x02:  # SYN
                if direction_src_to_dst:
                    self.syn = True
                else:
                    self.syn_ack = True
            if flags & 0x10 and (self.syn or self.syn_ack):
                self.ack_after_syn = True
            if flags & 0x01:
                self.fin = True
            if flags & 0x04:  # RST
                if direction_src_to_dst:
                    self.rst_from_cli = True
                else:
                    self.rst_from_srv = True
            if flags & 0x20:  # URG flag
                if direction_src_to_dst:
                    self.urgent_src += 1
                else:
                    self.urgent_dst += 1
        if direction_src_to_dst:
            self.src_bytes += length
        else:
            self.dst_bytes += length

def infer_flag(flow: Flow) -> str:
    """Approximate KDD 'flag' categorical value from TCP behavior."""
    if flow.proto != "tcp":
        return "OTH"
    # SYN sent, no response
    if flow.syn and not flow.syn_ack and not flow.rst_from_srv and not flow.fin:
        return "S0"
    # SYN -> RST from server (connection refused)
    if flow.syn and flow.rst_from_srv and not flow.syn_ack:
        return "REJ"
    # Established and closed normally
    if flow.syn and flow.syn_ack and flow.ack_after_syn and flow.fin and not (flow.rst_from_cli or flow.rst_from_srv):
        return "SF"
    # RST from client without handshake completion
    if flow.rst_from_cli and not flow.syn_ack:
        return "RSTO"
    # RST from server after established
    if flow.syn_ack and (flow.rst_from_srv or flow.rst_from_cli):
        return "RSTR"
    # SYN seen both ways but no FIN/RST
    if flow.syn and flow.syn_ack and not (flow.fin or flow.rst_from_cli or flow.rst_from_srv):
        return "S1"
    # SYN and data only one direction
    if flow.syn and not flow.ack_after_syn and (flow.fin or flow.rst_from_cli or flow.rst_from_srv):
        return "S2"
    return "OTH"

@dataclass
class ConnRecord:
    """Per-connection finalized record with features."""
    start: float
    end: float
    src: str
    dst: str
    sport: Optional[int]
    dport: Optional[int]
    proto: str
    service: str
    flag: str
    src_bytes: int
    dst_bytes: int
    land: int
    wrong_fragment: int
    urgent: int
    icmp_error: bool = False
    # Time-based window stats (2s prior window)
    count: int = 0
    srv_count: int = 0
    serror_rate: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    srv_rerror_rate: float = 0.0
    same_srv_rate: float = 0.0
    diff_srv_rate: float = 0.0
    srv_diff_host_rate: float = 0.0
    # Host-based window stats (last 100 to same dst host)
    dst_host_count: int = 0
    dst_host_srv_count: int = 0
    dst_host_same_srv_rate: float = 0.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    dst_host_srv_rerror_rate: float = 0.0


@dataclass
class PacketRecord:
    ts: float
    src: str
    dst: str
    proto: str
    length: int
    sport: Optional[int] = None
    dport: Optional[int] = None
    tcp_flags: Optional[int] = None
    wrong_fragment: bool = False
    icmp_error: bool = False


SERROR_FLAGS = {"S0", "REJ", "RSTR", "RSTO", "S2", "OTH"}
RERROR_FLAGS = {"REJ", "RSTR", "RSTO"}


def _inet_to_str(addr: bytes) -> Optional[str]:
    try:
        if len(addr) == 4:
            return socket.inet_ntop(socket.AF_INET, addr)
        if len(addr) == 16:
            return socket.inet_ntop(socket.AF_INET6, addr)
    except (ValueError, OSError):
        return None
    return None


def _iter_packets(in_pcap: str) -> Iterable[PacketRecord]:
    with open(in_pcap, "rb") as fh:
        try:
            iterator = dpkt.pcap.Reader(fh)
        except (ValueError, dpkt.dpkt.NeedData):
            fh.seek(0)
            iterator = dpkt.pcapng.Reader(fh)

        for record in iterator:
            if not isinstance(record, tuple) or len(record) < 2:
                continue
            ts, buf = record[0], record[1]
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except (dpkt.UnpackError, dpkt.dpkt.NeedData):
                continue
            ip_pkt = eth.data
            src = dst = None
            wrong_fragment = False
            payload = None

            if isinstance(ip_pkt, dpkt.ip.IP):
                src = _inet_to_str(ip_pkt.src)
                dst = _inet_to_str(ip_pkt.dst)
                wrong_fragment = bool(ip_pkt.off & (dpkt.ip.IP_MF | dpkt.ip.IP_OFFMASK))
                payload = ip_pkt.data
            elif isinstance(ip_pkt, dpkt.ip6.IP6):
                src = _inet_to_str(ip_pkt.src)
                dst = _inet_to_str(ip_pkt.dst)
                payload = ip_pkt.data
            else:
                continue

            if src is None or dst is None:
                continue

            proto = "other"
            sport = dport = None
            tcp_flags = None
            icmp_error = False

            if isinstance(payload, dpkt.tcp.TCP):
                proto = "tcp"
                sport = int(payload.sport)
                dport = int(payload.dport)
                tcp_flags = int(payload.flags)
            elif isinstance(payload, dpkt.udp.UDP):
                proto = "udp"
                sport = int(payload.sport)
                dport = int(payload.dport)
            elif isinstance(payload, dpkt.icmp.ICMP):
                proto = "icmp"
                if payload.type == 3:
                    icmp_error = True
            elif ICMP6_MESSAGE_CLASSES and isinstance(payload, ICMP6_MESSAGE_CLASSES):
                proto = "icmp"
                if getattr(payload, "type", None) in ICMP6_DEST_UNREACH_CODES:
                    icmp_error = True
            elif payload.__class__.__module__.startswith("dpkt.icmp6"):
                proto = "icmp"
                if getattr(payload, "type", None) in ICMP6_DEST_UNREACH_CODES:
                    icmp_error = True

            yield PacketRecord(
                ts=float(ts),
                src=src,
                dst=dst,
                proto=proto,
                length=len(buf),
                sport=sport,
                dport=dport,
                tcp_flags=tcp_flags,
                wrong_fragment=wrong_fragment,
                icmp_error=icmp_error,
            )


def _is_serror_record(record: "ConnRecord") -> bool:
    return record.flag in SERROR_FLAGS or record.icmp_error


def _is_rerror_record(record: "ConnRecord") -> bool:
    return record.flag in RERROR_FLAGS or record.icmp_error


class _SrcWindowState:
    __slots__ = (
        "records",
        "service_counts",
        "service_dst_counts",
        "serror_total",
        "rerror_total",
        "service_serror",
        "service_rerror",
    )

    def __init__(self) -> None:
        self.records = deque()
        self.service_counts: Dict[str, int] = {}
        self.service_dst_counts: Dict[str, Dict[str, int]] = {}
        self.serror_total = 0
        self.rerror_total = 0
        self.service_serror: Dict[str, int] = {}
        self.service_rerror: Dict[str, int] = {}


def _update_counter(counter: Dict[str, int], key: str, delta: int) -> None:
    if not delta:
        return
    new_val = counter.get(key, 0) + delta
    if new_val <= 0:
        counter.pop(key, None)
    else:
        counter[key] = new_val


def _src_window_update_counts(state: _SrcWindowState, record: "ConnRecord", delta: int) -> None:
    svc = record.service
    dst = record.dst

    _update_counter(state.service_counts, svc, delta)

    if delta > 0:
        dst_counts = state.service_dst_counts.setdefault(svc, {})
    else:
        dst_counts = state.service_dst_counts.get(svc)

    if dst_counts is not None:
        _update_counter(dst_counts, dst, delta)
        if not dst_counts:
            state.service_dst_counts.pop(svc, None)

    if _is_serror_record(record):
        state.serror_total = max(0, state.serror_total + delta)
        _update_counter(state.service_serror, svc, delta)

    if _is_rerror_record(record):
        state.rerror_total = max(0, state.rerror_total + delta)
        _update_counter(state.service_rerror, svc, delta)


def _src_window_add(state: _SrcWindowState, record: "ConnRecord") -> None:
    state.records.append(record)
    _src_window_update_counts(state, record, 1)


def _src_window_evict(state: _SrcWindowState, cutoff: float) -> None:
    records = state.records
    while records and records[0].start < cutoff:
        old = records.popleft()
        _src_window_update_counts(state, old, -1)


class _DstWindowState:
    __slots__ = (
        "records",
        "service_counts",
        "service_src_counts",
        "src_port_counts",
        "serror_total",
        "rerror_total",
        "service_serror",
        "service_rerror",
    )

    def __init__(self) -> None:
        self.records = deque()
        self.service_counts: Dict[str, int] = {}
        self.service_src_counts: Dict[str, Dict[str, int]] = {}
        self.src_port_counts: Dict[Optional[int], int] = {}
        self.serror_total = 0
        self.rerror_total = 0
        self.service_serror: Dict[str, int] = {}
        self.service_rerror: Dict[str, int] = {}


def _dst_window_update_counts(state: _DstWindowState, record: "ConnRecord", delta: int) -> None:
    svc = record.service
    src = record.src
    sport = record.sport

    _update_counter(state.service_counts, svc, delta)

    if delta > 0:
        src_counts = state.service_src_counts.setdefault(svc, {})
    else:
        src_counts = state.service_src_counts.get(svc)

    if src_counts is not None:
        _update_counter(src_counts, src, delta)
        if not src_counts:
            state.service_src_counts.pop(svc, None)

    _update_counter(state.src_port_counts, sport, delta)

    if _is_serror_record(record):
        state.serror_total = max(0, state.serror_total + delta)
        _update_counter(state.service_serror, svc, delta)

    if _is_rerror_record(record):
        state.rerror_total = max(0, state.rerror_total + delta)
        _update_counter(state.service_rerror, svc, delta)


def _dst_window_trim(state: _DstWindowState) -> None:
    records = state.records
    while records and len(records) >= 100:
        old = records.popleft()
        _dst_window_update_counts(state, old, -1)


def _dst_window_add(state: _DstWindowState, record: "ConnRecord") -> None:
    state.records.append(record)
    _dst_window_update_counts(state, record, 1)


def process_pcap(in_pcap: str) -> List[ConnRecord]:
    # Build flows
    flows: Dict[Tuple, Flow] = {}
    for packet in _iter_packets(in_pcap):
        ts = packet.ts
        src = packet.src
        dst = packet.dst
        prt = packet.proto
        sport = packet.sport
        dport = packet.dport

        key = (src, dst, sport, dport, prt)
        rkey = (dst, src, dport, sport, prt)
        if key not in flows and rkey not in flows:
            f = Flow(src=src, dst=dst, sport=sport, dport=dport, proto=prt, start=ts, end=ts)
            f.land = int(src == dst and (sport == dport if sport is not None and dport is not None else False))
            flows[key] = f
        if key in flows:
            flow = flows[key]
            direction_src_to_dst = True
        else:
            flow = flows[rkey]
            direction_src_to_dst = False
        flow.update(
            ts=ts,
            length=packet.length,
            direction_src_to_dst=direction_src_to_dst,
            tcp_flags=packet.tcp_flags,
            wrong_fragment=packet.wrong_fragment,
            icmp_error=packet.icmp_error,
        )
    # Convert flows to connection records
    conns: List[ConnRecord] = []
    for k, f in flows.items():
        service = service_name(f.proto, f.dport)
        flag = infer_flag(f)
        rec = ConnRecord(
            start=f.start, end=f.end, src=f.src, dst=f.dst, sport=f.sport, dport=f.dport,
            proto=f.proto, service=service, flag=flag,
            src_bytes=f.src_bytes, dst_bytes=f.dst_bytes,
            land=f.land, wrong_fragment=f.wrong_fragment, urgent=f.urgent_src,
            icmp_error=f.icmp_error
        )
        conns.append(rec)
    # Sort by start time for window calculations
    conns.sort(key=lambda c: c.start)
    # --- Time-based window (2 seconds) for each src host ---
    per_src_state: Dict[str, _SrcWindowState] = defaultdict(_SrcWindowState)
    for c in conns:
        state = per_src_state[c.src]
        cutoff = c.start - 2.0
        _src_window_evict(state, cutoff)
        window_size = len(state.records)
        if window_size:
            same_srv = state.service_counts.get(c.service, 0)
            c.count = window_size
            c.srv_count = same_srv
            c.same_srv_rate = same_srv / window_size
            c.diff_srv_rate = (window_size - same_srv) / window_size
            svc_dst_counts = state.service_dst_counts.get(c.service)
            same_srv_same_dst = svc_dst_counts.get(c.dst, 0) if svc_dst_counts else 0
            c.srv_diff_host_rate = (same_srv - same_srv_same_dst) / window_size
            c.serror_rate = state.serror_total / window_size
            c.rerror_rate = state.rerror_total / window_size
            if same_srv:
                c.srv_serror_rate = state.service_serror.get(c.service, 0) / same_srv
                c.srv_rerror_rate = state.service_rerror.get(c.service, 0) / same_srv
        _src_window_add(state, c)

    # --- Host-based window: last 100 connections to same dst host ---
    per_dst_state: Dict[str, _DstWindowState] = defaultdict(_DstWindowState)
    for c in conns:
        state = per_dst_state[c.dst]
        _dst_window_trim(state)
        # compute metrics
        window_size = len(state.records)
        if window_size:
            c.dst_host_count = window_size
            srv_count = state.service_counts.get(c.service, 0)
            c.dst_host_srv_count = srv_count
            c.dst_host_same_srv_rate = srv_count / window_size if window_size else 0.0
            c.dst_host_diff_srv_rate = 1.0 - c.dst_host_same_srv_rate
            same_src_port = state.src_port_counts.get(c.sport, 0)
            c.dst_host_same_src_port_rate = same_src_port / window_size
            if srv_count:
                src_counts = state.service_src_counts.get(c.service)
                same_service_same_src = src_counts.get(c.src, 0) if src_counts else 0
                c.dst_host_srv_diff_host_rate = (srv_count - same_service_same_src) / srv_count
                c.dst_host_srv_serror_rate = state.service_serror.get(c.service, 0) / srv_count
                c.dst_host_srv_rerror_rate = state.service_rerror.get(c.service, 0) / srv_count
            else:
                c.dst_host_srv_diff_host_rate = 0.0
            c.dst_host_serror_rate = state.serror_total / window_size
            c.dst_host_rerror_rate = state.rerror_total / window_size
        _dst_window_add(state, c)

    return conns

def write_csv(out_csv: str, conns: List[ConnRecord]):
    header = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
        "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for c in conns:
            duration = max(0.0, c.end - c.start)
            row = [
                round(duration,6), c.proto, c.service, c.flag, c.src_bytes, c.dst_bytes, c.land, c.wrong_fragment, c.urgent,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                c.count, c.srv_count, round(c.serror_rate,6), round(c.srv_serror_rate,6), round(c.rerror_rate,6), round(c.srv_rerror_rate,6),
                round(c.same_srv_rate,6), round(c.diff_srv_rate,6), round(c.srv_diff_host_rate,6),
                c.dst_host_count, c.dst_host_srv_count, round(c.dst_host_same_srv_rate,6), round(c.dst_host_diff_srv_rate,6),
                round(c.dst_host_same_src_port_rate,6), round(c.dst_host_srv_diff_host_rate,6),
                round(c.dst_host_serror_rate,6), round(c.dst_host_srv_serror_rate,6), round(c.dst_host_rerror_rate,6), round(c.dst_host_srv_rerror_rate,6),
            ]
            w.writerow(row)

def main():
    if len(sys.argv) < 3:
        print("Usage: python pcap_kdd_extractor.py <input.pcap> <output.csv>", file=sys.stderr)
        sys.exit(1)
    in_pcap = sys.argv[1]
    out_csv = sys.argv[2]
    conns = process_pcap(in_pcap)
    write_csv(out_csv, conns)
    print(f"Wrote {len(conns)} connection records to {out_csv}")

if __name__ == "__main__":
    main()
