"""
CPU Scheduler Simulator - Final Version
Run: python cpu_scheduler_final.py
Requires: pip install matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox
import copy, matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import deque

# ── Colors ─────────────────────────────────────────────────────────────
BG       = "#0D1117"
SURFACE  = "#161B22"
SURFACE2 = "#21262D"
SURFACE3 = "#30363D"
ACCENT   = "#58A6FF"
GREEN    = "#3FB950"
PURPLE   = "#BC8CFF"
ORANGE   = "#FF8C42"
PINK     = "#F778BA"
TEAL     = "#39D353"
GOLD     = "#E3B341"
RED      = "#F85149"
TEXT     = "#E6EDF3"
TEXT2    = "#8B949E"
BORDER   = "#30363D"

COLORS = ["#58A6FF","#BC8CFF","#3FB950","#FF8C42",
          "#F778BA","#39D353","#E3B341","#F85149",
          "#79C0FF","#D2A8FF"]

# ── Algorithms ─────────────────────────────────────────────────────────
def run_fcfs(procs):
    result, t = [], 0
    for p in sorted(procs, key=lambda x: (x["at"], x["pid"])):
        if t < p["at"]: t = p["at"]
        result.append((p["pid"], t, t + p["bt"]))
        t += p["bt"]
    return result

def run_sjf(procs):
    result, t, pool = [], 0, copy.deepcopy(procs)
    while pool:
        ready = [p for p in pool if p["at"] <= t]
        if not ready:
            t = min(p["at"] for p in pool)
            continue
        p = min(ready, key=lambda x: (x["bt"], x["at"]))
        pool.remove(p)
        result.append((p["pid"], t, t + p["bt"]))
        t += p["bt"]
    return result

def run_srtf(procs):
    result = []
    rem  = {p["pid"]: p["bt"] for p in procs}
    at   = {p["pid"]: p["at"] for p in procs}
    total = sum(p["bt"] for p in procs) + max(p["at"] for p in procs) + 1
    cur, seg = None, 0
    for t in range(total):
        ready = [pid for pid, r in rem.items() if r > 0 and at[pid] <= t]
        if not ready:
            if cur: result.append((cur, seg, t)); cur = None
            continue
        chosen = min(ready, key=lambda pid: (rem[pid], pid))
        if chosen != cur:
            if cur: result.append((cur, seg, t))
            cur, seg = chosen, t
        rem[chosen] -= 1
        if rem[chosen] == 0:
            result.append((chosen, seg, t + 1))
            del rem[chosen]; cur = None
        if not rem: break
    return result

def run_rr(procs, q):
    result = []
    pool   = sorted(copy.deepcopy(procs), key=lambda p: (p["at"], p["pid"]))
    rem    = {p["pid"]: p["bt"] for p in pool}
    queue  = deque()
    idx, t = 0, 0
    while idx < len(pool) and pool[idx]["at"] <= t:
        queue.append(pool[idx]["pid"]); idx += 1
    while queue:
        pid = queue.popleft()
        run = min(q, rem[pid])
        result.append((pid, t, t + run))
        t += run; rem[pid] -= run
        while idx < len(pool) and pool[idx]["at"] <= t:
            queue.append(pool[idx]["pid"]); idx += 1
        if rem[pid] > 0: queue.append(pid)
        if not queue and idx < len(pool):
            t = pool[idx]["at"]
            while idx < len(pool) and pool[idx]["at"] <= t:
                queue.append(pool[idx]["pid"]); idx += 1
    return result

def run_priority(procs):
    result = []
    rem  = {p["pid"]: p["bt"] for p in procs}
    at   = {p["pid"]: p["at"] for p in procs}
    pr   = {p["pid"]: p["pr"] for p in procs}
    total = sum(p["bt"] for p in procs) + max(p["at"] for p in procs) + 1
    cur, seg = None, 0
    for t in range(total):
        ready = [pid for pid, r in rem.items() if r > 0 and at[pid] <= t]
        if not ready:
            if cur: result.append((cur, seg, t)); cur = None
            continue
        chosen = min(ready, key=lambda pid: (pr[pid], at[pid], pid))
        if chosen != cur:
            if cur: result.append((cur, seg, t))
            cur, seg = chosen, t
        rem[chosen] -= 1
        if rem[chosen] == 0:
            result.append((chosen, seg, t + 1))
            del rem[chosen]; cur = None
        if not rem: break
    return result

def get_metrics(procs, timeline):
    first, last = {}, {}
    for pid, s, e in timeline:
        if pid not in first: first[pid] = s
        last[pid] = e
    out = []
    for p in procs:
        pid = p["pid"]
        if pid not in first: continue
        tat = last[pid]  - p["at"]
        wt  = tat        - p["bt"]
        rt  = first[pid] - p["at"]
        out.append({**p, "start": first[pid], "finish": last[pid],
                    "tat": tat, "wt": wt, "rt": rt})
    return out

def merge(tl):
    if not tl:
        return []
    m = []
    prev_end = 0
    for pid, s, e in tl:
        if s > prev_end:
            m.append(("IDLE", prev_end, s))
        if m and m[-1][0] == pid and m[-1][2] == s:
            m[-1] = (pid, m[-1][1], e)
        else:
            m.append((pid, s, e))
        prev_end = e
    return m

def clear_frame(frame):
    """Destroy all children and close any matplotlib figures inside."""
    for w in frame.winfo_children():
        w.destroy()
    plt.close("all")

# ── App ────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root  = root
        self.root.title("CPU Scheduler Simulator")
        self.root.geometry("1300x840")
        self.root.configure(bg=BG)
        self.procs = []
        self.cmap  = {}
        self.pid_n = 1
        self._canvas_refs = {}   # keep canvas references to prevent GC
        self._styles()
        self._build()
        self._load_demo()

    # ── Styles ──────────────────────────────────────────────────────────
    def _styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, font=("Consolas", 10))
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=SURFACE2, foreground=TEXT2,
                    padding=(16, 8), font=("Consolas", 10))
        s.map("TNotebook.Tab",
              background=[("selected", SURFACE)],
              foreground=[("selected", TEXT)])
        s.configure("Treeview", background=SURFACE, foreground=TEXT,
                    fieldbackground=SURFACE, rowheight=28, borderwidth=0,
                    font=("Consolas", 10))
        s.configure("Treeview.Heading", background=SURFACE2, foreground=TEXT2,
                    font=("Consolas", 9, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", PURPLE)])

    # ── Build UI ────────────────────────────────────────────────────────
    def _build(self):
        # Header
        hdr = tk.Frame(self.root, bg=SURFACE, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚙", bg=SURFACE, fg=ACCENT,
                 font=("Consolas", 18)).pack(side="left", padx=(16,8))
        tk.Label(hdr, text="CPU SCHEDULER SIMULATOR", bg=SURFACE, fg=TEXT,
                 font=("Consolas", 13, "bold")).pack(side="left")
        tk.Label(hdr, text="  —  FCFS · SJF · SRTF · Round Robin · Priority",
                 bg=SURFACE, fg=TEXT2, font=("Consolas", 9)).pack(side="left")

        # Body
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=10)

        left = tk.Frame(body, bg=SURFACE, width=310)
        left.pack(side="left", fill="y", padx=(0,10))
        left.pack_propagate(False)
        self._build_left(left)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_right(right)

        # Status bar
        bar = tk.Frame(self.root, bg=SURFACE2, height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status = tk.StringVar(value="Ready · Load demo or add processes then click Run")
        tk.Label(bar, textvariable=self.status, bg=SURFACE2, fg=TEXT2,
                 font=("Consolas", 9)).pack(side="left", padx=12, pady=4)

    # ── Sidebar ──────────────────────────────────────────────────────────
    def _build_left(self, parent):

        # ── RUN BUTTON at TOP ──
        run_top = tk.Frame(parent, bg=SURFACE, pady=10)
        run_top.pack(fill="x")
        self.run_btn = tk.Button(
            run_top, text="▶  RUN SIMULATION",
            bg=ACCENT, fg="#FFFFFF",
            font=("Consolas", 12, "bold"),
            relief="flat", cursor="hand2",
            pady=12, command=self._run)
        self.run_btn.pack(fill="x", padx=14)
        self.run_btn.bind("<Enter>", lambda e: self.run_btn.config(bg="#79C0FF"))
        self.run_btn.bind("<Leave>", lambda e: self.run_btn.config(bg=ACCENT))

        # ── Algorithm ──
        self._sep(parent, "ALGORITHM")
        self.algo = tk.StringVar(value="FCFS")
        algos = [("FCFS", ACCENT), ("SJF", PURPLE), ("SRTF", TEAL),
                 ("Round Robin", ORANGE), ("Priority", GOLD),
                 ("Compare All", GREEN)]
        self._dot_widgets = []
        for name, col in algos:
            f = tk.Frame(parent, bg=SURFACE, cursor="hand2")
            f.pack(fill="x", padx=14, pady=2)
            dot = tk.Label(f, text="◉", bg=SURFACE,
                           fg=ACCENT if name=="FCFS" else SURFACE3,
                           font=("Consolas", 11))
            dot.pack(side="left", padx=(0,8))
            lbl = tk.Label(f, text=name, bg=SURFACE,
                           fg=TEXT if name=="FCFS" else TEXT2,
                           font=("Consolas", 10))
            lbl.pack(side="left")
            dot._col  = col
            dot._name = name
            lbl._name = name
            self._dot_widgets.append((dot, lbl, col, name))
            def pick(n=name, c=col):
                self.algo.set(n)
                self._refresh_dots()
                self.q_frame.pack(fill="x", padx=14, pady=4) if n == "Round Robin" else self.q_frame.pack_forget()
            for w in (f, dot, lbl):
                w.bind("<Button-1>", lambda e, fn=pick: fn())

        # Quantum
        self.q_frame = tk.Frame(parent, bg=SURFACE)
        tk.Label(self.q_frame, text="Quantum:", bg=SURFACE, fg=TEXT2,
                 font=("Consolas", 9)).pack(side="left", padx=(0,6))
        self.q_var = tk.StringVar(value="2")
        tk.Entry(self.q_frame, textvariable=self.q_var, width=4,
                 bg=SURFACE2, fg=ORANGE, insertbackground=TEXT,
                 relief="flat", font=("Consolas", 12, "bold"),
                 justify="center").pack(side="left")
        self.q_frame.pack_forget()

        # ── Add Process ──
        self._sep(parent, "ADD PROCESS")
        ef = tk.Frame(parent, bg=SURFACE2)
        ef.pack(fill="x", padx=14, pady=4)

        self.fvars = {}
        for label, key, default, col in [
            ("Arrival Time","at","0",TEAL),
            ("Burst Time",  "bt","4",ACCENT),
            ("Priority",    "pr","1",GOLD),
        ]:
            row = tk.Frame(ef, bg=SURFACE2)
            row.pack(fill="x", padx=10, pady=6)
            tk.Label(row, text=label, bg=SURFACE2, fg=TEXT2,
                     font=("Consolas", 9), width=11, anchor="w").pack(side="left")
            v = tk.StringVar(value=default)
            self.fvars[key] = v
            e = tk.Entry(row, textvariable=v, width=6,
                         bg=SURFACE3, fg=col, insertbackground=TEXT,
                         relief="flat", font=("Consolas", 12, "bold"),
                         justify="center")
            e.pack(side="left", padx=6)
            e.bind("<Return>", lambda ev: self._add())
        tk.Label(
            ef,
            text="Lower number = Higher priority",
            bg=SURFACE2,
            fg=TEXT2,
            font=("Consolas", 8)
            ).pack(anchor="w", padx=12, pady=(0,6))

        brow = tk.Frame(parent, bg=SURFACE)
        brow.pack(fill="x", padx=14, pady=6)
        self._mk_btn(brow, "+ ADD", self._add, ACCENT).pack(side="left", padx=(0,6))
        self._mk_btn(brow, "CLEAR ALL", self._clear, SURFACE3).pack(side="left")

        # ── Process List ──
        self._sep(parent, "PROCESS LIST")
        lf = tk.Frame(parent, bg=SURFACE)
        lf.pack(fill="x", padx=14, pady=4)

        cols = ("PID","AT","BT","PR")
        self.tree = ttk.Treeview(lf, columns=cols, show="headings", height=7)
        for col, w in zip(cols, [60,55,55,55]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill="x")
        self._mk_btn(lf, "✕ REMOVE SELECTED", self._remove,
                     SURFACE3, w=260).pack(pady=(6,0), fill="x")

        # Demo
        self._sep(parent, "")
        self._mk_btn(parent, "⚡ LOAD DEMO", self._load_demo,
                     GOLD, w=260).pack(padx=14, pady=6, fill="x")

    def _mk_btn(self, parent, text, cmd, col, w=None):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=col, fg=TEXT, font=("Consolas", 9, "bold"),
                      relief="flat", cursor="hand2", pady=6,
                      width=w//8 if w else None)
        return b

    def _sep(self, parent, text):
        f = tk.Frame(parent, bg=SURFACE)
        f.pack(fill="x", padx=14, pady=(12,4))
        if text:
            tk.Label(f, text=text, bg=SURFACE, fg=SURFACE3,
                     font=("Consolas", 8, "bold")).pack(side="left")
        tk.Frame(f, bg=SURFACE3, height=1).pack(
            side="left", fill="x", expand=True, padx=(8,0), pady=7)

    def _refresh_dots(self):
        algo = self.algo.get()
        for dot, lbl, col, name in self._dot_widgets:
            selected = (name == algo)
            dot.config(fg=col if selected else SURFACE3)
            lbl.config(fg=TEXT if selected else TEXT2)

    # ── Right panel ──────────────────────────────────────────────────────
    def _build_right(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill="both", expand=True)

        self.t_gantt   = tk.Frame(self.nb, bg=BG)
        self.t_metrics = tk.Frame(self.nb, bg=BG)
        self.t_table   = tk.Frame(self.nb, bg=BG)
        self.t_compare = tk.Frame(self.nb, bg=BG)

        self.nb.add(self.t_gantt,   text="  Gantt Chart  ")
        self.nb.add(self.t_metrics, text="  Metrics  ")
        self.nb.add(self.t_table,   text="  Details  ")
        self.nb.add(self.t_compare, text="  Compare All  ")

        for t in (self.t_gantt, self.t_metrics, self.t_table, self.t_compare):
            tk.Label(t, text="Press  ▶ RUN SIMULATION  to see results",
                     bg=BG, fg=SURFACE3, font=("Consolas", 12)).pack(expand=True)

    # ── Process management ────────────────────────────────────────────────
    def _add(self):
        try:
            at = int(self.fvars["at"].get())
            bt = int(self.fvars["bt"].get())
            pr = int(self.fvars["pr"].get())
            assert at >= 0
            assert bt >= 1
            assert pr >= 0
        except:
            messagebox.showerror(
                "Error",
                "Arrival and Priority must be ≥ 0.\nBurst Time must be ≥ 1."
                )
            return
        pid = f"P{self.pid_n}"; self.pid_n += 1
        p = {"pid": pid, "at": at, "bt": bt, "pr": pr}
        self.procs.append(p)
        self.cmap[pid] = COLORS[len(self.procs) % len(COLORS)]
        self.tree.insert("", "end", values=(pid, at, bt, pr), tags=(pid,))
        self.tree.tag_configure(pid, foreground=self.cmap[pid])
        self.fvars["at"].set(str(at + bt))
        self.status.set(f"{len(self.procs)} processes · Press RUN SIMULATION")

    def _remove(self):
        for item in self.tree.selection():
            pid = self.tree.item(item)["values"][0]
            self.procs = [p for p in self.procs if p["pid"] != pid]
            self.tree.delete(item)

    def _clear(self):
        self.procs.clear(); self.cmap.clear(); self.pid_n = 1
        for item in self.tree.get_children(): self.tree.delete(item)
        self.status.set("Cleared · Add processes")

    def _load_demo(self):
        self._clear()
        for at, bt, pr in [(0,6,3),(2,4,1),(4,2,4),(6,5,2),(8,3,5)]:
            self.fvars["at"].set(str(at))
            self.fvars["bt"].set(str(bt))
            self.fvars["pr"].set(str(pr))
            self._add()
        self.status.set("Demo loaded · Press RUN SIMULATION")

    # ── Run ───────────────────────────────────────────────────────────────
    def _run(self):
        if not self.procs:
            messagebox.showwarning("No Processes", "Add at least one process.")
            return
        algo = self.algo.get()
        try:
            q = int(self.q_var.get()) if algo == "Round Robin" else 2
            assert q >= 1
        except:
            messagebox.showerror("Error", "Invalid quantum. Must be ≥ 1.")
            return

        self.status.set(f"Running {algo}…")
        self.root.update()

        try:
            if algo == "Compare All":
                self._do_compare(q)
                self.nb.select(3)
            else:
                tl = self._timeline(algo, q)
                m  = get_metrics(self.procs, tl)
                self._draw_gantt(tl, algo)
                self._draw_metrics(m)
                self._draw_table(m)
                self._do_compare(q)   # always refresh compare tab too
                self.nb.select(0)
            self.status.set(f"✓  {algo} done · {len(self.procs)} processes scheduled")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    def _timeline(self, algo, q=2):
        if algo == "FCFS":        return run_fcfs(self.procs)
        if algo == "SJF":         return run_sjf(self.procs)
        if algo == "SRTF":        return run_srtf(self.procs)
        if algo == "Round Robin": return run_rr(self.procs, q)
        if algo == "Priority":    return run_priority(self.procs)

    # ── Draw helpers ──────────────────────────────────────────────────────
    def _embed_fig(self, fig, tab, key):
        """Safely embed a matplotlib figure into a tab, replacing old one."""
        clear_frame(tab)
        cv = FigureCanvasTkAgg(fig, master=tab)
        cv.draw()
        cv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._canvas_refs[key] = cv   # keep reference alive

    # ── Gantt ─────────────────────────────────────────────────────────────
    def _draw_gantt(self, timeline, title):
        clear_frame(self.t_gantt)
        m     = merge(timeline)
        max_t = max(e for _, _, e in m)

        fig, ax = plt.subplots(figsize=(10, 2.8))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(SURFACE)

        for pid, s, e in m:
            col = "#444C56" if pid == "IDLE" else self.cmap.get(pid, COLORS[0])
            ax.barh(0.3, e-s, left=s, height=0.5,
                    color=col, edgecolor=BG, linewidth=1.5)
            if e - s >= 0.8:
                ax.text(s+(e-s)/2, 0.3, pid, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white" if pid != "IDLE" else TEXT2,
                        fontfamily="Consolas")

        ax.set_xlim(-0.2, max_t+0.2)
        ax.set_xticks(range(0, max_t+1))
        ax.set_yticks([0.3]); ax.set_yticklabels(["CPU"], color=TEXT2)
        ax.set_ylim(0, 0.9)
        ax.set_xlabel("Time →", color=TEXT2, fontsize=9)
        ax.set_title(f"{title}  ·  Gantt Chart", color=TEXT,
                     fontsize=11, fontweight="bold", fontfamily="Consolas", pad=10)
        ax.tick_params(colors=TEXT2, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(axis="x", color=BORDER, linestyle="--", alpha=0.5)
        patches = [mpatches.Patch(color=self.cmap.get(p["pid"], COLORS[0]),
                                  label=p["pid"]) for p in self.procs]
        ax.legend(handles=patches, fontsize=8, labelcolor=TEXT,
                  facecolor=SURFACE2, edgecolor=BORDER, framealpha=0.8,
                  loc="upper right", ncol=len(patches))
        plt.tight_layout()
        self._embed_fig(fig, self.t_gantt, "gantt")

    # ── Metrics ───────────────────────────────────────────────────────────
    def _animate_count(self, label, target, suffix, steps=25, delay=30):
        """Animate a label counting up from 0 to target."""
        def step(current, inc):
            current = round(min(current + inc, target), 2)
            label.config(text=f"{current}{suffix}")
            if current < target:
                label.after(delay, step, current, inc)
            else:
                label.config(text=f"{target}{suffix}")
        inc = target / steps if target > 0 else 0
        step(0.0, inc)

    def _draw_metrics(self, metrics):
        clear_frame(self.t_metrics)
        n       = len(metrics)
        avg_wt  = round(sum(m["wt"]  for m in metrics)/n, 2)
        avg_tt  = round(sum(m["tat"] for m in metrics)/n, 2)
        avg_rt  = round(sum(m["rt"]  for m in metrics)/n, 2)
        start_t = min(m["at"] for m in metrics)
        max_t   = max(m["finish"] for m in metrics)
        busy_t  = sum(m["bt"] for m in metrics)
        util = round((busy_t / (max_t - start_t)) * 100, 1)
        throughput = round(len(metrics) / max_t, 2)
        algo = self.algo.get()

        # Animated stat cards matching screenshot style
        cf = tk.Frame(self.t_metrics, bg=BG)
        cf.pack(fill="x", padx=10, pady=10)

        card_data = [
            (avg_wt,  "",  ACCENT,  "Avg Waiting"),
            (avg_tt,  "",  PURPLE,  "Avg Turnaround"),
            (avg_rt,  "",  TEAL,    "Avg Response"),
            (util,    "%", GOLD,    "CPU Utilization"),
            (throughput, "", ORANGE,  "Throughput"),
            (algo,    "",  GREEN,   "Algorithm"),
        ]

        for val, suffix, col, label_text in card_data:
            card = tk.Frame(cf, bg=SURFACE2, padx=16, pady=14)
            card.pack(side="left", fill="x", expand=True, padx=5)
            tk.Frame(card, bg=col, height=3).pack(fill="x", pady=(0,10))
            num_lbl = tk.Label(card,
                               text=f"{val}{suffix}" if isinstance(val, str) else f"0{suffix}",
                               bg=SURFACE2, fg=col,
                               font=("Consolas", 28, "bold"),
                               anchor="w")
            num_lbl.pack(fill="x")
            tk.Label(card, text=label_text,
                     bg=SURFACE2, fg=TEXT2,
                     font=("Consolas", 10),
                     anchor="w").pack(fill="x", pady=(4,0))
            if not isinstance(val, str):
                self._animate_count(num_lbl, val, suffix)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
        fig.patch.set_facecolor(BG)
        pids   = [m["pid"] for m in metrics]
        x, bw  = np.arange(len(pids)), 0.25
        for ax in (ax1, ax2): ax.set_facecolor(SURFACE)

        ax1.bar(x-bw, [m["wt"]  for m in metrics], bw, color=ACCENT,  alpha=0.9, label="Wait")
        ax1.bar(x,    [m["tat"] for m in metrics], bw, color=PURPLE, alpha=0.9, label="Turnaround")
        ax1.bar(x+bw, [m["rt"]  for m in metrics], bw, color=TEAL,   alpha=0.9, label="Response")
        ax1.set_xticks(x); ax1.set_xticklabels(pids, color=TEXT2, fontsize=9)
        ax1.tick_params(colors=TEXT2)
        ax1.set_title("Time Metrics per Process", color=TEXT, fontweight="bold", fontfamily="Consolas")
        ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=SURFACE2, framealpha=0.3, edgecolor=BORDER)
        for sp in ax1.spines.values(): sp.set_edgecolor(BORDER)
        ax1.grid(axis="y", color=BORDER, linestyle="--", alpha=0.4)

        colors = [self.cmap.get(m["pid"], COLORS[0]) for m in metrics]
        ax2.bar(pids, [m["bt"] for m in metrics], color=colors, width=0.5, alpha=0.9)
        ax2.set_title("Burst Time per Process", color=TEXT, fontweight="bold", fontfamily="Consolas")
        ax2.tick_params(colors=TEXT2)
        ax2.set_xticklabels(pids, color=TEXT2, fontsize=9)
        for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)
        ax2.grid(axis="y", color=BORDER, linestyle="--", alpha=0.4)

        plt.tight_layout()
        self._embed_fig(fig, self.t_metrics, "metrics")

    # ── Table ─────────────────────────────────────────────────────────────
    def _draw_table(self, metrics):
        clear_frame(self.t_table)
        n      = len(metrics)
        avg_wt = round(sum(m["wt"]  for m in metrics)/n, 2)
        avg_tt = round(sum(m["tat"] for m in metrics)/n, 2)
        avg_rt = round(sum(m["rt"]  for m in metrics)/n, 2)
        start_t = min(m["at"] for m in metrics)
        max_t   = max(m["finish"] for m in metrics)
        busy_t  = sum(m["bt"] for m in metrics)
        util = round((busy_t / (max_t - start_t)) * 100, 1)
        algo   = self.algo.get()

        # ── Stat cards at top (exact screenshot style) ─────────────────
        cf = tk.Frame(self.t_table, bg=BG)
        cf.pack(fill="x", padx=10, pady=(10,6))

        card_data = [
            (avg_wt,  "",  "#58A6FF", "Avg Waiting"),
            (avg_tt,  "",  "#BC8CFF", "Avg Turnaround"),
            (avg_rt,  "",  "#3FB950", "Avg Response"),
            (util,    "%", "#E3B341", "CPU Utilization"),
            (algo,    "",  "#3FB950", "Algorithm"),
        ]

        for val, suffix, col, label_text in card_data:
            card = tk.Frame(cf, bg="#1C2128", padx=18, pady=14)
            card.pack(side="left", fill="x", expand=True, padx=5)

            # Long coloured top line like screenshot
            tk.Frame(card, bg=col, height=3).pack(fill="x", pady=(0,10))

            # Big value
            num_lbl = tk.Label(card,
                               text=f"{val}{suffix}" if isinstance(val, str) else "0",
                               bg="#1C2128", fg=col,
                               font=("Consolas", 26, "bold"),
                               anchor="w")
            num_lbl.pack(fill="x")

            # Small label below
            tk.Label(card, text=label_text,
                     bg="#1C2128", fg="#6B7799",
                     font=("Consolas", 10),
                     anchor="w").pack(fill="x", pady=(4,0))

            # Animate numeric values
            if not isinstance(val, str):
                self._animate_count(num_lbl, val, suffix)

        # ── PROCESS DETAILS label ──────────────────────────────────────
        tk.Label(self.t_table, text="PROCESS DETAILS",
                 bg=BG, fg="#3D444D",
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=14, pady=(6,4))

        # ── Process table ──────────────────────────────────────────────
        cols = ("PID","Arrival","Burst","Priority","Start","Finish",
                "Waiting","Turnaround","Response")
        tree = ttk.Treeview(self.t_table, columns=cols, show="headings")
        for col, w in zip(cols, [60,70,60,70,60,70,70,95,80]):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center")
        for m in metrics:
            pid = m["pid"]
            tree.insert("", "end",
                        values=(pid, m["at"], m["bt"], m["pr"],
                                m["start"], m["finish"],
                                m["wt"], m["tat"], m["rt"]),
                        tags=(pid,))
            tree.tag_configure(pid, foreground=self.cmap.get(pid, TEXT))
        tree.pack(fill="both", expand=True, padx=8, pady=(0,8))

    # ── Compare All ───────────────────────────────────────────────────────
    def _do_compare(self, q=2):
        clear_frame(self.t_compare)

        specs = [
            ("FCFS",       run_fcfs(self.procs),     ACCENT),
            ("SJF",        run_sjf(self.procs),      PURPLE),
            ("SRTF",       run_srtf(self.procs),     TEAL),
            (f"RR(q={q})", run_rr(self.procs, q),    ORANGE),
            ("Priority",   run_priority(self.procs), GOLD),
        ]

        summaries = []
        for name, tl, col in specs:
            m   = get_metrics(self.procs, tl)
            n   = len(m)
            fin = max(x["finish"] for x in m)
            summaries.append({
                "name": name, "color": col, "tl": tl,
                "wt":   round(sum(x["wt"]  for x in m)/n, 2),
                "tat":  round(sum(x["tat"] for x in m)/n, 2),
                "rt":   round(sum(x["rt"]  for x in m)/n, 2),
                "util": round(sum(x["bt"]  for x in m)/fin*100, 1),
                "throughput": round(len(m)/fin, 2),
            })

        best_wt  = min(s["wt"]   for s in summaries)
        best_tat = min(s["tat"]  for s in summaries)
        best_rt  = min(s["rt"]   for s in summaries)
        best_u   = max(s["util"] for s in summaries)

        # Summary cards — animated style matching screenshot
        cf = tk.Frame(self.t_compare, bg=BG)
        cf.pack(fill="x", padx=10, pady=10)
        for s in summaries:
            col  = s["color"]
            card = tk.Frame(cf, bg=SURFACE2, padx=12, pady=12)
            card.pack(side="left", fill="x", expand=True, padx=4)
            tk.Frame(card, bg=col, height=3).pack(fill="x", pady=(0,8))
            tk.Label(card, text=s["name"], bg=SURFACE2, fg=col,
                     font=("Consolas", 14, "bold"), anchor="w").pack(fill="x")
            for lbl, val, best, mc in [
                ("Avg Waiting",    s["wt"],   best_wt,  ACCENT),
                ("Avg Turnaround", s["tat"],  best_tat, PURPLE),
                ("Avg Response",   s["rt"],   best_rt,  TEAL),
                ("CPU Util",       s["util"], best_u,   GOLD),
            ]:
                row = tk.Frame(card, bg=SURFACE2)
                row.pack(fill="x", pady=1)
                is_best = (val == best)
                suffix  = "%" if lbl == "CPU Util" else ""
                tk.Label(row, text="★ " if is_best else "  ",
                         bg=SURFACE2, fg=GREEN,
                         font=("Consolas", 9)).pack(side="left")
                tk.Label(row, text=lbl + ":", bg=SURFACE2, fg=TEXT2,
                         font=("Consolas", 8), width=13, anchor="w").pack(side="left")
                tk.Label(row, text=f"{val}{suffix}", bg=SURFACE2,
                         fg=mc if is_best else TEXT,
                         font=("Consolas", 10, "bold")).pack(side="left")

        # Charts
        fig = plt.figure(figsize=(11, 6.2))
        fig.patch.set_facecolor(BG)
        names = [s["name"]  for s in summaries]
        cols  = [s["color"] for s in summaries]
        x, bw = np.arange(len(names)), 0.22

        # Grouped bar — time metrics
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor(SURFACE)
        ax1.bar(x-bw, [s["wt"]  for s in summaries], bw, label="Avg Wait",       color=ACCENT,  alpha=0.9)
        ax1.bar(x,    [s["tat"] for s in summaries], bw, label="Avg Turnaround", color=PURPLE,  alpha=0.9)
        ax1.bar(x+bw, [s["rt"]  for s in summaries], bw, label="Avg Response",   color=TEAL,    alpha=0.9)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, color=TEXT2, fontsize=8, rotation=12, ha="right")
        ax1.tick_params(colors=TEXT2, labelsize=8)
        ax1.set_title("Time Metrics Comparison", color=TEXT, fontweight="bold",
                      fontfamily="Consolas", fontsize=10)
        ax1.legend(fontsize=7, labelcolor=TEXT, facecolor=SURFACE2,
                   framealpha=0.3, edgecolor=BORDER)
        for sp in ax1.spines.values(): sp.set_edgecolor(BORDER)
        ax1.grid(axis="y", color=BORDER, linestyle="--", alpha=0.4)

        # CPU utilization
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor(SURFACE)
        bars = ax2.bar(names, [s["util"] for s in summaries],
                       color=cols, width=0.5, alpha=0.9)
        for bar, s in zip(bars, summaries):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"{s['util']}%", ha="center", va="bottom", color=TEXT2, fontsize=8)
        ax2.set_ylim(0, 115)
        ax2.set_title("CPU Utilization (%)", color=TEXT, fontweight="bold",
                      fontfamily="Consolas", fontsize=10)
        ax2.tick_params(colors=TEXT2, labelsize=8)
        ax2.set_xticklabels(names, color=TEXT2, fontsize=8, rotation=12, ha="right")
        for sp in ax2.spines.values(): sp.set_edgecolor(BORDER)
        ax2.grid(axis="y", color=BORDER, linestyle="--", alpha=0.4)

        # Stacked Gantt all algos
        ax3 = fig.add_subplot(2, 1, 2)
        ax3.set_facecolor(SURFACE)
        max_t = max(e for s in summaries for _, _, e in s["tl"])
        n_a, bh, gap = len(summaries), 0.35, 0.55
        for i, s in enumerate(summaries):
            y = (n_a - 1 - i) * gap
            for pid, start, end in merge(s["tl"]):
                col = self.cmap.get(pid, COLORS[0])
                ax3.barh(y, end-start, left=start, height=bh,
                         color=col, edgecolor=BG, linewidth=0.8, alpha=0.9)
                if end - start >= 1.2:
                    ax3.text(start+(end-start)/2, y, pid,
                             ha="center", va="center", fontsize=7,
                             fontweight="bold", color="white", fontfamily="Consolas")

        ax3.set_yticks([(n_a-1-i)*gap for i in range(n_a)])
        ax3.set_yticklabels(names, color=TEXT2, fontsize=9)
        step = max(1, max_t//16)
        ticks = list(range(0, max_t+1, step))
        if max_t not in ticks: ticks.append(max_t)
        ax3.set_xticks(ticks)
        ax3.set_xlim(-0.3, max_t+0.3)
        ax3.tick_params(colors=TEXT2, labelsize=8)
        ax3.set_title("All Algorithms — Gantt Comparison", color=TEXT,
                      fontweight="bold", fontfamily="Consolas", fontsize=10)
        ax3.set_xlabel("Time →", color=TEXT2, fontsize=9)
        for sp in ax3.spines.values(): sp.set_edgecolor(BORDER)
        ax3.grid(axis="x", color=BORDER, linestyle="--", alpha=0.4)

        patches = [mpatches.Patch(color=self.cmap.get(p["pid"], COLORS[0]),
                                  label=p["pid"]) for p in self.procs]
        ax3.legend(handles=patches, fontsize=7, labelcolor=TEXT,
                   facecolor=SURFACE2, edgecolor=BORDER, framealpha=0.8,
                   loc="upper right", ncol=len(patches))

        plt.tight_layout(pad=1.5)
        self._embed_fig(fig, self.t_compare, "compare")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
