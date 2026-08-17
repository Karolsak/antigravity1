#!/usr/bin/env python3
"""
3-Phase Induction Motor – Comprehensive Analysis Suite
======================================================
Problem: 400 V, 3-phase motor, R2=0.02 Ω, X2=0.1 Ω, N1/N2 = 5
Find  : Starter resistance for maximum starting torque; rotor current.

Tabs  : Solution | Torque-Speed | Fault Current | Protection |
        Speed Controller | Thermal | Economic | Harmonics
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import solve_ivp

# ───────────────────────────────────────────────────────────────
# Colour palette
# ───────────────────────────────────────────────────────────────
BG     = '#0f172a'
PANEL  = '#1e293b'
ACCENT = '#38bdf8'
ACC2   = '#a78bfa'
ACC3   = '#34d399'
ACC4   = '#fb923c'
ACC5   = '#f472b6'
TXT    = '#f1f5f9'
TXTD   = '#94a3b8'
GRID   = '#334155'
WARN   = '#fbbf24'
ERR    = '#f87171'

MPL_STYLE = {
    'figure.facecolor':  PANEL,
    'axes.facecolor':    BG,
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   TXT,
    'axes.titlecolor':   ACCENT,
    'xtick.color':       TXTD,
    'ytick.color':       TXTD,
    'grid.color':        GRID,
    'text.color':        TXT,
    'legend.facecolor':  PANEL,
    'legend.edgecolor':  GRID,
    'legend.labelcolor': TXT,
    'lines.linewidth':   2.0,
}
matplotlib.rcParams.update(MPL_STYLE)


# ───────────────────────────────────────────────────────────────
# Main Application Class
# ───────────────────────────────────────────────────────────────
class MotorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("3-Phase Induction Motor — Comprehensive Analysis Suite")
        self.root.configure(bg=BG)
        try:
            self.root.state('zoomed')
        except tk.TclError:
            self.root.attributes('-zoomed', True)

        self.sim_running = False
        self._after_id = None
        self.fuzzy_mode = tk.BooleanVar(value=False)

        self._init_params()
        self._build_ui()
        self.root.after(300, self.refresh_all)

    # ── Parameters ────────────────────────────────────────────
    def _init_params(self):
        self.p = {
            'V_line':      tk.DoubleVar(value=400.0),
            'f':           tk.DoubleVar(value=50.0),
            'poles':       tk.IntVar(value=4),
            'R2':          tk.DoubleVar(value=0.02),
            'X2':          tk.DoubleVar(value=0.10),
            'R1':          tk.DoubleVar(value=0.01),
            'X1':          tk.DoubleVar(value=0.05),
            'Xm':          tk.DoubleVar(value=2.00),
            'turns_ratio': tk.DoubleVar(value=5.0),
            'J':           tk.DoubleVar(value=0.50),
            'B':           tk.DoubleVar(value=0.01),
            'TL':          tk.DoubleVar(value=50.0),
            'Kp':          tk.DoubleVar(value=2.00),
            'Ki':          tk.DoubleVar(value=0.50),
            'Kd':          tk.DoubleVar(value=0.10),
            'R_fault':     tk.DoubleVar(value=0.001),
            'load_step':   tk.DoubleVar(value=100.0),
            'step_time':   tk.DoubleVar(value=2.0),
        }

    def v(self, key):
        return self.p[key].get()

    # ── Base motor calculations ──────────────────────────────────
    def calc_base(self):
        V_line = self.v('V_line')
        f      = self.v('f')
        poles  = self.v('poles')
        R2     = self.v('R2')
        X2     = self.v('X2')
        R1     = self.v('R1')
        X1     = self.v('X1')
        n      = self.v('turns_ratio')

        V_ph   = V_line / np.sqrt(3)
        E2     = V_ph / n                         # Rotor EMF per phase
        ws     = 2 * np.pi * f / (poles / 2)      # Synchronous speed (mech rad/s)
        Ns     = 120 * f / poles                   # Synchronous speed (RPM)

        # For maximum starting torque at s=1: R2_total = X2
        R_ext    = max(X2 - R2, 0)
        R_total  = R2 + R_ext                      # = X2

        Z_start  = np.sqrt(R_total**2 + X2**2)    # sqrt(2) * X2
        I2_start = E2 / Z_start

        # Electromagnetic torque at start
        T_start  = (3 * E2**2 * R_total) / (ws * (R_total**2 + X2**2))

        # Efficiency (approx)
        s_rated  = 0.05
        Nr       = Ns * (1 - s_rated)
        Pmech    = T_start * 2 * np.pi * Nr / 60
        Pin      = 3 * V_ph * I2_start

        # Max torque (no external resistance)
        s_Tmax   = R2 / np.sqrt(R1**2 + (X1 + X2)**2)
        T_max    = (3 / ws) * V_ph**2 / (
                    2 * (R1 + np.sqrt(R1**2 + (X1 + X2)**2)))

        return dict(V_ph=V_ph, E2=E2, ws=ws, Ns=Ns,
                    R_ext=R_ext, R_total=R_total,
                    I2_start=I2_start, T_start=T_start,
                    s_Tmax=s_Tmax, T_max=T_max,
                    Pin=abs(Pin), Pmech=abs(Pmech))

    def torque_speed_curve(self, s_arr=None, R_add=0.0):
        """Return (slip, RPM, Torque) arrays."""
        if s_arr is None:
            s_arr = np.linspace(0.001, 1.0, 600)
        V_ph  = self.v('V_line') / np.sqrt(3)
        R1    = self.v('R1')
        R2    = self.v('R2') + R_add
        X1    = self.v('X1')
        X2    = self.v('X2')
        f     = self.v('f')
        poles = self.v('poles')
        ws    = 2 * np.pi * f / (poles / 2)
        Ns    = 120 * f / poles

        with np.errstate(divide='ignore', invalid='ignore'):
            T = (3 / ws) * V_ph**2 * (R2 / s_arr) / (
                (R1 + R2 / s_arr)**2 + (X1 + X2)**2)
        N = Ns * (1 - s_arr)
        return s_arr, N, np.nan_to_num(T, nan=0.0, posinf=0.0)

    def fault_current(self, t_arr):
        """Symmetrical 3-phase fault current (sub-transient/transient/steady)."""
        V_ph   = self.v('V_line') / np.sqrt(3)
        X2     = self.v('X2')
        X1     = self.v('X1')
        R1     = self.v('R1')
        R_f    = self.v('R_fault')
        f      = self.v('f')
        ws_e   = 2 * np.pi * f

        Xd_pp  = (X1 + X2) / 2          # Sub-transient
        Xd_p   = X1 + X2                 # Transient
        Xd     = X1 + self.v('Xm')       # Steady-state

        I_pp   = V_ph / Xd_pp
        I_p    = V_ph / Xd_p
        I_ss   = V_ph / np.sqrt(Xd**2 + (R1 + R_f)**2)

        tau_pp = 0.05   # Sub-transient time constant (s)
        tau_p  = 0.30   # Transient time constant (s)

        i_ac = (I_pp * np.exp(-t_arr / tau_pp) +
                (I_p - I_ss) * np.exp(-t_arr / tau_p) + I_ss)
        i_dc  = I_pp * np.sqrt(2) * np.exp(-t_arr / tau_p)
        i_total = np.sqrt(2) * i_ac + i_dc
        return i_ac, i_dc, i_total

    def sim_speed_controller(self, t_end=6.0):
        """Simulate speed with PID or fuzzy controller + step load."""
        f     = self.v('f')
        poles = self.v('poles')
        ws    = 2 * np.pi * f / (poles / 2)
        Ns    = 120 * f / poles
        ws_ref = ws * 0.95  # 95% of sync speed

        TL0   = self.v('TL')
        TL1   = self.v('load_step')
        t_step= self.v('step_time')
        J     = self.v('J')
        B     = self.v('B')
        Kp    = self.v('Kp')
        Ki    = self.v('Ki')
        Kd    = self.v('Kd')

        V_ph  = self.v('V_line') / np.sqrt(3)
        R2    = self.v('R2')
        R1    = self.v('R1')
        X1    = self.v('X1')
        X2    = self.v('X2')

        def Te(omega):
            s = max((ws - omega) / ws, 1e-4)
            t = (3 / ws) * V_ph**2 * (R2 / s) / ((R1 + R2 / s)**2 + (X1 + X2)**2)
            return min(t, 800.0)

        def fuzzy_torque(e, de):
            """Simple fuzzy P+D output gain."""
            ke  = np.clip(e / ws, -1, 1)
            kde = np.clip(de / (ws * 5), -1, 1)
            return float(max(0.5, 1.0 + 2.0 * ke - 0.5 * kde))

        def ode(t, y):
            omega, integral_e, prev_e = y
            TL = TL0 if t < t_step else TL1
            e  = ws_ref - omega
            if self.fuzzy_mode.get():
                de  = e - prev_e
                gain = fuzzy_torque(e, de)
                u   = gain * (Kp * e + Ki * integral_e)
            else:
                u = Kp * e + Ki * integral_e + Kd * (e - prev_e)
            u = np.clip(u, -500, 500)
            domega  = (Te(omega) - TL - B * omega + u) / J
            d_int   = e
            d_prev  = e
            return [domega, d_int, d_prev]

        t_span = (0, t_end)
        t_eval = np.linspace(0, t_end, 800)
        sol = solve_ivp(ode, t_span, [0.0, 0.0, 0.0],
                        t_eval=t_eval, method='RK45',
                        rtol=1e-4, atol=1e-6, max_step=0.02)
        return sol.t, sol.y[0], Ns, ws, TL0, TL1, t_step

    def thermal_model(self, t_arr):
        """Winding temperature rise model (simplified thermal RC)."""
        R2    = self.v('R2') + self.v('R2')   # total rotor resistance
        I2    = self.calc_base()['I2_start']
        R1    = self.v('R1')
        I1    = I2 * (1 / self.v('turns_ratio'))
        P_cu  = 3 * (I2**2 * self.v('R2') + I1**2 * R1)  # copper loss (W)
        P_fe  = 150.0   # assumed iron loss (W)
        P_loss= P_cu + P_fe

        tau_th  = 600.0    # thermal time constant (s) ~10 min
        T_amb   = 40.0     # ambient temperature (°C)
        R_th    = 0.08     # thermal resistance (°C/W)
        T_ss    = T_amb + P_loss * R_th
        T_t     = T_ss + (T_amb - T_ss) * np.exp(-t_arr / tau_th)
        return T_t, P_cu, P_fe, T_ss

    def harmonic_spectrum(self):
        """Return harmonic orders, magnitudes, THD."""
        f     = self.v('f')
        V_ph  = self.v('V_line') / np.sqrt(3)
        R2    = self.v('R2')
        X2    = self.v('X2')

        orders = np.array([1, 5, 7, 11, 13, 17, 19, 23, 25])
        # Typical induction motor current harmonic magnitudes (pu)
        mag_pu = np.array([1.000, 0.175, 0.110, 0.045,
                           0.029, 0.015, 0.010, 0.009, 0.008])
        I_fund = self.calc_base()['I2_start']
        mags   = mag_pu * I_fund

        I_harm = np.sqrt(np.sum(mags[1:]**2))
        THD    = I_harm / mags[0] * 100
        return orders, mags, THD

    def economic_analysis(self, hours=8760):
        """Annual energy cost and losses."""
        base   = self.calc_base()
        eta    = 0.92                      # assumed efficiency
        Pin    = base['Pin'] / eta
        energy = Pin * hours / 1000        # kWh
        rate   = 0.12                      # $/kWh
        cost   = energy * rate
        co2    = energy * 0.45             # kg CO2 per kWh
        return Pin, energy, cost, co2

    # ── UI Layout ───────────────────────────────────────────
    def _build_ui(self):
        # Left: parameter panel
        self.left = tk.Frame(self.root, bg=PANEL, width=290)
        self.left.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 2), pady=4)
        self.left.pack_propagate(False)

        # Right: notebook
        self.right = tk.Frame(self.root, bg=BG)
        self.right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                        padx=(2, 4), pady=4)

        self._build_param_panel()
        self._build_notebook()

    def _build_param_panel(self):
        lf = self.left

        # Title
        tk.Label(lf, text="⚡ Motor Parameters",
                 bg=PANEL, fg=ACCENT,
                 font=('Helvetica', 11, 'bold')).pack(pady=(10, 4))

        tk.Frame(lf, bg=GRID, height=1).pack(fill=tk.X, padx=8)

        # Scrollable canvas for sliders
        canvas = tk.Canvas(lf, bg=PANEL, bd=0, highlightthickness=0)
        sb     = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=PANEL)
        win_id = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(win_id, width=canvas.winfo_width())

        inner.bind('<Configure>', _on_inner_configure)
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        sliders = [
            ("Supply voltage (V)",       'V_line',    300,  500,  1.0),
            ("Frequency (Hz)",           'f',          45,   65,  0.5),
            ("Poles",                    'poles',       2,   12,  2),
            ("Rotor resistance R2 (Ω)",  'R2',       0.005, 0.20, 0.005),
            ("Standstill reactance X2",  'X2',       0.02,  0.5,  0.005),
            ("Stator resistance R1 (Ω)", 'R1',       0.002, 0.10, 0.002),
            ("Stator reactance X1 (Ω)", 'X1',        0.01,  0.30, 0.005),
            ("Magnetising Xm (Ω)",       'Xm',        0.5,   5.0,  0.1),
            ("Turns ratio N1/N2",        'turns_ratio', 1,   10,  0.5),
            ("Inertia J (kg·m²)",        'J',          0.1,  5.0,  0.1),
            ("Friction B",               'B',         0.001, 0.1,  0.001),
            ("Load torque TL (N·m)",     'TL',          0,  300,  5),
            ("PID Kp",                   'Kp',         0.1,  10.0, 0.1),
            ("PID Ki",                   'Ki',         0.0,   5.0,  0.05),
            ("PID Kd",                   'Kd',         0.0,   2.0,  0.02),
            ("Fault resistance Rf (Ω)",  'R_fault',  0.0001, 0.05, 0.0005),
            ("Step load TL2 (N·m)",      'load_step',   0,  400,  10),
            ("Step time (s)",            'step_time',   0.5,  5.0,  0.1),
        ]

        for label, key, lo, hi, res in sliders:
            frame = tk.Frame(inner, bg=PANEL)
            frame.pack(fill=tk.X, padx=8, pady=2)

            tk.Label(frame, text=label, bg=PANEL, fg=TXT,
                     font=('Helvetica', 8), anchor='w').pack(fill=tk.X)

            row = tk.Frame(frame, bg=PANEL)
            row.pack(fill=tk.X)

            val_var = tk.StringVar()

            def _update_label(v, vv=val_var, k=key):
                vv.set(f"{self.p[k].get():.4g}")
                self.refresh_all()

            sl = tk.Scale(row, variable=self.p[key],
                          from_=lo, to=hi, resolution=res,
                          orient=tk.HORIZONTAL, bg=PANEL, fg=TXT,
                          troughcolor=BG, activebackground=ACCENT,
                          highlightthickness=0, sliderlength=14,
                          command=_update_label, showvalue=False)
            sl.pack(side=tk.LEFT, fill=tk.X, expand=True)

            val_var.set(f"{self.p[key].get():.4g}")
            tk.Label(row, textvariable=val_var, bg=PANEL, fg=ACCENT,
                     font=('Courier', 8, 'bold'), width=7,
                     anchor='e').pack(side=tk.LEFT)

        tk.Frame(inner, bg=GRID, height=1).pack(fill=tk.X, padx=8, pady=6)

        # Controller mode
        tk.Label(inner, text="Controller Mode", bg=PANEL, fg=TXTD,
                 font=('Helvetica', 8, 'bold')).pack(anchor='w', padx=10)
        tk.Radiobutton(inner, text="PID", variable=self.fuzzy_mode,
                       value=False, bg=PANEL, fg=TXT,
                       selectcolor=BG, activebackground=PANEL,
                       font=('Helvetica', 8)).pack(anchor='w', padx=20)
        tk.Radiobutton(inner, text="Fuzzy Logic", variable=self.fuzzy_mode,
                       value=True, bg=PANEL, fg=TXT,
                       selectcolor=BG, activebackground=PANEL,
                       font=('Helvetica', 8)).pack(anchor='w', padx=20)

        # Control buttons
        tk.Frame(inner, bg=GRID, height=1).pack(fill=tk.X, padx=8, pady=6)

        btn_cfg = dict(bg=BG, fg=TXT, relief=tk.FLAT,
                       font=('Helvetica', 9, 'bold'), pady=6, padx=10,
                       cursor='hand2', bd=0)

        btn_row = tk.Frame(inner, bg=PANEL)
        btn_row.pack(fill=tk.X, padx=8, pady=4)

        self.btn_start = tk.Button(btn_row, text="▶ Start",
                                   bg=ACC3, fg=BG, **{k: v for k, v in btn_cfg.items()
                                                       if k not in ('bg', 'fg')},
                                   command=self.start_sim)
        self.btn_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_stop = tk.Button(btn_row, text="■ Stop",
                                  bg=ERR, fg=BG, **{k: v for k, v in btn_cfg.items()
                                                      if k not in ('bg', 'fg')},
                                  command=self.stop_sim)
        self.btn_stop.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_reset = tk.Button(inner, text="↺ Reset Defaults",
                                   bg=PANEL, fg=ACCENT,
                                   font=('Helvetica', 9), pady=4,
                                   relief=tk.FLAT, bd=0, cursor='hand2',
                                   command=self.reset_defaults)
        self.btn_reset.pack(fill=tk.X, padx=8, pady=2)

    def _build_notebook(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=TXTD,
                        padding=[10, 5], font=('Helvetica', 9, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', BG)],
                  foreground=[('selected', ACCENT)])

        self.nb = ttk.Notebook(self.right)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tabs = {}
        tab_names = [
            ('solution',    '📐 Solution'),
            ('torque',      '⚙ Torque-Speed'),
            ('fault',       '⚡ Fault Current'),
            ('protection',  '🛡 Protection'),
            ('controller',  '🎛 Speed Control'),
            ('thermal',     '🌡 Thermal'),
            ('economic',    '💰 Economic'),
            ('harmonics',   '〰 Harmonics'),
        ]
        for key, title in tab_names:
            frame = tk.Frame(self.nb, bg=BG)
            self.nb.add(frame, text=title)
            self.tabs[key] = frame

        self._build_tab_solution()
        self._build_tab_torque()
        self._build_tab_fault()
        self._build_tab_protection()
        self._build_tab_controller()
        self._build_tab_thermal()
        self._build_tab_economic()
        self._build_tab_harmonics()

    # ─── Helper: embedded matplotlib figure ───────────────────────
    def _embed_fig(self, parent, figsize=(10, 6), nrows=1, ncols=1,
                   toolbar=True):
        fig = Figure(figsize=figsize, tight_layout=True)
        fig.patch.set_facecolor(PANEL)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        if toolbar:
            tb_frame = tk.Frame(parent, bg=PANEL)
            tb_frame.pack(fill=tk.X)
            tb = NavigationToolbar2Tk(canvas, tb_frame)
            tb.config(background=PANEL)
            for child in tb.winfo_children():
                try:
                    child.config(background=PANEL)
                except tk.TclError:
                    pass
        return fig, canvas

    def _styled_ax(self, ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor(BG)
        ax.set_title(title, color=ACCENT, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel(xlabel, color=TXTD, fontsize=8)
        ax.set_ylabel(ylabel, color=TXTD, fontsize=8)
        ax.tick_params(colors=TXTD, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(True, color=GRID, alpha=0.5, linestyle='--', linewidth=0.6)
        return ax

    # ────────────────────────────────────────────────────────────
    # TAB 1: Problem Solution
    # ────────────────────────────────────────────────────────────
    def _build_tab_solution(self):
        tab = self.tabs['solution']

        top = tk.Frame(tab, bg=BG)
        top.pack(fill=tk.X, padx=10, pady=(8, 4))

        tk.Label(top, text="3-Phase Induction Motor — Starter Resistance for Maximum Starting Torque",
                 bg=BG, fg=ACCENT, font=('Helvetica', 12, 'bold')).pack(anchor='w')
        tk.Label(top, text="Detailed mathematical derivation and numerical results",
                 bg=BG, fg=TXTD, font=('Helvetica', 9)).pack(anchor='w')

        mid = tk.Frame(tab, bg=BG)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Left: text results
        left = tk.Frame(mid, bg=PANEL, bd=0)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self.sol_text = tk.Text(left, bg=PANEL, fg=TXT,
                                font=('Courier', 9), wrap=tk.WORD,
                                relief=tk.FLAT, padx=10, pady=10,
                                state=tk.DISABLED)
        sb_sol = ttk.Scrollbar(left, command=self.sol_text.yview)
        self.sol_text.config(yscrollcommand=sb_sol.set)
        sb_sol.pack(side=tk.RIGHT, fill=tk.Y)
        self.sol_text.pack(fill=tk.BOTH, expand=True)

        # Right: equivalent circuit diagram + phasor
        right = tk.Frame(mid, bg=PANEL, width=420)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.pack_propagate(False)

        self.fig_sol, self.canvas_sol = self._embed_fig(right, figsize=(6, 5))

    def _update_tab_solution(self):
        b = self.calc_base()

        # ── Text panel ─────────────────────────────────────────
        lines = []
        lines.append("=" * 62)
        lines.append("  PROBLEM STATEMENT")
        lines.append("=" * 62)
        lines.append(f"  Supply voltage  V  = {self.v('V_line')} V  (line-to-line, 3-phase)")
        lines.append(f"  Rotor resistance   = {self.v('R2')} Ω")
        lines.append(f"  Standstill X2      = {self.v('X2')} Ω")
        lines.append(f"  Turns ratio N1/N2  = {self.v('turns_ratio')}")
        lines.append("")
        lines.append("=" * 62)
        lines.append("  STEP 1 — Phase Voltage")
        lines.append("=" * 62)
        lines.append(f"  V_ph = V_line / √3")
        lines.append(f"       = {self.v('V_line')} / {np.sqrt(3):.5f}")
        lines.append(f"       = {b['V_ph']:.4f} V")
        lines.append("")
        lines.append("=" * 62)
        lines.append("  STEP 2 — Rotor EMF per Phase (referred to rotor)")
        lines.append("=" * 62)
        lines.append(f"  E2 = V_ph / (N1/N2)")
        lines.append(f"     = {b['V_ph']:.4f} / {self.v('turns_ratio')}")
        lines.append(f"     = {b['E2']:.4f} V")
        lines.append("")
        lines.append("=" * 62)
        lines.append("  STEP 3 — Condition for Maximum Torque at Starting")
        lines.append("=" * 62)
        lines.append("  At slip s = 1 (standstill), max torque requires:")
        lines.append("    R2_total = X2  (rotor resistance = rotor reactance)")
        lines.append("")
        lines.append(f"  R2_total needed = X2 = {self.v('X2')} Ω")
        lines.append(f"  R2 (rotor)      = {self.v('R2')} Ω")
        lines.append("")
        lines.append("  External starter resistance:")
        lines.append(f"  R_ext = X2 - R2")
        lines.append(f"        = {self.v('X2')} - {self.v('R2')}")
        lines.append(f"        = {b['R_ext']:.4f} Ω   ◄ ANSWER 1")
        lines.append("")
        lines.append("=" * 62)
        lines.append("  STEP 4 — Rotor Current at Starting")
        lines.append("=" * 62)
        lines.append(f"  Impedance: Z = √(R_total² + X2²)")
        lines.append(f"              = √({b['R_total']:.4f}² + {self.v('X2')}²)")
        lines.append(f"              = √({b['R_total']**2:.6f} + {self.v('X2')**2:.6f})")
        lines.append(f"              = {np.sqrt(b['R_total']**2 + self.v('X2')**2):.4f} Ω")
        lines.append("")
        lines.append(f"  I2 = E2 / Z")
        lines.append(f"     = {b['E2']:.4f} / {np.sqrt(b['R_total']**2 + self.v('X2')**2):.4f}")
        lines.append(f"     = {b['I2_start']:.4f} A   ◄ ANSWER 2")
        lines.append("")
        lines.append("=" * 62)
        lines.append("  STEP 5 — Starting Torque & Other Results")
        lines.append("=" * 62)
        lines.append(f"  Synchronous speed Ns    = {b['Ns']:.1f} RPM")
        lines.append(f"  Sync. ang. speed  ωs    = {b['ws']:.4f} rad/s")
        lines.append(f"  Starting torque   T_st  = {b['T_start']:.4f} N·m")
        lines.append(f"  Max torque (no R_ext)   = {b['T_max']:.4f} N·m")
        lines.append(f"  Slip at max torque      = {b['s_Tmax']:.4f}")
        lines.append("")
        lines.append("=" * 62)
        lines.append("  PHYSICAL INTERPRETATION")
        lines.append("=" * 62)
        lines.append("  Adding external resistance in the rotor circuit shifts")
        lines.append("  the peak of the torque-speed curve towards s=1 (standstill).")
        lines.append("  When R_ext = X2 - R2, the peak occurs exactly at s=1,")
        lines.append("  giving maximum torque at the moment of starting.")
        lines.append("  This is the principle of the wound-rotor induction motor")
        lines.append("  starter (rheostat starter).")
        lines.append("")
        lines.append("  Note: The external resistance is gradually reduced as the")
        lines.append("  motor accelerates, maintaining high torque throughout.")

        self.sol_text.config(state=tk.NORMAL)
        self.sol_text.delete('1.0', tk.END)

        # Tag-based colouring
        self.sol_text.tag_configure('header', foreground=ACCENT,
                                    font=('Courier', 9, 'bold'))
        self.sol_text.tag_configure('answer', foreground=ACC3,
                                    font=('Courier', 9, 'bold'))
        self.sol_text.tag_configure('normal', foreground=TXT,
                                    font=('Courier', 9))

        for line in lines:
            if line.startswith('='):
                self.sol_text.insert(tk.END, line + '\n', 'header')
            elif '◄ ANSWER' in line:
                self.sol_text.insert(tk.END, line + '\n', 'answer')
            elif line.startswith('  STEP') or line.startswith('  PROBLEM') or \
                 line.startswith('  PHYSICAL'):
                self.sol_text.insert(tk.END, line + '\n', 'header')
            else:
                self.sol_text.insert(tk.END, line + '\n', 'normal')

        self.sol_text.config(state=tk.DISABLED)

        # ── Circuit diagram ────────────────────────────────────
        self.fig_sol.clear()
        gs = gridspec.GridSpec(2, 1, figure=self.fig_sol,
                               hspace=0.45)

        # Phasor diagram
        ax1 = self.fig_sol.add_subplot(gs[0])
        self._styled_ax(ax1, 'Rotor Phasor Diagram at Starting (s=1)',
                        'Real axis (Ω)', 'Imaginary axis (Ω)')

        R_tot = b['R_total']
        X2    = self.v('X2')
        phi   = np.arctan2(X2, R_tot)
        Z     = np.sqrt(R_tot**2 + X2**2)

        ax1.annotate('', xy=(R_tot, 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=ACCENT, lw=2))
        ax1.annotate('', xy=(R_tot, X2), xytext=(R_tot, 0),
                     arrowprops=dict(arrowstyle='->', color=WARN, lw=2))
        ax1.annotate('', xy=(R_tot, X2), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=ACC3, lw=2.5))

        ax1.text(R_tot / 2, -0.005, f'R={R_tot:.3f}Ω', color=ACCENT,
                 ha='center', va='top', fontsize=8)
        ax1.text(R_tot + 0.003, X2 / 2, f'X={X2:.3f}Ω', color=WARN,
                 ha='left', va='center', fontsize=8)
        ax1.text(R_tot / 2 - 0.01, X2 / 2 + 0.005,
                 f'Z={Z:.3f}Ω', color=ACC3, fontsize=8, rotation=np.degrees(phi))

        ax1.set_xlim(-0.01, R_tot + 0.04)
        ax1.set_ylim(-0.02, X2 + 0.04)
        ax1.set_aspect('equal')
        ax1.legend(['R_total', 'jX2', 'Z (impedance)'],
                   fontsize=7, loc='lower right')

        # Bar chart – summary results
        ax2 = self.fig_sol.add_subplot(gs[1])
        self._styled_ax(ax2, 'Key Results Summary', '', '')

        labels  = ['R_ext\n(Ω)', 'I2_start\n(A)', 'T_start\n(N·m)',
                   'T_max\n(N·m)', 'Ns\n(RPM÷10)']
        values  = [b['R_ext'], b['I2_start'], b['T_start'],
                   b['T_max'], b['Ns'] / 10]
        colours = [ACCENT, ACC2, ACC3, WARN, ACC4]

        bars = ax2.bar(labels, values, color=colours, width=0.5, alpha=0.85)
        for bar, val, lbl in zip(bars, [b['R_ext'], b['I2_start'],
                                         b['T_start'], b['T_max'], b['Ns']],
                                  labels):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(values) * 0.02,
                     f"{val:.3g}", ha='center', va='bottom',
                     fontsize=8, color=TXT)

        ax2.set_ylabel('Value', color=TXTD, fontsize=8)
        ax2.tick_params(axis='x', labelsize=8)

        self.canvas_sol.draw()

    # ────────────────────────────────────────────────────────────
    # TAB 2: Torque-Speed Characteristics
    # ────────────────────────────────────────────────────────────
    def _build_tab_torque(self):
        self.fig_torque, self.canvas_torque = \
            self._embed_fig(self.tabs['torque'], figsize=(11, 7))

    def _update_tab_torque(self):
        self.fig_torque.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig_torque,
                               hspace=0.45, wspace=0.35)

        s_arr, N, T_no = self.torque_speed_curve()[:3]
        b    = self.calc_base()
        R_ext = b['R_ext']

        # Curves with different external resistances
        ax1 = self.fig_torque.add_subplot(gs[0, :])
        self._styled_ax(ax1, 'Torque vs. Speed — Effect of External Rotor Resistance',
                        'Speed (RPM)', 'Torque (N·m)')

        r_vals   = [0.0, R_ext / 2, R_ext, R_ext * 1.5, R_ext * 2]
        r_labels = ['R_ext=0 (normal)', f'R_ext/2={R_ext/2:.3f}Ω',
                    f'R_ext={R_ext:.3f}Ω (max torque start)',
                    f'1.5×R_ext', f'2×R_ext']
        clrs     = [ACCENT, ACC2, ACC3, WARN, ACC4]

        for r_add, lbl, clr in zip(r_vals, r_labels, clrs):
            _, Nv, Tv = self.torque_speed_curve(R_add=r_add)
            lw   = 2.5 if r_add == R_ext else 1.5
            ax1.plot(Nv, Tv, color=clr, lw=lw, label=lbl)

        # Operating point
        TL = self.v('TL')
        ax1.axhline(TL, color=ERR, lw=1.2, linestyle='--', label=f'Load torque={TL:.0f} N·m')
        ax1.legend(fontsize=7, loc='upper right')

        # Slip axis (twin)
        ax1b = ax1.twiny()
        ax1b.set_xlim(ax1.get_xlim())
        Ns = b['Ns']
        ticks_N = np.linspace(0, Ns, 6)
        ax1b.set_xticks(ticks_N)
        ax1b.set_xticklabels([f'{1-n/Ns:.2f}' for n in ticks_N],
                              color=TXTD, fontsize=7)
        ax1b.set_xlabel('Slip', color=TXTD, fontsize=7)
        ax1b.tick_params(colors=TXTD)
        for sp in ax1b.spines.values():
            sp.set_color(GRID)

        # ── Torque vs Slip ──────────────────────────────────
        ax2 = self.fig_torque.add_subplot(gs[1, 0])
        self._styled_ax(ax2, 'Torque vs. Slip',
                        'Slip s', 'Torque (N·m)')
        for r_add, lbl, clr in zip(r_vals, r_labels, clrs):
            sv, _, Tv = self.torque_speed_curve(R_add=r_add)
            ax2.plot(sv, Tv, color=clr, lw=1.5, label=lbl.split('(')[0].strip())
        ax2.legend(fontsize=6)
        ax2.invert_xaxis()

        # ── Current vs Speed ────────────────────────────────
        ax3 = self.fig_torque.add_subplot(gs[1, 1])
        self._styled_ax(ax3, 'Rotor Current vs. Speed',
                        'Speed (RPM)', 'Rotor Current I2 (A)')

        V_ph = self.v('V_line') / np.sqrt(3)
        for r_add, lbl, clr in zip(r_vals[:3], r_labels[:3], clrs[:3]):
            R2  = self.v('R2') + r_add
            X2  = self.v('X2')
            E2  = V_ph / self.v('turns_ratio')
            s_v = np.linspace(0.001, 1, 600)
            I2v = E2 / np.sqrt((R2 / s_v)**2 + X2**2)
            Nv2 = b['Ns'] * (1 - s_v)
            ax3.plot(Nv2, I2v, color=clr, lw=1.5,
                     label=lbl.split('(')[0].strip())
        ax3.legend(fontsize=6)

        self.canvas_torque.draw()

    # ────────────────────────────────────────────────────────────
    # TAB 3: Fault Current
    # ────────────────────────────────────────────────────────────
    def _build_tab_fault(self):
        self.fig_fault, self.canvas_fault = \
            self._embed_fig(self.tabs['fault'], figsize=(11, 7))

    def _update_tab_fault(self):
        self.fig_fault.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig_fault,
                               hspace=0.45, wspace=0.35)

        t  = np.linspace(0, 0.5, 2000)
        i_ac, i_dc, i_tot = self.fault_current(t)
        f  = self.v('f')
        t_cyc = np.linspace(0, 0.1, 1000)

        # ── Envelope plot ──────────────────────────────────
        ax1 = self.fig_fault.add_subplot(gs[0, :])
        self._styled_ax(ax1, 'Fault Current — Sub-transient / Transient / DC Components',
                        'Time (s)', 'Current (A)')
        ax1.plot(t, i_tot, color=ACCENT, lw=1.5, label='Total (peak)', alpha=0.9)
        ax1.plot(t, np.sqrt(2) * i_ac, color=ACC3, lw=1.8, label='AC envelope (√2·I_ac)')
        ax1.plot(t, i_dc, color=WARN, lw=1.5, linestyle='--', label='DC offset')
        ax1.fill_between(t, -i_tot, i_tot, color=ACCENT, alpha=0.05)
        ax1.legend(fontsize=8)

        # ── Instantaneous waveform ────────────────────────────
        ax2 = self.fig_fault.add_subplot(gs[1, 0])
        self._styled_ax(ax2, 'Instantaneous Fault Current (first 100 ms)',
                        'Time (ms)', 'Current (A)')
        i_ac2, i_dc2, i_tot2 = self.fault_current(t_cyc)
        i_inst = np.sqrt(2) * i_ac2 * np.sin(2 * np.pi * f * t_cyc) + i_dc2
        ax2.plot(t_cyc * 1000, i_inst, color=ACCENT, lw=1.5)
        ax2.axhline(0, color=GRID, lw=0.8)

        # ── Fault current vs fault resistance ────────────────────
        ax3 = self.fig_fault.add_subplot(gs[1, 1])
        self._styled_ax(ax3, 'Fault Current vs. Fault Resistance',
                        'Fault Resistance Rf (Ω)', 'Fault Current (A)')
        Rf_arr = np.linspace(0.0001, 0.5, 300)
        V_ph   = self.v('V_line') / np.sqrt(3)
        X_total = self.v('X1') + self.v('X2')
        R_total = self.v('R1')
        I_f_arr = V_ph / np.sqrt((R_total + Rf_arr)**2 + X_total**2) * np.sqrt(2)
        ax3.plot(Rf_arr, I_f_arr, color=ERR, lw=2)
        ax3.axvline(self.v('R_fault'), color=WARN, lw=1.2, linestyle='--',
                    label=f"Current Rf = {self.v('R_fault'):.4f} Ω")
        ax3.legend(fontsize=7)

        self.canvas_fault.draw()

    # ────────────────────────────────────────────────────────────
    # TAB 4: Protection Coordination
    # ────────────────────────────────────────────────────────────
    def _build_tab_protection(self):
        tab = self.tabs['protection']

        top = tk.Frame(tab, bg=BG)
        top.pack(fill=tk.X, padx=10, pady=(6, 2))
        tk.Label(top, text="Overcurrent Protection Coordination",
                 bg=BG, fg=ACCENT, font=('Helvetica', 11, 'bold')).pack(anchor='w')

        self.fig_prot, self.canvas_prot = \
            self._embed_fig(tab, figsize=(11, 6))

    def _update_tab_protection(self):
        self.fig_prot.clear()
        gs = gridspec.GridSpec(1, 2, figure=self.fig_prot,
                               wspace=0.35)

        b     = self.calc_base()
        I_fl  = b['I2_start'] * 0.30   # approximate full-load current
        I_fault_sym = b['I2_start']

        # Relay TCC curves (IEC standard inverse)
        def relay_tcc(I_mult, pickup, dial, k=0.14, alpha=0.02):
            with np.errstate(divide='ignore', invalid='ignore'):
                t = dial * k / ((I_mult / pickup)**alpha - 1)
            return np.where(I_mult > pickup, np.clip(t, 0.01, 100), np.inf)

        I_arr = np.logspace(np.log10(I_fl * 0.8), np.log10(I_fault_sym * 3), 400)

        # Primary relay (motor protection)
        pickup_1 = I_fl * 1.25
        dial_1   = 0.5
        t_1      = relay_tcc(I_arr, pickup_1, dial_1)

        # Backup relay (feeder)
        pickup_2 = I_fl * 1.1
        dial_2   = 1.5
        t_2      = relay_tcc(I_arr, pickup_2, dial_2)

        # Fuse curve (approximate)
        t_fuse = 10 * (I_fl / I_arr)**2

        ax1 = self.fig_prot.add_subplot(gs[0])
        self._styled_ax(ax1, 'Time-Current Coordination (TCC) Curves',
                        'Current (A)', 'Operating Time (s)')
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        ax1.plot(I_arr, t_1, color=ACCENT, lw=2, label=f'Motor Relay (PU={pickup_1:.1f}A, TD={dial_1})')
        ax1.plot(I_arr, t_2, color=WARN,   lw=2, label=f'Feeder Relay (PU={pickup_2:.1f}A, TD={dial_2})')
        ax1.plot(I_arr, t_fuse, color=ERR, lw=2, linestyle='--', label='Fuse (approx)')

        ax1.axvline(I_fl,        color=ACC3, lw=1, linestyle=':', alpha=0.8)
        ax1.axvline(I_fault_sym, color=ERR,  lw=1, linestyle=':', alpha=0.8)
        ax1.text(I_fl,        0.015, f'I_FL\n{I_fl:.1f}A',   color=ACC3, fontsize=7, ha='right')
        ax1.text(I_fault_sym, 0.015, f'I_fault\n{I_fault_sym:.1f}A', color=ERR, fontsize=7)

        ax1.set_ylim(0.01, 100)
        ax1.set_ylabel('Time (s)', color=TXTD, fontsize=8)
        ax1.legend(fontsize=7)

        # CTI zones
        I_check = I_arr[(I_arr > pickup_1) & (I_arr > pickup_2)]
        if len(I_check) > 0:
            t1_chk = relay_tcc(I_check, pickup_1, dial_1)
            t2_chk = relay_tcc(I_check, pickup_2, dial_2)
            valid  = (t2_chk > t1_chk) & np.isfinite(t1_chk) & np.isfinite(t2_chk)
            if valid.any():
                ax1.fill_between(I_check[valid], t1_chk[valid], t2_chk[valid],
                                 alpha=0.15, color=ACC3, label='CTI zone')

        # ── Protection settings summary ─────────────────────
        ax2 = self.fig_prot.add_subplot(gs[1])
        ax2.set_facecolor(BG)
        ax2.axis('off')
        self._styled_ax(ax2, 'Protection Settings Summary', '', '')

        CTI = 0.3   # coordination time interval
        rows = [
            ["Parameter",              "Value",           "Unit"],
            ["Full-load current",      f"{I_fl:.2f}",     "A"],
            ["Fault current (sym)",    f"{I_fault_sym:.2f}", "A"],
            ["Motor relay pickup",     f"{pickup_1:.2f}", "A"],
            ["Motor relay time-dial",  f"{dial_1}",       "—"],
            ["Feeder relay pickup",    f"{pickup_2:.2f}", "A"],
            ["Feeder relay time-dial", f"{dial_2}",       "—"],
            ["Target CTI",             f"{CTI}",          "s"],
            ["Motor protection class", "10A",             "IEC"],
        ]

        col_w  = [0.52, 0.28, 0.20]
        header_clr = [ACCENT, ACCENT, ACCENT]
        row_clrs   = [TXT, ACC3, TXTD]

        y = 0.95
        for i, row in enumerate(rows):
            x = 0.02
            clrs = header_clr if i == 0 else row_clrs
            fw   = 'bold' if i == 0 else 'normal'
            for cell, cw, clr in zip(row, col_w, clrs):
                ax2.text(x, y, cell, transform=ax2.transAxes,
                         color=clr, fontsize=8.5,
                         fontweight=fw, va='top')
                x += cw
            if i == 0:
                ax2.axhline(y - 0.04, color=GRID, lw=0.8, xmin=0.02, xmax=0.98,
                            transform=ax2.transAxes)
            y -= 0.10

        self.canvas_prot.draw()

    # ────────────────────────────────────────────────────────────
    # TAB 5: Speed Controller
    # ────────────────────────────────────────────────────────────
    def _build_tab_controller(self):
        self.fig_ctrl, self.canvas_ctrl = \
            self._embed_fig(self.tabs['controller'], figsize=(11, 7))

    def _update_tab_controller(self):
        self.fig_ctrl.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig_ctrl,
                               hspace=0.45, wspace=0.35)

        t, omega, Ns, ws, TL0, TL1, t_step = self.sim_speed_controller()
        RPM = omega * 60 / (2 * np.pi)
        ref_RPM = ws * 0.95 * 60 / (2 * np.pi)

        # ── Speed response ────────────────────────────────
        ax1 = self.fig_ctrl.add_subplot(gs[0, :])
        mode = "Fuzzy Logic" if self.fuzzy_mode.get() else "PID"
        self._styled_ax(ax1, f'Speed Response — {mode} Controller (Step Load Change)',
                        'Time (s)', 'Speed (RPM)')

        ax1.plot(t, RPM, color=ACCENT, lw=2, label='Motor speed')
        ax1.axhline(ref_RPM, color=ACC3, lw=1.5, linestyle='--',
                    label=f'Reference {ref_RPM:.1f} RPM')
        ax1.axhline(Ns, color=TXTD, lw=1, linestyle=':', label=f'Sync speed {Ns:.0f} RPM')
        ax1.axvline(t_step, color=WARN, lw=1.5, linestyle='--', alpha=0.7,
                    label=f'Load step at t={t_step}s')
        ax1.fill_between(t, RPM, ref_RPM, alpha=0.1, color=ACCENT)
        ax1.legend(fontsize=8, loc='lower right')

        # ── Load torque profile ─────────────────────────────
        ax2 = self.fig_ctrl.add_subplot(gs[1, 0])
        self._styled_ax(ax2, 'Load Torque Profile', 'Time (s)', 'Torque (N·m)')
        TL_profile = np.where(t < t_step, TL0, TL1)
        ax2.step(t, TL_profile, color=WARN, lw=2, where='post', label='Load torque')
        ax2.legend(fontsize=8)

        # ── Speed error ───────────────────────────────────
        ax3 = self.fig_ctrl.add_subplot(gs[1, 1])
        self._styled_ax(ax3, 'Speed Error (Reference − Actual)',
                        'Time (s)', 'Error (RPM)')
        e_rpm = ref_RPM - RPM
        ax3.plot(t, e_rpm, color=ERR, lw=1.5, label='Error')
        ax3.fill_between(t, e_rpm, 0, alpha=0.15, color=ERR)
        ax3.axhline(0, color=GRID, lw=0.8)
        ax3.legend(fontsize=8)

        self.canvas_ctrl.draw()

    # ────────────────────────────────────────────────────────────
    # TAB 6: Thermal Analysis
    # ────────────────────────────────────────────────────────────
    def _build_tab_thermal(self):
        self.fig_therm, self.canvas_therm = \
            self._embed_fig(self.tabs['thermal'], figsize=(11, 7))

    def _update_tab_thermal(self):
        self.fig_therm.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig_therm,
                               hspace=0.45, wspace=0.35)

        t_arr = np.linspace(0, 3600, 1000)   # 1 hour
        T_t, P_cu, P_fe, T_ss = self.thermal_model(t_arr)

        # ── Temperature rise ───────────────────────────────
        ax1 = self.fig_therm.add_subplot(gs[0, :])
        self._styled_ax(ax1, 'Winding Temperature Rise Model',
                        'Time (min)', 'Temperature (°C)')
        ax1.plot(t_arr / 60, T_t, color=ERR, lw=2, label='Winding temperature')
        ax1.axhline(T_ss, color=WARN, lw=1.5, linestyle='--',
                    label=f'Steady-state: {T_ss:.1f} °C')
        ax1.axhline(130, color=ACC4, lw=1.2, linestyle=':',
                    label='Class B insulation limit (130°C)')
        ax1.axhline(155, color=ERR,  lw=1.2, linestyle=':',
                    label='Class F insulation limit (155°C)')
        ax1.fill_between(t_arr / 60, T_t, 40, alpha=0.15, color=ERR)
        ax1.legend(fontsize=8)

        # ── Loss breakdown ────────────────────────────────
        ax2 = self.fig_therm.add_subplot(gs[1, 0])
        self._styled_ax(ax2, 'Loss Breakdown', '', 'Power (W)')
        labels  = ['Copper\nLoss', 'Iron\nLoss', 'Mech.\n(stray)']
        P_stray = 50.0
        values  = [P_cu, P_fe, P_stray]
        colours = [ERR, WARN, ACC2]
        bars = ax2.bar(labels, values, color=colours, alpha=0.85, width=0.5)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5, f'{val:.1f}W',
                     ha='center', fontsize=8, color=TXT)

        # ── Overload vs thermal limit ───────────────────────
        ax3 = self.fig_therm.add_subplot(gs[1, 1])
        self._styled_ax(ax3, 'Thermal Overload Curve',
                        'Overload Current (pu)', 'Trip Time (s)')
        I_pu  = np.linspace(1.05, 5.0, 300)
        tau_m = 600.0   # motor thermal time const
        T_trip = tau_m * np.log((I_pu**2) / (I_pu**2 - 1))
        ax3.semilogy(I_pu, T_trip, color=ERR, lw=2)
        ax3.fill_between(I_pu, T_trip, 1, alpha=0.15, color=ERR)
        ax3.set_ylim(1, tau_m)
        ax3.axvline(1.5, color=WARN, lw=1.2, linestyle='--',
                    label='1.5 pu (typical NEC overload)')
        ax3.legend(fontsize=7)

        self.canvas_therm.draw()

    # ────────────────────────────────────────────────────────────
    # TAB 7: Economic Analysis
    # ────────────────────────────────────────────────────────────
    def _build_tab_economic(self):
        self.fig_econ, self.canvas_econ = \
            self._embed_fig(self.tabs['economic'], figsize=(11, 7))

    def _update_tab_economic(self):
        self.fig_econ.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig_econ,
                               hspace=0.45, wspace=0.35)

        Pin, energy_yr, cost_yr, co2_yr = self.economic_analysis()

        # ── Annual cost vs operating hours ────────────────────
        ax1 = self.fig_econ.add_subplot(gs[0, :])
        self._styled_ax(ax1, 'Annual Energy Cost vs. Operating Hours',
                        'Operating Hours / Year', 'Annual Cost (USD)')
        hrs    = np.linspace(0, 8760, 300)
        cost_h = Pin / 0.92 / 1000 * 0.12 * hrs
        co2_h  = Pin / 0.92 / 1000 * 0.45 * hrs

        ax1.plot(hrs, cost_h, color=ACC3, lw=2, label='Energy cost ($)')
        ax1_r = ax1.twinx()
        ax1_r.plot(hrs, co2_h, color=WARN, lw=2, linestyle='--', label='CO₂ (kg)')
        ax1_r.set_ylabel('CO₂ Emissions (kg)', color=WARN, fontsize=8)
        ax1_r.tick_params(colors=WARN)
        for sp in ax1_r.spines.values():
            sp.set_color(GRID)

        ax1.axvline(8760, color=TXTD, lw=0.8, linestyle=':')
        ax1.legend(fontsize=8, loc='upper left')
        ax1_r.legend(fontsize=8, loc='lower right')

        # ── Efficiency vs load ──────────────────────────────
        ax2 = self.fig_econ.add_subplot(gs[1, 0])
        self._styled_ax(ax2, 'Efficiency vs. Load',
                        'Load (%)', 'Efficiency (%)')
        load_pct = np.linspace(10, 125, 300)
        eta = 95 - 0.001 * (load_pct - 75)**2 - 1.5 / (load_pct / 100 + 0.1)
        eta = np.clip(eta, 0, 100)
        ax2.plot(load_pct, eta, color=ACCENT, lw=2)
        ax2.axvline(100, color=TXTD, lw=0.8, linestyle=':')
        ax2.set_ylim(50, 100)

        # ── Cost summary pie ───────────────────────────────
        ax3 = self.fig_econ.add_subplot(gs[1, 1])
        ax3.set_facecolor(BG)
        ax3.set_title('Annual Cost Breakdown', color=ACCENT,
                      fontsize=10, fontweight='bold')

        capex_annual = 800.0    # amortised capital ($/yr)
        maint_annual = 300.0    # maintenance ($/yr)
        energy_cost  = cost_yr

        sizes  = [energy_cost, capex_annual, maint_annual]
        labels = [f'Energy\n${energy_cost:,.0f}',
                  f'Capital\n${capex_annual:,.0f}',
                  f'Maint.\n${maint_annual:,.0f}']
        clrs   = [ACCENT, ACC3, WARN]

        wedges, _, autotexts = ax3.pie(
            sizes, labels=labels, colors=clrs, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75,
            textprops=dict(color=TXT, fontsize=7))
        for at in autotexts:
            at.set_color(BG)
            at.set_fontsize(7)

        self.canvas_econ.draw()

    # ────────────────────────────────────────────────────────────
    # TAB 8: Harmonics & Power Quality
    # ────────────────────────────────────────────────────────────
    def _build_tab_harmonics(self):
        self.fig_harm, self.canvas_harm = \
            self._embed_fig(self.tabs['harmonics'], figsize=(11, 7))

    def _update_tab_harmonics(self):
        self.fig_harm.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig_harm,
                               hspace=0.45, wspace=0.35)

        orders, mags, THD = self.harmonic_spectrum()
        f     = self.v('f')
        t_arr = np.linspace(0, 3 / f, 2000)

        # Reconstructed current waveform
        i_wave = np.zeros_like(t_arr)
        for h, m in zip(orders, mags):
            i_wave += m * np.sqrt(2) * np.sin(2 * np.pi * h * f * t_arr)

        # ── Waveform ────────────────────────────────────────
        ax1 = self.fig_harm.add_subplot(gs[0, :])
        self._styled_ax(ax1, 'Rotor Current Waveform (with harmonics)',
                        'Time (ms)', 'Current (A)')
        i_fund = mags[0] * np.sqrt(2) * np.sin(2 * np.pi * f * t_arr)
        ax1.plot(t_arr * 1000, i_wave, color=ACCENT, lw=1.5,
                 label=f'Total (THD={THD:.2f}%)')
        ax1.plot(t_arr * 1000, i_fund, color=ACC3,  lw=1.2,
                 linestyle='--', label='Fundamental')
        ax1.legend(fontsize=8)

        # ── Harmonic spectrum bar ──────────────────────────
        ax2 = self.fig_harm.add_subplot(gs[1, 0])
        self._styled_ax(ax2, 'Harmonic Spectrum',
                        'Harmonic Order', 'Current (A)')
        clr_bars = [ACCENT if o == 1 else ACC4 for o in orders]
        ax2.bar(orders, mags, color=clr_bars, alpha=0.85)
        ax2.set_xticks(orders)
        for o, m in zip(orders, mags):
            ax2.text(o, m + max(mags) * 0.02, f'{m:.1f}', ha='center',
                     fontsize=7, color=TXT)

        # ── THD vs load ────────────────────────────────────
        ax3 = self.fig_harm.add_subplot(gs[1, 1])
        self._styled_ax(ax3, 'THD vs. Load Current',
                        'Load (%)', 'THD (%)')
        load_pct = np.linspace(10, 125, 200)
        thd_arr  = THD * (1 + 0.3 * np.exp(-load_pct / 40))
        ax3.plot(load_pct, thd_arr, color=WARN, lw=2, label='THD_I')
        ax3.axhline(5, color=ERR, lw=1.2, linestyle='--',
                    label='IEEE 519 limit (5%)')
        ax3.axhline(THD, color=ACCENT, lw=1.2, linestyle=':',
                    label=f'Current operating point {THD:.2f}%')
        ax3.legend(fontsize=7)
        ax3.set_ylim(0, max(thd_arr) * 1.25)

        # Power factor
        PF = 1 / np.sqrt(1 + (THD / 100)**2) * 0.88
        ax3.text(0.65, 0.92, f'PF ≈ {PF:.3f}',
                 transform=ax3.transAxes, color=ACC3, fontsize=9,
                 fontweight='bold', bbox=dict(facecolor=PANEL, edgecolor=GRID, pad=3))

        self.canvas_harm.draw()

    # ── Refresh / control ─────────────────────────────────
    def refresh_all(self, *_):
        try:
            self._update_tab_solution()
            self._update_tab_torque()
            self._update_tab_fault()
            self._update_tab_protection()
            self._update_tab_controller()
            self._update_tab_thermal()
            self._update_tab_economic()
            self._update_tab_harmonics()
        except Exception as exc:
            print(f"[refresh] {exc}")

    def start_sim(self):
        if not self.sim_running:
            self.sim_running = True
            self._sim_loop()

    def _sim_loop(self):
        if self.sim_running:
            self.refresh_all()
            self._after_id = self.root.after(2000, self._sim_loop)

    def stop_sim(self):
        self.sim_running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def reset_defaults(self):
        defaults = dict(V_line=400, f=50, poles=4, R2=0.02, X2=0.1,
                        R1=0.01, X1=0.05, Xm=2.0, turns_ratio=5.0,
                        J=0.5, B=0.01, TL=50, Kp=2.0, Ki=0.5, Kd=0.1,
                        R_fault=0.001, load_step=100.0, step_time=2.0)
        for k, v in defaults.items():
            self.p[k].set(v)
        self.refresh_all()

    def _on_resize(self, event):
        pass   # tight_layout handles auto-scaling via matplotlib


# ───────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    app  = MotorApp(root)
    root.mainloop()
