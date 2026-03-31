import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
import math

# Division safety epsilon used in power-quality calculations
_EPSILON = 1e-12

# ─────────────────────────────────────────────
# Helper / Calculation class
# ─────────────────────────────────────────────
class DCMotorCalc:
    @staticmethod
    def calc_all(Vs, freq, N_ref, Kb, HP, Ra, La, J):
        Vm = math.sqrt(2) * Vs
        Vdc_max = 2 * Vm / math.pi
        Prated = HP * 746
        Vrated = 110.0
        Ia_rated = Prated / Vrated
        E = Kb * N_ref
        Vdc_req = E + Ia_rated * Ra
        Vdc_req = min(Vdc_req, Vdc_max)
        ratio = Vdc_req / Vdc_max
        ratio = max(-1.0, min(1.0, ratio))
        alpha_rad = math.acos(ratio)
        alpha_deg = math.degrees(alpha_rad)
        Is_rms = Ia_rated
        IT_avg = Ia_rated / 2.0
        IT_rms = Ia_rated / math.sqrt(2)
        PF = (2 * math.sqrt(2) / math.pi) * math.cos(alpha_rad)
        P_in = Vdc_req * Ia_rated
        S = Vs * Is_rms
        THD = math.sqrt((math.pi**2 / 8) - 1) * 100
        Kb_SI = Kb * 60 / (2 * math.pi)
        return {
            "Vm": Vm, "Vdc_max": Vdc_max, "Prated": Prated,
            "Ia_rated": Ia_rated, "E": E, "Vdc_req": Vdc_req,
            "alpha_deg": alpha_deg, "alpha_rad": alpha_rad,
            "Is_rms": Is_rms, "IT_avg": IT_avg, "IT_rms": IT_rms,
            "PF": PF, "P_in": P_in, "S": S, "THD": THD,
            "Kb_SI": Kb_SI, "Ra": Ra, "La": La, "J": J,
            "Vs": Vs, "freq": freq, "N_ref": N_ref, "Kb": Kb, "HP": HP
        }

    @staticmethod
    def calc_from_alpha(Vs, alpha_deg, Ia):
        Vm = math.sqrt(2) * Vs
        Vdc_max = 2 * Vm / math.pi
        alpha_rad = math.radians(alpha_deg)
        Vdc = Vdc_max * math.cos(alpha_rad)
        Is_rms = Ia
        IT_avg = Ia / 2.0
        IT_rms = Ia / math.sqrt(2)
        PF = (2 * math.sqrt(2) / math.pi) * math.cos(alpha_rad)
        THD = math.sqrt((math.pi**2 / 8) - 1) * 100
        return {
            "Vdc": Vdc, "alpha_rad": alpha_rad, "alpha_deg": alpha_deg,
            "Is_rms": Is_rms, "IT_avg": IT_avg, "IT_rms": IT_rms,
            "PF": PF, "THD": THD, "Vm": Vm
        }


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
class DCMotorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DC Motor Speed Control \u2013 Full Bridge Converter Analysis")
        self.geometry("1280x820")
        self.minsize(900, 600)

        # Default parameters
        self.Vs_var = tk.DoubleVar(value=120.0)
        self.freq_var = tk.DoubleVar(value=60.0)
        self.N_ref_var = tk.DoubleVar(value=1000.0)
        self.Kb_var = tk.DoubleVar(value=0.055)
        self.HP_var = tk.DoubleVar(value=5.0)
        self.Ra_var = tk.DoubleVar(value=0.5)
        self.La_var = tk.DoubleVar(value=0.02)
        self.J_var = tk.DoubleVar(value=0.05)
        self.alpha_slider_var = tk.DoubleVar(value=59.35)
        self.TL_var = tk.DoubleVar(value=5.0)
        self.Kp_var = tk.DoubleVar(value=2.0)
        self.Ki_var = tk.DoubleVar(value=0.5)
        self.Kd_var = tk.DoubleVar(value=0.1)
        self.Rth_var = tk.DoubleVar(value=0.5)
        self.Cth_var = tk.DoubleVar(value=500.0)
        self.Ta_var = tk.DoubleVar(value=25.0)
        self.tariff_var = tk.DoubleVar(value=0.12)
        self.hours_day_var = tk.DoubleVar(value=8.0)
        self.days_year_var = tk.DoubleVar(value=250.0)
        self.Ip_var = tk.DoubleVar(value=34.0)
        self.TMS_var = tk.DoubleVar(value=0.1)
        self.fuse_rating_var = tk.DoubleVar(value=50.0)
        self.Ra_fault_var = tk.DoubleVar(value=0.5)

        self._sim_running = False
        self._sim_after_id = None

        self._setup_style()
        self._build_ui()
        self.bind("<Configure>", self._on_resize)

        self._update_status()

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        bg = "#1a2636"
        fg = "#e0eaf5"
        accent = "#2a6496"
        btn = "#1f4e79"
        style.configure(".", background=bg, foreground=fg, font=("Segoe UI", 10))
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TLabelframe", background=bg, foreground=fg)
        style.configure("TLabelframe.Label", background=bg, foreground="#7ec8e3")
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background="#0d1b2a", foreground=fg,
                        padding=[12, 6], font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", accent)],
                  foreground=[("selected", "white")])
        style.configure("TButton", background=btn, foreground="white",
                        font=("Segoe UI", 9, "bold"), padding=6)
        style.map("TButton", background=[("active", accent)])
        style.configure("TScale", background=bg, troughcolor="#0d1b2a")
        style.configure("TEntry", fieldbackground="#0d1b2a", foreground=fg)
        style.configure("TScrollbar", background=bg, troughcolor="#0d1b2a")
        style.configure("Status.TLabel", background="#0d1b2a", foreground="#7ec8e3",
                        font=("Segoe UI", 9), padding=[6, 3])
        self.configure(bg=bg)
        self._bg = bg
        self._fg = fg
        self._accent = accent

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._tab_overview()
        self._tab_parameters()
        self._tab_converter()
        self._tab_simulation()
        self._tab_fault()
        self._tab_protection()
        self._tab_speed_controller()
        self._tab_thermal_economic()
        self._tab_harmonics()

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, style="Status.TLabel",
                               anchor="w")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Tab 1: Overview ──────────────────────────────────────────────────────
    def _tab_overview(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="\U0001f4cb Overview")

        text_widget = tk.Text(frame, wrap=tk.WORD, bg="#0d1b2a", fg="#e0eaf5",
                              font=("Consolas", 11), relief=tk.FLAT, padx=20, pady=20)
        sb = ttk.Scrollbar(frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Tags
        text_widget.tag_configure("title", font=("Segoe UI", 16, "bold"),
                                  foreground="#7ec8e3")
        text_widget.tag_configure("heading", font=("Segoe UI", 12, "bold"),
                                  foreground="#f0a500")
        text_widget.tag_configure("result", font=("Consolas", 11),
                                  foreground="#90ee90")
        text_widget.tag_configure("label", font=("Consolas", 11),
                                  foreground="#e0eaf5")
        text_widget.tag_configure("formula", font=("Consolas", 10, "italic"),
                                  foreground="#c9a0dc")

        content = [
            ("title", "DC Motor Speed Control \u2013 Full Bridge Converter Analysis\n\n"),
            ("heading", "Problem Statement\n"),
            ("label", "  \u2022 Motor rating : 110 V, 5 hp DC motor\n"),
            ("label", "  \u2022 Supply        : 120 V rms, 60 Hz single-phase AC\n"),
            ("label", "  \u2022 Converter     : Single-phase full-bridge (fully controlled)\n"),
            ("label", "  \u2022 Motor Kb      : 0.055 V/rpm\n"),
            ("label", "  \u2022 Assumption    : Ideal motor (Ra \u2248 0), high La (continuous Ia)\n\n"),
            ("heading", "Given Data\n"),
            ("formula", "  Vs_rms = 120 V,  f = 60 Hz,  N_ref = 1000 rpm\n\n"),
            ("heading", "Step-by-Step Solution\n"),
            ("label", "1. Peak supply voltage:\n"),
            ("formula", "     Vm = \u221a2 \u00d7 Vs = \u221a2 \u00d7 120 = 169.706 V\n\n"),
            ("label", "2. Maximum DC output voltage (full-bridge, \u03b1=0):\n"),
            ("formula", "     Vdc_max = 2Vm/\u03c0 = 2\u00d7169.706/\u03c0 = 108.06 V\n\n"),
            ("label", "3. Rated armature current:\n"),
            ("formula", "     Prated = 5 \u00d7 746 = 3730 W\n"),
            ("formula", "     Ia_rated = P/V = 3730/110 = 33.91 A\n\n"),
            ("label", "4. Back-EMF at N = 1000 rpm:\n"),
            ("formula", "     E = Kb \u00d7 N = 0.055 \u00d7 1000 = 55 V\n\n"),
            ("label", "5. Required DC output voltage (Ra \u2248 0):\n"),
            ("formula", "     Vdc_req = E + Ia\u00d7Ra = 55 + 0 = 55 V\n\n"),
            ("label", "6. Firing angle:\n"),
            ("formula", "     \u03b1 = arccos(Vdc/Vdc_max) = arccos(55/108.06)\n"),
            ("result", "     \u03b1 = 59.35\u00b0\n\n"),
            ("label", "7. Supply RMS current (square wave, high La):\n"),
            ("formula", "     Is_rms = Ia = 33.91 A\n\n"),
            ("label", "8. Thyristor average and RMS currents:\n"),
            ("formula", "     IT_avg = Ia/2 = 16.95 A\n"),
            ("formula", "     IT_rms = Ia/\u221a2 = 23.99 A\n\n"),
            ("label", "9. Power factor:\n"),
            ("formula", "     PF = (2\u221a2/\u03c0) \u00d7 cos(\u03b1) = 0.9003 \u00d7 cos(59.35\u00b0)\n"),
            ("result", "     PF = 0.4582\n\n"),
            ("label", "10. Input power and apparent power:\n"),
            ("formula", "     P = Vdc \u00d7 Ia = 55 \u00d7 33.91 = 1865 W\n"),
            ("formula", "     S = Vs \u00d7 Is_rms = 120 \u00d7 33.91 = 4069 VA\n\n"),
            ("label", "11. Total Harmonic Distortion:\n"),
            ("formula", "     THD = \u221a(\u03c0\u00b2/8 \u2212 1) \u00d7 100\n"),
            ("result", "     THD = 48.43 %\n\n"),
            ("heading", "Summary of Results\n"),
            ("result", "  Vm         = 169.706 V\n"),
            ("result", "  Vdc_max    = 108.06  V\n"),
            ("result", "  Vdc_req    =  55.00  V\n"),
            ("result", "  \u03b1          =  59.35  \u00b0\n"),
            ("result", "  Ia_rated   =  33.91  A\n"),
            ("result", "  Is_rms     =  33.91  A\n"),
            ("result", "  IT_avg     =  16.95  A\n"),
            ("result", "  IT_rms     =  23.99  A\n"),
            ("result", "  PF         =   0.458\n"),
            ("result", "  P_in       = 1865    W\n"),
            ("result", "  S          = 4069    VA\n"),
            ("result", "  THD        =  48.43  %\n\n"),
            ("heading", "Notes\n"),
            ("label", "  \u2022 Continuous conduction assumed throughout (large La).\n"),
            ("label", "  \u2022 All four thyristors of the H-bridge are ideal switches.\n"),
            ("label", "  \u2022 freewheeling is not used; average output is 2Vm/\u03c0 \u00d7 cos(\u03b1).\n"),
            ("label", "  \u2022 PF is low at large \u03b1; capacitor compensation may improve it.\n"),
            ("label", "  \u2022 High THD (48%) indicates significant harmonic injection.\n"),
        ]
        for tag, text in content:
            text_widget.insert(tk.END, text, tag)
        text_widget.configure(state=tk.DISABLED)

    # ── Tab 2: Parameters ────────────────────────────────────────────────────
    def _tab_parameters(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="\u2699\ufe0f Parameters")

        left = ttk.LabelFrame(frame, text=" Motor & Converter Parameters ")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        sliders = [
            ("Vs (V)", self.Vs_var, 60, 240, 1),
            ("Freq (Hz)", self.freq_var, 50, 60, 10),
            ("N_ref (rpm)", self.N_ref_var, 0, 2000, 1),
            ("Kb (V/rpm)", self.Kb_var, 0.01, 0.2, 0.001),
            ("HP", self.HP_var, 1, 20, 1),
            ("Ra (\u03a9)", self.Ra_var, 0.0, 5.0, 0.01),
            ("La (H)", self.La_var, 0.001, 0.1, 0.001),
            ("J (kg\u00b7m\u00b2)", self.J_var, 0.001, 1.0, 0.001),
        ]

        for label, var, from_, to, res in sliders:
            row = ttk.Frame(left)
            row.pack(fill=tk.X, padx=8, pady=4)
            ttk.Label(row, text=label, width=14, anchor="w").pack(side=tk.LEFT)
            sl = ttk.Scale(row, variable=var, from_=from_, to=to,
                           orient=tk.HORIZONTAL, length=220,
                           command=lambda v, lbl=label, variable=var: self._param_changed())
            sl.pack(side=tk.LEFT, padx=4)
            val_lbl = ttk.Label(row, width=8, anchor="e")
            val_lbl.pack(side=tk.LEFT)
            var.trace_add("write", lambda *a, v=var, l=val_lbl: l.config(text=f"{v.get():.3f}"))
            val_lbl.config(text=f"{var.get():.3f}")

        ttk.Button(left, text="Calculate", command=self._calc_params).pack(pady=10)

        right = ttk.LabelFrame(frame, text=" Calculation Results ")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.params_result_text = tk.Text(right, bg="#0d1b2a", fg="#90ee90",
                                          font=("Consolas", 11), relief=tk.FLAT,
                                          padx=10, pady=10)
        self.params_result_text.pack(fill=tk.BOTH, expand=True)
        self._calc_params()

    def _param_changed(self):
        # Reserved for future live-update behaviour on slider drag
        pass

    def _calc_params(self):
        r = DCMotorCalc.calc_all(
            self.Vs_var.get(), self.freq_var.get(), self.N_ref_var.get(),
            self.Kb_var.get(), self.HP_var.get(), self.Ra_var.get(),
            self.La_var.get(), self.J_var.get()
        )
        lines = [
            f"{'='*45}",
            f" DC Motor Full-Bridge Converter Results",
            f"{'='*45}",
            f" Supply voltage (Vs)    : {r['Vs']:.2f} V rms",
            f" Frequency              : {r['freq']:.0f} Hz",
            f" Peak voltage (Vm)      : {r['Vm']:.3f} V",
            f" Max DC voltage (Vdc_max): {r['Vdc_max']:.3f} V",
            f"{'─'*45}",
            f" Motor HP               : {r['HP']:.1f} hp",
            f" Rated power            : {r['Prated']:.1f} W",
            f" Rated armature current : {r['Ia_rated']:.3f} A",
            f" Reference speed (N_ref): {r['N_ref']:.1f} rpm",
            f" Kb                     : {r['Kb']:.4f} V/rpm",
            f" Kb_SI                  : {r['Kb_SI']:.4f} V/(rad/s)",
            f" Back-EMF (E)           : {r['E']:.3f} V",
            f" Ra                     : {r['Ra']:.3f} \u03a9",
            f" Required Vdc           : {r['Vdc_req']:.3f} V",
            f"{'─'*45}",
            f" Firing angle (\u03b1)       : {r['alpha_deg']:.3f} \u00b0",
            f" Supply I_rms           : {r['Is_rms']:.3f} A",
            f" Thyristor I_avg        : {r['IT_avg']:.3f} A",
            f" Thyristor I_rms        : {r['IT_rms']:.3f} A",
            f" Power factor (PF)      : {r['PF']:.4f}",
            f" Input power (P)        : {r['P_in']:.2f} W",
            f" Apparent power (S)     : {r['S']:.2f} VA",
            f" THD                    : {r['THD']:.2f} %",
            f"{'='*45}",
        ]
        self.params_result_text.configure(state=tk.NORMAL)
        self.params_result_text.delete("1.0", tk.END)
        self.params_result_text.insert(tk.END, "\n".join(lines))
        self.params_result_text.configure(state=tk.DISABLED)
        self._update_status()

    # ── Tab 3: Converter Analysis ────────────────────────────────────────────
    def _tab_converter(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="\U0001f50c Converter Analysis")

        # Left: plots
        plot_frame = ttk.Frame(frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.conv_fig = Figure(figsize=(8, 6), facecolor="#0d1b2a")
        self.conv_canvas = FigureCanvasTkAgg(self.conv_fig, master=plot_frame)
        toolbar = NavigationToolbar2Tk(self.conv_canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.conv_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: results + slider
        right = ttk.Frame(frame, width=240)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        right.pack_propagate(False)

        ttk.Label(right, text="\u03b1 (degrees)", font=("Segoe UI", 10, "bold")).pack(pady=(10, 2))
        self.alpha_disp = ttk.Label(right, text="59.35\u00b0",
                                    font=("Consolas", 14, "bold"),
                                    foreground="#f0a500")
        self.alpha_disp.pack()
        ttk.Scale(right, variable=self.alpha_slider_var, from_=0, to=180,
                  orient=tk.HORIZONTAL, length=200,
                  command=self._conv_alpha_changed).pack(pady=4)

        self.conv_results_frame = ttk.LabelFrame(right, text=" Results ")
        self.conv_results_frame.pack(fill=tk.X, padx=4, pady=10)
        self.conv_result_labels = {}
        for key in ["Vdc", "Is_rms", "IT_avg", "IT_rms", "PF", "THD"]:
            row = ttk.Frame(self.conv_results_frame)
            row.pack(fill=tk.X, padx=4, pady=2)
            ttk.Label(row, text=key, width=10, anchor="w").pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="--", foreground="#90ee90",
                            font=("Consolas", 10), anchor="e", width=12)
            lbl.pack(side=tk.RIGHT)
            self.conv_result_labels[key] = lbl

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self._draw_converter()

    def _on_tab_change(self, event):
        sel = self.notebook.index(self.notebook.select())
        if sel == 2:
            self._draw_converter()

    def _conv_alpha_changed(self, val=None):
        a = self.alpha_slider_var.get()
        self.alpha_disp.config(text=f"{a:.1f}\u00b0")
        self._draw_converter()

    def _draw_converter(self):
        alpha_deg = self.alpha_slider_var.get()
        Vs = self.Vs_var.get()
        Ia = DCMotorCalc.calc_all(
            Vs, self.freq_var.get(), self.N_ref_var.get(),
            self.Kb_var.get(), self.HP_var.get(), self.Ra_var.get(),
            self.La_var.get(), self.J_var.get()
        )["Ia_rated"]
        r = DCMotorCalc.calc_from_alpha(Vs, alpha_deg, Ia)

        self.conv_fig.clear()
        axes = self.conv_fig.subplots(3, 1)
        theta = np.linspace(0, 2 * np.pi, 1000)
        Vm = r["Vm"]
        alpha_r = r["alpha_rad"]
        vs = Vm * np.sin(theta)

        # Rectified output
        vout = np.zeros_like(theta)
        for i, t in enumerate(theta):
            ang = t % (2 * np.pi)
            if alpha_r <= ang <= np.pi:
                vout[i] = Vm * np.sin(ang)
            elif np.pi + alpha_r <= ang <= 2 * np.pi:
                vout[i] = -Vm * np.sin(ang)
            else:
                vout[i] = 0.0

        ax1 = axes[0]
        ax1.set_facecolor("#0d1b2a")
        ax1.plot(np.degrees(theta), vs, color="#4fc3f7", lw=1.5, label="Vs")
        ax1.plot(np.degrees(theta), vout, color="#f0a500", lw=2, label="Vdc")
        ax1.axhline(r["Vdc"], color="#90ee90", ls="--", lw=1,
                    label=f"Vdc_avg={r['Vdc']:.1f}V")
        ax1.axvline(np.degrees(alpha_r), color="red", ls=":", lw=1.5,
                    label=f"\u03b1={alpha_deg:.1f}\u00b0")
        ax1.axvline(180 + np.degrees(alpha_r), color="red", ls=":", lw=1.5)
        ax1.set_xlim(0, 360)
        ax1.set_title("AC Supply & Rectified Output", color="#e0eaf5", fontsize=9)
        ax1.set_ylabel("Voltage (V)", color="#e0eaf5", fontsize=8)
        ax1.tick_params(colors="#e0eaf5", labelsize=7)
        ax1.legend(fontsize=7, facecolor="#1a2636", edgecolor="#4fc3f7",
                   labelcolor="#e0eaf5")
        for sp in ax1.spines.values():
            sp.set_color("#2a6496")

        # Supply current
        ax2 = axes[1]
        ax2.set_facecolor("#0d1b2a")
        is_wave = np.zeros_like(theta)
        for i, t in enumerate(theta):
            ang = t % (2 * np.pi)
            if alpha_r <= ang <= np.pi + alpha_r:
                is_wave[i] = Ia
            else:
                is_wave[i] = -Ia
        ax2.plot(np.degrees(theta), is_wave, color="#c9a0dc", lw=2, label="Is")
        ax2.axhline(0, color="#e0eaf5", lw=0.5)
        ax2.set_xlim(0, 360)
        ax2.set_title("Supply Current (Square Wave)", color="#e0eaf5", fontsize=9)
        ax2.set_ylabel("Current (A)", color="#e0eaf5", fontsize=8)
        ax2.tick_params(colors="#e0eaf5", labelsize=7)
        ax2.legend(fontsize=7, facecolor="#1a2636", edgecolor="#4fc3f7",
                   labelcolor="#e0eaf5")
        for sp in ax2.spines.values():
            sp.set_color("#2a6496")

        # Firing pulses
        ax3 = axes[2]
        ax3.set_facecolor("#0d1b2a")
        pulse = np.zeros_like(theta)
        w = np.radians(5)
        for i, t in enumerate(theta):
            ang = t % (2 * np.pi)
            if abs(ang - alpha_r) < w or abs(ang - (np.pi + alpha_r)) < w:
                pulse[i] = 1.0
        ax3.fill_between(np.degrees(theta), 0, pulse, color="#f0a500", alpha=0.8)
        ax3.set_xlim(0, 360)
        ax3.set_ylim(-0.1, 1.4)
        ax3.set_title("Firing Pulses (T1,T3 / T2,T4)", color="#e0eaf5", fontsize=9)
        ax3.set_ylabel("Gate Signal", color="#e0eaf5", fontsize=8)
        ax3.set_xlabel("Angle (degrees)", color="#e0eaf5", fontsize=8)
        ax3.tick_params(colors="#e0eaf5", labelsize=7)
        for sp in ax3.spines.values():
            sp.set_color("#2a6496")

        self.conv_fig.patch.set_facecolor("#0d1b2a")
        self.conv_fig.tight_layout(pad=1.5)
        self.conv_canvas.draw()

        # Update result labels
        units = {"Vdc": "V", "Is_rms": "A", "IT_avg": "A",
                 "IT_rms": "A", "PF": "", "THD": "%"}
        for key, lbl in self.conv_result_labels.items():
            val = r.get(key, 0)
            lbl.config(text=f"{val:.3f} {units[key]}")

    # ── Tab 4: Modeling & Simulation ─────────────────────────────────────────
    def _tab_simulation(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="\U0001f4e1 Modeling & Simulation")

        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        ttk.Button(ctrl, text="Start", command=self._sim_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Stop", command=self._sim_stop).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Reset", command=self._sim_reset).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="  TL (N\u00b7m):").pack(side=tk.LEFT)
        self.TL_label = ttk.Label(ctrl, text="5.00", foreground="#f0a500",
                                  font=("Consolas", 10))
        self.TL_label.pack(side=tk.LEFT)
        ttk.Scale(ctrl, variable=self.TL_var, from_=0, to=20,
                  orient=tk.HORIZONTAL, length=150,
                  command=lambda v: self.TL_label.config(
                      text=f"{self.TL_var.get():.2f}")).pack(side=tk.LEFT, padx=4)

        self.sim_fig = Figure(figsize=(8, 5), facecolor="#0d1b2a")
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, master=frame)
        toolbar = NavigationToolbar2Tk(self.sim_canvas, frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.sim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._sim_t = []
        self._sim_ia = []
        self._sim_omega = []
        self._draw_simulation_empty()

    def _draw_simulation_empty(self):
        self.sim_fig.clear()
        ax1, ax2 = self.sim_fig.subplots(2, 1)
        for ax, ylabel, title in [
            (ax1, "Ia (A)", "Armature Current"),
            (ax2, "\u03c9 (rpm)", "Rotor Speed"),
        ]:
            ax.set_facecolor("#0d1b2a")
            ax.set_ylabel(ylabel, color="#e0eaf5", fontsize=8)
            ax.set_title(title, color="#e0eaf5", fontsize=9)
            ax.tick_params(colors="#e0eaf5", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#2a6496")
        ax2.set_xlabel("Time (s)", color="#e0eaf5", fontsize=8)
        self.sim_fig.patch.set_facecolor("#0d1b2a")
        self.sim_fig.tight_layout(pad=1.5)
        self.sim_canvas.draw()

    def _sim_start(self):
        if self._sim_running:
            return
        self._sim_running = True
        r = DCMotorCalc.calc_all(
            self.Vs_var.get(), self.freq_var.get(), self.N_ref_var.get(),
            self.Kb_var.get(), self.HP_var.get(), self.Ra_var.get(),
            self.La_var.get(), self.J_var.get()
        )
        Vdc = r["Vdc_req"]
        Ra = r["Ra"]
        La = r["La"]
        J = r["J"]
        Kb_SI = r["Kb_SI"]
        TL = self.TL_var.get()
        Bfr = 0.01

        def odes(t, y):
            Ia, omega = y
            dIa = (Vdc - Ra * Ia - Kb_SI * omega) / La
            domega = (Kb_SI * Ia - TL - Bfr * omega) / J
            return [dIa, domega]

        t_span = (0, 3.0)
        t_eval = np.linspace(0, 3.0, 1500)
        y0 = [0.0, 0.0]
        try:
            sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method="RK45")
            self._sim_t = sol.t
            self._sim_ia = sol.y[0]
            self._sim_omega = sol.y[1] * 60 / (2 * math.pi)
        except (ValueError, RuntimeError) as exc:
            self._sim_t = np.array([])
            self._sim_ia = np.array([])
            self._sim_omega = np.array([])
            self.status_var.set(f"  Simulation error: {exc}")

        self._draw_simulation()
        self._sim_running = False

    def _sim_stop(self):
        self._sim_running = False
        if self._sim_after_id:
            self.after_cancel(self._sim_after_id)
            self._sim_after_id = None

    def _sim_reset(self):
        self._sim_stop()
        self._sim_t = []
        self._sim_ia = []
        self._sim_omega = []
        self._draw_simulation_empty()

    def _draw_simulation(self):
        self.sim_fig.clear()
        ax1, ax2 = self.sim_fig.subplots(2, 1)

        if len(self._sim_t) > 0:
            ax1.plot(self._sim_t, self._sim_ia, color="#f0a500", lw=1.5)
            ax2.plot(self._sim_t, self._sim_omega, color="#4fc3f7", lw=1.5)

        for ax, ylabel, title in [
            (ax1, "Ia (A)", "Armature Current"),
            (ax2, "Speed (rpm)", "Rotor Speed"),
        ]:
            ax.set_facecolor("#0d1b2a")
            ax.set_ylabel(ylabel, color="#e0eaf5", fontsize=8)
            ax.set_title(title, color="#e0eaf5", fontsize=9)
            ax.tick_params(colors="#e0eaf5", labelsize=7)
            ax.grid(True, color="#1a2636", lw=0.5)
            for sp in ax.spines.values():
                sp.set_color("#2a6496")
        ax2.set_xlabel("Time (s)", color="#e0eaf5", fontsize=8)
        self.sim_fig.patch.set_facecolor("#0d1b2a")
        self.sim_fig.tight_layout(pad=1.5)
        self.sim_canvas.draw()

    # ── Tab 5: Fault Current ──────────────────────────────────────────────────
    def _tab_fault(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="\u26a1 Fault Current")

        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(ctrl, text="Ra (\u03a9):").pack(side=tk.LEFT)
        self.Ra_fault_lbl = ttk.Label(ctrl, text=f"{self.Ra_fault_var.get():.3f}",
                                      foreground="#f0a500", font=("Consolas", 10))
        self.Ra_fault_lbl.pack(side=tk.LEFT, padx=4)
        ttk.Scale(ctrl, variable=self.Ra_fault_var, from_=0.01, to=5.0,
                  orient=tk.HORIZONTAL, length=200,
                  command=self._fault_ra_changed).pack(side=tk.LEFT, padx=4)

        ttk.Button(ctrl, text="Plot Fault Current", command=self._draw_fault).pack(
            side=tk.LEFT, padx=10)

        self.fault_info = ttk.Label(ctrl, text="", foreground="#90ee90",
                                    font=("Consolas", 10))
        self.fault_info.pack(side=tk.LEFT, padx=8)

        self.fault_fig = Figure(figsize=(8, 4.5), facecolor="#0d1b2a")
        self.fault_canvas = FigureCanvasTkAgg(self.fault_fig, master=frame)
        toolbar = NavigationToolbar2Tk(self.fault_canvas, frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.fault_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_fault()

    def _fault_ra_changed(self, val=None):
        self.Ra_fault_lbl.config(text=f"{self.Ra_fault_var.get():.3f}")
        self._draw_fault()

    def _draw_fault(self):
        Ra = max(self.Ra_fault_var.get(), 0.001)
        La = self.La_var.get()
        Vdc = DCMotorCalc.calc_all(
            self.Vs_var.get(), self.freq_var.get(), self.N_ref_var.get(),
            self.Kb_var.get(), self.HP_var.get(), Ra,
            La, self.J_var.get()
        )["Vdc_req"]
        tau = La / Ra
        I_peak = Vdc / Ra
        t = np.linspace(0, 5 * tau, 500)
        i_fault = I_peak * (1 - np.exp(-t / tau))

        self.fault_fig.clear()
        ax = self.fault_fig.add_subplot(111)
        ax.set_facecolor("#0d1b2a")
        ax.plot(t * 1000, i_fault, color="#f0a500", lw=2,
                label=f"i(t) = {I_peak:.1f}(1-e^-t/{tau*1000:.1f}ms)")
        ax.axhline(I_peak, color="#90ee90", ls="--", lw=1,
                   label=f"I_peak = {I_peak:.2f} A")
        ax.axhline(0.632 * I_peak, color="#c9a0dc", ls=":", lw=1,
                   label=f"0.632\u00d7I_peak at \u03c4={tau*1000:.2f} ms")
        ax.axvline(tau * 1000, color="#4fc3f7", ls=":", lw=1)
        ax.set_xlabel("Time (ms)", color="#e0eaf5", fontsize=9)
        ax.set_ylabel("Current (A)", color="#e0eaf5", fontsize=9)
        ax.set_title("Armature Short-Circuit Transient Current", color="#e0eaf5", fontsize=10)
        ax.legend(fontsize=8, facecolor="#1a2636", edgecolor="#4fc3f7",
                  labelcolor="#e0eaf5")
        ax.grid(True, color="#1a2636", lw=0.5)
        ax.tick_params(colors="#e0eaf5", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#2a6496")
        self.fault_fig.patch.set_facecolor("#0d1b2a")
        self.fault_fig.tight_layout(pad=1.5)
        self.fault_canvas.draw()
        self.fault_info.config(
            text=f"I_peak = {I_peak:.2f} A   \u03c4 = {tau*1000:.2f} ms")

    # ── Tab 6: Protection Coordination ───────────────────────────────────────
    def _tab_protection(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="\U0001f6e1 Protection")

        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        params = [
            ("Ip (A)", self.Ip_var, 1, 100, 1),
            ("TMS", self.TMS_var, 0.05, 1.0, 0.01),
            ("Fuse (A)", self.fuse_rating_var, 10, 200, 1),
        ]
        for label, var, from_, to, res in params:
            ttk.Label(ctrl, text=f"  {label}:").pack(side=tk.LEFT)
            lbl = ttk.Label(ctrl, text=f"{var.get():.2f}",
                            foreground="#f0a500", font=("Consolas", 10), width=6)
            lbl.pack(side=tk.LEFT)
            ttk.Scale(ctrl, variable=var, from_=from_, to=to,
                      orient=tk.HORIZONTAL, length=120,
                      command=lambda v, l=lbl, variable=var: (
                          l.config(text=f"{variable.get():.2f}"),
                          self._draw_protection()
                      )).pack(side=tk.LEFT, padx=2)

        self.prot_fig = Figure(figsize=(8, 5), facecolor="#0d1b2a")
        self.prot_canvas = FigureCanvasTkAgg(self.prot_fig, master=frame)
        toolbar = NavigationToolbar2Tk(self.prot_canvas, frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.prot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_protection()

    def _draw_protection(self):
        Ip = max(self.Ip_var.get(), 0.1)
        TMS = self.TMS_var.get()
        fuse_rating = self.fuse_rating_var.get()

        I = np.logspace(np.log10(max(1.1 * Ip, Ip + 0.1)), np.log10(20 * Ip), 500)
        # IEC 60255 Normal Inverse
        ratio = I / Ip
        relay_t = TMS * 0.14 / (ratio**0.02 - 1)

        # Fuse: simplified t = k / (I/Ir - 1)^1.5
        Ir = fuse_rating
        fuse_mask = I > 1.1 * Ir
        fuse_I = I[fuse_mask]
        fuse_t = 10 / ((fuse_I / Ir - 1) ** 1.5 + 0.1)

        self.prot_fig.clear()
        ax = self.prot_fig.add_subplot(111)
        ax.set_facecolor("#0d1b2a")
        ax.loglog(I, relay_t, color="#4fc3f7", lw=2,
                  label=f"Relay IEC NI (Ip={Ip:.0f}A, TMS={TMS:.2f})")
        if len(fuse_I) > 0:
            ax.loglog(fuse_I, fuse_t, color="#f0a500", lw=2,
                      label=f"Fuse ({fuse_rating:.0f} A)")
        ax.axvline(Ip, color="red", ls=":", lw=1.5, label=f"Pickup Ip={Ip:.0f}A")
        ax.set_xlabel("Current (A)", color="#e0eaf5", fontsize=9)
        ax.set_ylabel("Time (s)", color="#e0eaf5", fontsize=9)
        ax.set_title("Protection Coordination \u2013 IEC 60255 Normal Inverse",
                     color="#e0eaf5", fontsize=10)
        ax.legend(fontsize=8, facecolor="#1a2636", edgecolor="#4fc3f7",
                  labelcolor="#e0eaf5")
        ax.grid(True, which="both", color="#1a2636", lw=0.5)
        ax.tick_params(colors="#e0eaf5", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#2a6496")
        self.prot_fig.patch.set_facecolor("#0d1b2a")
        self.prot_fig.tight_layout(pad=1.5)
        self.prot_canvas.draw()

    # ── Tab 7: Speed Controller ───────────────────────────────────────────────
    def _tab_speed_controller(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="\U0001f39b Speed Controller")

        sub_nb = ttk.Notebook(frame)
        sub_nb.pack(fill=tk.BOTH, expand=True)

        # PID sub-tab
        pid_frame = ttk.Frame(sub_nb)
        sub_nb.add(pid_frame, text="PID Controller")
        self._build_pid_tab(pid_frame)

        # Fuzzy sub-tab
        fuzzy_frame = ttk.Frame(sub_nb)
        sub_nb.add(fuzzy_frame, text="Fuzzy Controller")
        self._build_fuzzy_tab(fuzzy_frame)

    def _build_pid_tab(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        pid_params = [
            ("Kp", self.Kp_var, 0.1, 20.0),
            ("Ki", self.Ki_var, 0.0, 5.0),
            ("Kd", self.Kd_var, 0.0, 2.0),
        ]
        for label, var, from_, to in pid_params:
            ttk.Label(ctrl, text=f"  {label}:").pack(side=tk.LEFT)
            lbl = ttk.Label(ctrl, text=f"{var.get():.2f}",
                            foreground="#f0a500", font=("Consolas", 10), width=6)
            lbl.pack(side=tk.LEFT)
            ttk.Scale(ctrl, variable=var, from_=from_, to=to,
                      orient=tk.HORIZONTAL, length=120,
                      command=lambda v, l=lbl, variable=var: (
                          l.config(text=f"{variable.get():.2f}"),
                      )).pack(side=tk.LEFT, padx=2)

        ttk.Button(ctrl, text="Simulate PID",
                   command=self._draw_pid).pack(side=tk.LEFT, padx=10)

        self.pid_fig = Figure(figsize=(8, 5), facecolor="#0d1b2a")
        self.pid_canvas = FigureCanvasTkAgg(self.pid_fig, master=parent)
        toolbar = NavigationToolbar2Tk(self.pid_canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.pid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_pid()

    def _draw_pid(self):
        N_ref = self.N_ref_var.get()
        Kp = self.Kp_var.get()
        Ki = self.Ki_var.get()
        Kd = self.Kd_var.get()
        r = DCMotorCalc.calc_all(
            self.Vs_var.get(), self.freq_var.get(), N_ref,
            self.Kb_var.get(), self.HP_var.get(), self.Ra_var.get(),
            self.La_var.get(), self.J_var.get()
        )
        Kb_SI = r["Kb_SI"]
        Ra = r["Ra"]
        La = r["La"]
        J = r["J"]
        Bfr = 0.01
        Vdc_max = r["Vdc_max"]
        omega_ref = N_ref * 2 * math.pi / 60

        dt = 0.001
        t_end = 5.0
        N = int(t_end / dt)
        t_arr = np.linspace(0, t_end, N)
        omega_arr = np.zeros(N)
        ia_arr = np.zeros(N)
        u_arr = np.zeros(N)
        omega = 0.0
        ia = 0.0
        integral = 0.0
        prev_err = 0.0
        TL_step = self.TL_var.get()

        for i in range(1, N):
            t = t_arr[i]
            TL = TL_step if t > 2.0 else 0.0
            err = omega_ref - omega
            integral += err * dt
            derivative = (err - prev_err) / dt
            u = Kp * err + Ki * integral + Kd * derivative
            u = max(0.0, min(u, Vdc_max))
            prev_err = err
            dia = (u - Ra * ia - Kb_SI * omega) / La
            domega = (Kb_SI * ia - TL - Bfr * omega) / J
            ia += dia * dt
            omega += domega * dt
            omega = max(omega, 0.0)
            ia = max(ia, 0.0)
            omega_arr[i] = omega * 60 / (2 * math.pi)
            ia_arr[i] = ia
            u_arr[i] = u

        self.pid_fig.clear()
        ax1, ax2 = self.pid_fig.subplots(2, 1)
        ax1.plot(t_arr, omega_arr, color="#4fc3f7", lw=1.5, label="Speed (rpm)")
        ax1.axhline(N_ref, color="#90ee90", ls="--", lw=1,
                    label=f"N_ref = {N_ref:.0f} rpm")
        ax1.axvline(2.0, color="red", ls=":", lw=1, label="Load step")
        ax2.plot(t_arr, u_arr, color="#f0a500", lw=1.5, label="Control effort (Vdc)")

        for ax, ylabel, title in [
            (ax1, "Speed (rpm)", f"PID Speed Response (Kp={Kp:.1f}, Ki={Ki:.1f}, Kd={Kd:.1f})"),
            (ax2, "Vdc (V)", "Control Effort"),
        ]:
            ax.set_facecolor("#0d1b2a")
            ax.set_ylabel(ylabel, color="#e0eaf5", fontsize=8)
            ax.set_title(title, color="#e0eaf5", fontsize=9)
            ax.tick_params(colors="#e0eaf5", labelsize=7)
            ax.grid(True, color="#1a2636", lw=0.5)
            ax.legend(fontsize=7, facecolor="#1a2636", edgecolor="#4fc3f7",
                      labelcolor="#e0eaf5")
            for sp in ax.spines.values():
                sp.set_color("#2a6496")
        ax2.set_xlabel("Time (s)", color="#e0eaf5", fontsize=8)
        self.pid_fig.patch.set_facecolor("#0d1b2a")
        self.pid_fig.tight_layout(pad=1.5)
        self.pid_canvas.draw()

    def _build_fuzzy_tab(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        ttk.Button(ctrl, text="Run Fuzzy vs PID Comparison",
                   command=self._draw_fuzzy).pack(side=tk.LEFT, padx=8)

        self.fuzzy_fig = Figure(figsize=(8, 5), facecolor="#0d1b2a")
        self.fuzzy_canvas = FigureCanvasTkAgg(self.fuzzy_fig, master=parent)
        toolbar = NavigationToolbar2Tk(self.fuzzy_canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.fuzzy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_fuzzy()

    def _fuzzy_membership(self, x, x_min, x_max):
        """Return NB,NS,ZO,PS,PB membership degrees for value x."""
        centers = np.linspace(x_min, x_max, 5)
        sigma = (x_max - x_min) / 8.0
        mf = np.exp(-0.5 * ((x - centers) / sigma) ** 2)
        mf = mf / (mf.sum() + _EPSILON)
        return mf  # [NB, NS, ZO, PS, PB]

    def _fuzzy_output(self, error, d_error, e_max, de_max, u_max):
        """Simplified Mamdani fuzzy inference (5x5 rule table)."""
        rule_table = np.array([
            [0, 0, 0, 1, 2],
            [0, 0, 1, 2, 3],
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 4],
            [2, 3, 4, 4, 4],
        ])
        centers_u = np.linspace(-u_max, u_max, 5)
        mf_e = self._fuzzy_membership(error, -e_max, e_max)
        mf_de = self._fuzzy_membership(d_error, -de_max, de_max)
        num = 0.0
        den = 0.0
        for i in range(5):
            for j in range(5):
                strength = min(mf_e[i], mf_de[j])
                out_center = centers_u[rule_table[i, j]]
                num += strength * out_center
                den += strength
        if den < _EPSILON:
            return 0.0
        return num / den

    def _draw_fuzzy(self):
        N_ref = self.N_ref_var.get()
        r = DCMotorCalc.calc_all(
            self.Vs_var.get(), self.freq_var.get(), N_ref,
            self.Kb_var.get(), self.HP_var.get(), self.Ra_var.get(),
            self.La_var.get(), self.J_var.get()
        )
        Kb_SI = r["Kb_SI"]
        Ra = r["Ra"]
        La = r["La"]
        J = r["J"]
        Bfr = 0.01
        Vdc_max = r["Vdc_max"]
        omega_ref = N_ref * 2 * math.pi / 60
        e_max = omega_ref if omega_ref > 1 else 100.0
        de_max = e_max * 10
        Kp = self.Kp_var.get()
        Ki = self.Ki_var.get()
        Kd = self.Kd_var.get()

        dt = 0.002
        t_end = 5.0
        N = int(t_end / dt)
        t_arr = np.linspace(0, t_end, N)

        # PID
        pid_speed = np.zeros(N)
        omega = 0.0
        ia = 0.0
        integral = 0.0
        prev_err = 0.0
        for i in range(1, N):
            t = t_arr[i]
            TL = self.TL_var.get() if t > 2.0 else 0.0
            err = omega_ref - omega
            integral += err * dt
            deriv = (err - prev_err) / dt
            u = Kp * err + Ki * integral + Kd * deriv
            u = max(0.0, min(u, Vdc_max))
            prev_err = err
            ia += ((u - Ra * ia - Kb_SI * omega) / La) * dt
            omega += ((Kb_SI * ia - TL - Bfr * omega) / J) * dt
            omega = max(omega, 0.0)
            ia = max(ia, 0.0)
            pid_speed[i] = omega * 60 / (2 * math.pi)

        # Fuzzy
        fuzzy_speed = np.zeros(N)
        omega = 0.0
        ia = 0.0
        prev_err = 0.0
        u_fuzzy = Vdc_max * 0.5
        for i in range(1, N):
            t = t_arr[i]
            TL = self.TL_var.get() if t > 2.0 else 0.0
            err = omega_ref - omega
            d_err = (err - prev_err) / dt
            delta_u = self._fuzzy_output(err, d_err, e_max, de_max, Vdc_max * 0.3)
            u_fuzzy = max(0.0, min(u_fuzzy + delta_u * dt, Vdc_max))
            prev_err = err
            ia += ((u_fuzzy - Ra * ia - Kb_SI * omega) / La) * dt
            omega += ((Kb_SI * ia - TL - Bfr * omega) / J) * dt
            omega = max(omega, 0.0)
            ia = max(ia, 0.0)
            fuzzy_speed[i] = omega * 60 / (2 * math.pi)

        self.fuzzy_fig.clear()
        ax = self.fuzzy_fig.add_subplot(111)
        ax.set_facecolor("#0d1b2a")
        ax.plot(t_arr, pid_speed, color="#4fc3f7", lw=1.5, label="PID")
        ax.plot(t_arr, fuzzy_speed, color="#f0a500", lw=1.5,
                ls="--", label="Fuzzy")
        ax.axhline(N_ref, color="#90ee90", ls=":", lw=1,
                   label=f"N_ref={N_ref:.0f} rpm")
        ax.axvline(2.0, color="red", ls=":", lw=1, label="Load step")
        ax.set_xlabel("Time (s)", color="#e0eaf5", fontsize=9)
        ax.set_ylabel("Speed (rpm)", color="#e0eaf5", fontsize=9)
        ax.set_title("Speed Response: PID vs Fuzzy Controller",
                     color="#e0eaf5", fontsize=10)
        ax.legend(fontsize=8, facecolor="#1a2636", edgecolor="#4fc3f7",
                  labelcolor="#e0eaf5")
        ax.grid(True, color="#1a2636", lw=0.5)
        ax.tick_params(colors="#e0eaf5", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#2a6496")
        self.fuzzy_fig.patch.set_facecolor("#0d1b2a")
        self.fuzzy_fig.tight_layout(pad=1.5)
        self.fuzzy_canvas.draw()

    # ── Tab 8: Thermal & Economic ─────────────────────────────────────────────
    def _tab_thermal_economic(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="\U0001f321 Thermal & Economic")

        sub_nb = ttk.Notebook(frame)
        sub_nb.pack(fill=tk.BOTH, expand=True)

        thermal_frame = ttk.Frame(sub_nb)
        sub_nb.add(thermal_frame, text="Thermal Model")
        self._build_thermal_tab(thermal_frame)

        econ_frame = ttk.Frame(sub_nb)
        sub_nb.add(econ_frame, text="Economic Analysis")
        self._build_economic_tab(econ_frame)

    def _build_thermal_tab(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        thermal_params = [
            ("Rth (\u00b0C/W)", self.Rth_var, 0.1, 2.0),
            ("Cth (J/\u00b0C)", self.Cth_var, 50, 2000),
            ("Ta (\u00b0C)", self.Ta_var, 0, 60),
        ]
        for label, var, from_, to in thermal_params:
            ttk.Label(ctrl, text=f"  {label}:").pack(side=tk.LEFT)
            lbl = ttk.Label(ctrl, text=f"{var.get():.1f}",
                            foreground="#f0a500", font=("Consolas", 10), width=7)
            lbl.pack(side=tk.LEFT)
            ttk.Scale(ctrl, variable=var, from_=from_, to=to,
                      orient=tk.HORIZONTAL, length=120,
                      command=lambda v, l=lbl, variable=var: (
                          l.config(text=f"{variable.get():.1f}"),
                          self._draw_thermal()
                      )).pack(side=tk.LEFT, padx=2)

        self.thermal_info = ttk.Label(ctrl, text="", foreground="#90ee90",
                                      font=("Consolas", 10))
        self.thermal_info.pack(side=tk.LEFT, padx=10)

        self.thermal_fig = Figure(figsize=(8, 4.5), facecolor="#0d1b2a")
        self.thermal_canvas = FigureCanvasTkAgg(self.thermal_fig, master=parent)
        toolbar = NavigationToolbar2Tk(self.thermal_canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.thermal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_thermal()

    def _draw_thermal(self):
        r = DCMotorCalc.calc_all(
            self.Vs_var.get(), self.freq_var.get(), self.N_ref_var.get(),
            self.Kb_var.get(), self.HP_var.get(), self.Ra_var.get(),
            self.La_var.get(), self.J_var.get()
        )
        Ia = r["Ia_rated"]
        Ra = r["Ra"]
        Pcu = Ia ** 2 * Ra
        Rth = self.Rth_var.get()
        Cth = self.Cth_var.get()
        Ta = self.Ta_var.get()
        tau_th = Rth * Cth
        T_ss = Ta + Pcu * Rth
        t = np.linspace(0, 5 * tau_th, 500)
        T = Ta + Pcu * Rth * (1 - np.exp(-t / tau_th))

        self.thermal_fig.clear()
        ax = self.thermal_fig.add_subplot(111)
        ax.set_facecolor("#0d1b2a")
        ax.plot(t / 60, T, color="#f0a500", lw=2, label="Temperature")
        ax.axhline(T_ss, color="#90ee90", ls="--", lw=1,
                   label=f"T_ss = {T_ss:.1f} \u00b0C")
        ax.axhline(Ta, color="#4fc3f7", ls=":", lw=1,
                   label=f"Ambient = {Ta:.0f} \u00b0C")
        ax.set_xlabel("Time (min)", color="#e0eaf5", fontsize=9)
        ax.set_ylabel("Temperature (\u00b0C)", color="#e0eaf5", fontsize=9)
        ax.set_title(f"Motor Winding Temperature Rise  (Pcu = {Pcu:.1f} W)",
                     color="#e0eaf5", fontsize=10)
        ax.legend(fontsize=8, facecolor="#1a2636", edgecolor="#4fc3f7",
                  labelcolor="#e0eaf5")
        ax.grid(True, color="#1a2636", lw=0.5)
        ax.tick_params(colors="#e0eaf5", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#2a6496")
        self.thermal_fig.patch.set_facecolor("#0d1b2a")
        self.thermal_fig.tight_layout(pad=1.5)
        self.thermal_canvas.draw()
        self.thermal_info.config(
            text=f"Pcu={Pcu:.1f}W  T_ss={T_ss:.1f}\u00b0C  \u03c4_th={tau_th/60:.1f}min")

    def _build_economic_tab(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        econ_params = [
            ("Tariff ($/kWh)", self.tariff_var, 0.05, 0.5),
            ("Hours/day", self.hours_day_var, 1, 24),
            ("Days/year", self.days_year_var, 1, 365),
        ]
        for label, var, from_, to in econ_params:
            ttk.Label(ctrl, text=f"  {label}:").pack(side=tk.LEFT)
            lbl = ttk.Label(ctrl, text=f"{var.get():.2f}",
                            foreground="#f0a500", font=("Consolas", 10), width=7)
            lbl.pack(side=tk.LEFT)
            ttk.Scale(ctrl, variable=var, from_=from_, to=to,
                      orient=tk.HORIZONTAL, length=120,
                      command=lambda v, l=lbl, variable=var: (
                          l.config(text=f"{variable.get():.2f}"),
                          self._draw_economic()
                      )).pack(side=tk.LEFT, padx=2)

        self.econ_fig = Figure(figsize=(8, 4.5), facecolor="#0d1b2a")
        self.econ_canvas = FigureCanvasTkAgg(self.econ_fig, master=parent)
        toolbar = NavigationToolbar2Tk(self.econ_canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.econ_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_economic()

    def _draw_economic(self):
        tariff = self.tariff_var.get()
        hours = self.hours_day_var.get()
        days = self.days_year_var.get()

        speeds = [500, 750, 1000, 1250, 1500, 1750, 2000]
        annual_costs = []
        labels = []
        for N in speeds:
            r = DCMotorCalc.calc_all(
                self.Vs_var.get(), self.freq_var.get(), float(N),
                self.Kb_var.get(), self.HP_var.get(), self.Ra_var.get(),
                self.La_var.get(), self.J_var.get()
            )
            P_kW = r["P_in"] / 1000.0
            annual_kWh = P_kW * hours * days
            annual_cost = annual_kWh * tariff
            annual_costs.append(annual_cost)
            labels.append(f"{N} rpm")

        self.econ_fig.clear()
        ax = self.econ_fig.add_subplot(111)
        ax.set_facecolor("#0d1b2a")
        bars = ax.bar(labels, annual_costs, color="#2a6496", edgecolor="#4fc3f7",
                      linewidth=0.8)
        for bar, cost in zip(bars, annual_costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"${cost:.0f}", ha="center", va="bottom",
                    color="#90ee90", fontsize=8)
        ax.set_xlabel("Operating Speed", color="#e0eaf5", fontsize=9)
        ax.set_ylabel("Annual Energy Cost ($)", color="#e0eaf5", fontsize=9)
        ax.set_title(
            f"Annual Energy Cost vs Speed  (${tariff:.2f}/kWh, "
            f"{hours:.0f}h/day, {days:.0f}days/yr)",
            color="#e0eaf5", fontsize=10)
        ax.tick_params(colors="#e0eaf5", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#2a6496")
        self.econ_fig.patch.set_facecolor("#0d1b2a")
        self.econ_fig.tight_layout(pad=1.5)
        self.econ_canvas.draw()

    # ── Tab 9: Harmonics & Power Quality ─────────────────────────────────────
    def _tab_harmonics(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="\U0001f4ca Harmonics & PQ")

        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        ttk.Button(ctrl, text="Compute & Plot Harmonics",
                   command=self._draw_harmonics).pack(side=tk.LEFT, padx=8)

        self.harm_info = ttk.Label(ctrl, text="", foreground="#90ee90",
                                   font=("Consolas", 10))
        self.harm_info.pack(side=tk.LEFT, padx=8)

        self.harm_fig = Figure(figsize=(8, 5), facecolor="#0d1b2a")
        self.harm_canvas = FigureCanvasTkAgg(self.harm_fig, master=frame)
        toolbar = NavigationToolbar2Tk(self.harm_canvas, frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.harm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_harmonics()

    def _draw_harmonics(self):
        r = DCMotorCalc.calc_all(
            self.Vs_var.get(), self.freq_var.get(), self.N_ref_var.get(),
            self.Kb_var.get(), self.HP_var.get(), self.Ra_var.get(),
            self.La_var.get(), self.J_var.get()
        )
        Ia = r["Ia_rated"]
        alpha_r = r["alpha_rad"]

        N_harm = 19
        harmonic_orders = list(range(1, N_harm + 1, 2))
        amplitudes = []
        for n in harmonic_orders:
            I_n = (4 * Ia / (n * math.pi))
            amplitudes.append(I_n)

        I1 = amplitudes[0]
        I_rms_total = math.sqrt(sum(a ** 2 for a in amplitudes) / 2)
        THD_calc = math.sqrt(sum(a ** 2 for a in amplitudes[1:]) / 2) / (I1 / math.sqrt(2)) * 100
        disp_PF = math.cos(alpha_r)
        dist_factor = I1 / math.sqrt(2) / (I_rms_total + _EPSILON)
        apparent_PF = disp_PF * dist_factor

        # IEC 61000-3-2 Class A limits (A rms) for odd harmonics
        iec_limits = {
            1: 9999, 3: 2.30, 5: 1.14, 7: 0.77, 9: 0.40,
            11: 0.33, 13: 0.21, 15: 0.15, 17: 0.132, 19: 0.118
        }

        self.harm_fig.clear()
        ax1, ax2 = self.harm_fig.subplots(1, 2)

        ax1.set_facecolor("#0d1b2a")
        x = np.arange(len(harmonic_orders))
        ax1.bar(x, [a / math.sqrt(2) for a in amplitudes],
                color="#2a6496", edgecolor="#4fc3f7", lw=0.8,
                label="Harmonic RMS (A)")
        iec_vals = [iec_limits.get(n, None) for n in harmonic_orders]
        iec_x = [x[i] for i, v in enumerate(iec_vals) if v is not None and v < 99]
        iec_y = [v for v in iec_vals if v is not None and v < 99]
        if iec_x:
            ax1.scatter(iec_x, iec_y, marker="_", color="red", s=200, linewidths=2,
                        zorder=5, label="IEC 61000-3-2 Limit")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(n) for n in harmonic_orders],
                            color="#e0eaf5", fontsize=8)
        ax1.set_xlabel("Harmonic Order", color="#e0eaf5", fontsize=9)
        ax1.set_ylabel("Current (A rms)", color="#e0eaf5", fontsize=9)
        ax1.set_title("Harmonic Spectrum of Supply Current",
                      color="#e0eaf5", fontsize=9)
        ax1.legend(fontsize=7, facecolor="#1a2636", edgecolor="#4fc3f7",
                   labelcolor="#e0eaf5")
        ax1.tick_params(colors="#e0eaf5", labelsize=7)
        ax1.grid(True, axis="y", color="#1a2636", lw=0.5)
        for sp in ax1.spines.values():
            sp.set_color("#2a6496")

        ax2.set_facecolor("#0d1b2a")
        ax2.axis("off")
        summary = [
            "Power Quality Summary",
            "\u2500" * 30,
            f"Fundamental (I1)  : {I1/math.sqrt(2):.3f} A",
            f"I_rms (total)     : {I_rms_total:.3f} A",
            f"THD               : {THD_calc:.2f} %",
            f"Displacement PF   : {disp_PF:.4f}",
            f"Distortion Factor : {dist_factor:.4f}",
            f"Apparent PF       : {apparent_PF:.4f}",
            "\u2500" * 30,
            f"Firing angle \u03b1  : {r['alpha_deg']:.2f}\u00b0",
            f"Vdc               : {r['Vdc_req']:.2f} V",
            f"Ia_rated          : {r['Ia_rated']:.2f} A",
            "\u2500" * 30,
            "IEC 61000-3-2",
            "Red markers show",
            "Class A limits.",
        ]
        for i, line in enumerate(summary):
            color = "#7ec8e3" if i in (1, 8, 12) else "#e0eaf5"
            if i == 0:
                color = "#f0a500"
            ax2.text(0.05, 0.97 - i * 0.063, line, transform=ax2.transAxes,
                     fontsize=9, color=color, fontfamily="Consolas",
                     verticalalignment="top")

        self.harm_fig.patch.set_facecolor("#0d1b2a")
        self.harm_fig.tight_layout(pad=1.5)
        self.harm_canvas.draw()
        self.harm_info.config(
            text=f"THD={THD_calc:.2f}%  dispPF={disp_PF:.4f}  appPF={apparent_PF:.4f}")

    # ── Resize & Status ───────────────────────────────────────────────────────
    def _on_resize(self, event):
        pass  # Tkinter handles geometry; figures resize via expand=True

    def _update_status(self):
        try:
            r = DCMotorCalc.calc_all(
                self.Vs_var.get(), self.freq_var.get(), self.N_ref_var.get(),
                self.Kb_var.get(), self.HP_var.get(), self.Ra_var.get(),
                self.La_var.get(), self.J_var.get()
            )
            self.status_var.set(
                f"  \u2022 Vs={r['Vs']:.0f}V  f={r['freq']:.0f}Hz  "
                f"N={r['N_ref']:.0f}rpm  Vdc={r['Vdc_req']:.2f}V  "
                f"\u03b1={r['alpha_deg']:.2f}\u00b0  Ia={r['Ia_rated']:.2f}A  "
                f"PF={r['PF']:.4f}  THD={r['THD']:.2f}%"
            )
        except (ValueError, ArithmeticError):
            self.status_var.set("  Error updating status")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = DCMotorApp()
    app.mainloop()
