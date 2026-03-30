"""
3-Phase Induction Motor - Comprehensive Electrical Engineering Analysis
=======================================================================
Problem (reference defaults):
  6-pole, 50 Hz, 3-phase induction motor at full load develops
  useful torque = 160 Nm; rotor EMF makes 120 cycles/min;
  friction + core-loss torque = 10 Nm; stator total loss = 800 W.

Tabs:
  1. Overview        - step-by-step solution with explanation
  2. Model & Sim     - equivalent circuit, torque-slip curve, power flow
  3. Fault Current   - symmetrical components, fault levels, decay curve
  4. Protection      - IEC inverse-time OCR coordination curves
  5. Speed Control   - PID and Fuzzy step-load response comparison
  6. Thermal & Econ  - 1st-order thermal model, economic analysis
  7. Harmonics       - THD, power quality, synthesised waveform
  8. Validation      - power-balance, sanity checks, efficiency vs slip
"""
import tkinter as tk
from tkinter import ttk
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PI = math.pi


# =============================================================================
# Mathematical core functions
# =============================================================================

def solve_motor(poles, freq, T_useful, T_fric, rotor_cpm,
                stator_loss, v_line, pf):
    poles = max(2, int(round(poles / 2)) * 2)
    Ns = 120.0 * freq / poles
    f_rotor = rotor_cpm / 60.0
    s = max(1e-9, min(f_rotor / freq, 0.9999))
    Nr = Ns * (1.0 - s)
    omega_r = 2.0 * PI * Nr / 60.0
    omega_s = 2.0 * PI * Ns / 60.0
    T_total = T_useful + T_fric
    P_airgap = T_total * omega_s
    P_rotor_cu = s * P_airgap
    P_mech = (1.0 - s) * P_airgap
    P_fric_w = T_fric * omega_r
    P_shaft = T_useful * omega_r
    P_input = P_airgap + stator_loss
    eff = P_shaft / P_input * 100.0 if P_input > 0 else 0.0
    denom = math.sqrt(3.0) * v_line * pf
    I_line = P_input / denom if denom > 0 else 0.0
    return dict(
        poles=poles, freq=freq, Ns=Ns, s=s, Nr=Nr,
        omega_r=omega_r, omega_s=omega_s,
        T_total=T_total, T_useful=T_useful, T_fric=T_fric,
        P_airgap=P_airgap, P_rotor_cu=P_rotor_cu,
        P_mech=P_mech, P_fric_w=P_fric_w, P_shaft=P_shaft,
        P_input=P_input, stator_loss=stator_loss,
        eff=eff, I_line=I_line, f_rotor=f_rotor,
        v_line=v_line, pf=pf,
    )


def slip_torque_curve(poles, freq, T_total, s_op, n_pts=300):
    s_max = max(0.08, s_op * 0.9)
    T_max = T_total * (s_op / s_max + s_max / s_op) / 2.0
    s_arr = np.linspace(0.001, 1.0, n_pts)
    T_arr = T_max * 2.0 / (s_arr / s_max + s_max / s_arr)
    return s_arr, T_arr


def pid_sim(Kp, Ki, Kd, omega_ref, T_step,
            J=0.5, B=0.1, t_end=5.0, dt=0.005):
    t = np.arange(0.0, t_end, dt)
    w = np.zeros(len(t))
    w[0] = omega_ref
    integral = 0.0
    prev_e = 0.0
    T_load = 0.0
    for i in range(1, len(t)):
        if t[i] >= 1.0:
            T_load = T_step
        e = omega_ref - w[i - 1]
        integral += e * dt
        deriv = (e - prev_e) / dt
        T_ctrl = float(np.clip(Kp * e + Ki * integral + Kd * deriv,
                               -500.0, 500.0))
        w[i] = max(0.0,
                   w[i - 1] + (T_ctrl - T_load - B * w[i - 1]) / J * dt)
        prev_e = e
    return t, w


def fuzzy_sim(omega_ref, T_step, J=0.5, B=0.1, t_end=5.0, dt=0.005):
    t = np.arange(0.0, t_end, dt)
    w = np.zeros(len(t))
    w[0] = omega_ref
    T_load = 0.0
    prev_e = 0.0

    def fuzz(e, de):
        en = float(np.clip(e / 50.0, -1.0, 1.0))
        dn = float(np.clip(de / 200.0, -1.0, 1.0))
        return float(np.clip(en * 150.0 + dn * 30.0, -300.0, 300.0))

    for i in range(1, len(t)):
        if t[i] >= 1.0:
            T_load = T_step
        e = omega_ref - w[i - 1]
        de = (e - prev_e) / dt
        T_ctrl = fuzz(e, de)
        w[i] = max(0.0,
                   w[i - 1] + (T_ctrl - T_load - B * w[i - 1]) / J * dt)
        prev_e = e
    return t, w


def protection_curves(I_pickup, TMS=1.0, n_pts=300):
    I_arr = np.linspace(I_pickup * 1.05, I_pickup * 20.0, n_pts)
    ratio = I_arr / I_pickup
    t_EI = np.clip(TMS * 80.0 / (ratio ** 2 - 1.0), 0.01, 1000.0)
    t_NI = np.clip(TMS * 0.14 / (ratio ** 0.02 - 1.0), 0.01, 1000.0)
    return I_arr, t_EI, t_NI


def harmonic_spectrum(I_fund):
    ords = np.array([1, 3, 5, 7, 11, 13])
    pct = np.array([100.0, 2.5, 18.0, 9.0, 4.0, 2.5])
    I_h = I_fund * pct / 100.0
    thd_calc = math.sqrt(float(np.sum(I_h[1:] ** 2))) / I_h[0] * 100.0
    I_rms = math.sqrt(float(np.sum(I_h ** 2)))
    return ords, pct, I_h, I_rms, thd_calc


def thermal_model(P_loss, R_th=0.02, C_th=10000.0,
                  t_end=3600, dt=10):
    t = np.arange(0.0, t_end + dt, dt)
    T = np.zeros(len(t))
    for i in range(1, len(t)):
        dT = (P_loss - T[i - 1] / R_th) / C_th * dt
        T[i] = T[i - 1] + dT
    return t / 60.0, T + 25.0


def fault_currents(V_line, Z1, Z2, Z0):
    V_ph = V_line / math.sqrt(3.0)
    I_3ph = V_ph / abs(Z1) if abs(Z1) > 0 else 0.0
    dZ_slg = abs(Z1 + Z2 + Z0)
    I_slg = 3.0 * V_ph / dZ_slg if dZ_slg > 0 else 0.0
    dZ_ll = abs(Z1 + Z2)
    I_ll = math.sqrt(3.0) * V_ph / dZ_ll if dZ_ll > 0 else 0.0
    return I_3ph, I_slg, I_ll


# =============================================================================
# GUI Application
# =============================================================================

class MotorApp:
    def __init__(self, root):
        self.root = root
        root.title('3-Phase Induction Motor - Advanced EE Analysis')
        root.minsize(1100, 760)
        try:
            root.state('zoomed')
        except Exception:
            root.geometry('1280x860')
        self._build_vars()
        self._build_ui()
        self.update()

    # -------------------------------------------------------------------------
    # Variables
    # -------------------------------------------------------------------------
    def _build_vars(self):
        self.v_poles     = tk.IntVar(value=6)
        self.v_freq      = tk.DoubleVar(value=50.0)
        self.v_T_useful  = tk.DoubleVar(value=160.0)
        self.v_T_fric    = tk.DoubleVar(value=10.0)
        self.v_rotor_cpm = tk.DoubleVar(value=120.0)
        self.v_stator_W  = tk.DoubleVar(value=800.0)
        self.v_vline     = tk.DoubleVar(value=400.0)
        self.v_pf        = tk.DoubleVar(value=0.87)
        self.v_Kp        = tk.DoubleVar(value=5.0)
        self.v_Ki        = tk.DoubleVar(value=2.0)
        self.v_Kd        = tk.DoubleVar(value=0.5)
        self.v_T_step    = tk.DoubleVar(value=50.0)
        self.v_tariff    = tk.DoubleVar(value=0.12)
        self.v_hours     = tk.DoubleVar(value=8000.0)
        self.v_ctrl      = tk.StringVar(value='PID')
        self.v_status    = tk.StringVar(value='Ready')

    # -------------------------------------------------------------------------
    # UI layout
    # -------------------------------------------------------------------------
    def _build_ui(self):
        paned = ttk.PanedWindow(self.root, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=6, pady=6)

        # --- LEFT: inputs ---
        lframe = ttk.LabelFrame(paned,
                                text='Input Parameters & Control',
                                width=350)
        paned.add(lframe, weight=0)
        lframe.grid_columnconfigure(1, weight=1)

        sliders = [
            ('Poles',                  self.v_poles,      2,    12),
            ('Frequency (Hz)',         self.v_freq,       40,   60),
            ('Useful torque (Nm)',     self.v_T_useful,   50,  400),
            ('Fric+core torque (Nm)',  self.v_T_fric,      1,   50),
            ('Rotor cycles/min',       self.v_rotor_cpm,  10,  300),
            ('Stator loss (W)',        self.v_stator_W,  200, 3000),
            ('Line voltage (V)',       self.v_vline,     200,  660),
            ('Power factor',          self.v_pf,         0.5,  1.0),
            ('Kp (PID)',              self.v_Kp,         0.1,   20),
            ('Ki (PID)',              self.v_Ki,         0.0,   10),
            ('Kd (PID)',              self.v_Kd,         0.0,    5),
            ('Load step torque (Nm)', self.v_T_step,       5,  200),
            ('Tariff ($/kWh)',        self.v_tariff,    0.05,  0.5),
            ('Operating hours/year',  self.v_hours,     1000, 8760),
        ]
        for row, (lab, var, frm, to) in enumerate(sliders):
            ttk.Label(lframe, text=lab).grid(
                row=row, column=0, sticky='w', padx=5, pady=2)
            sc = ttk.Scale(lframe, variable=var, from_=frm, to=to,
                           orient='horizontal',
                           command=lambda _e: self.update())
            sc.grid(row=row, column=1, sticky='ew', padx=4, pady=2)
            ent = ttk.Entry(lframe, textvariable=var, width=9)
            ent.grid(row=row, column=2, padx=4, pady=2)
            ent.bind('<Return>', lambda _e: self.update())

        r = len(sliders)
        ttk.Label(lframe, text='Controller').grid(
            row=r, column=0, sticky='w', padx=5, pady=3)
        ttk.Combobox(
            lframe, textvariable=self.v_ctrl,
            values=['Open Loop', 'PID', 'Fuzzy'],
            state='readonly',
        ).grid(row=r, column=1, columnspan=2, sticky='ew', padx=4, pady=3)

        r += 1
        btns = ttk.Frame(lframe)
        btns.grid(row=r, column=0, columnspan=3, sticky='ew', padx=5, pady=6)
        for txt, cmd in [('Start', self._start),
                         ('Stop', self._stop),
                         ('Reset', self._reset)]:
            ttk.Button(btns, text=txt, command=cmd).pack(
                side='left', fill='x', expand=True, padx=2)

        r += 1
        ttk.Label(lframe, textvariable=self.v_status,
                  foreground='navy').grid(
            row=r, column=0, columnspan=3, sticky='w', padx=5, pady=4)

        # --- RIGHT: notebook ---
        rframe = ttk.Frame(paned)
        paned.add(rframe, weight=1)
        self._summary_lbl = ttk.Label(
            rframe, text='',
            font=('Segoe UI', 11, 'bold'), foreground='#003366')
        self._summary_lbl.pack(fill='x', padx=8, pady=(4, 0))

        self.nb = ttk.Notebook(rframe)
        self.nb.pack(fill='both', expand=True, pady=4)

        self._tabs = {}
        for name in ['Overview', 'Model & Sim', 'Fault Current',
                     'Protection', 'Speed Control', 'Thermal & Econ',
                     'Harmonics', 'Validation']:
            f = ttk.Frame(self.nb)
            self.nb.add(f, text=name)
            self._tabs[name] = f

        self._build_overview_tab()
        self._build_model_tab()
        self._build_fault_tab()
        self._build_protection_tab()
        self._build_speed_tab()
        self._build_thermal_tab()
        self._build_harmonics_tab()
        self._build_validation_tab()

    # -------------------------------------------------------------------------
    # Helper builders
    # -------------------------------------------------------------------------
    @staticmethod
    def _make_text(parent):
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True)
        sb = ttk.Scrollbar(frame)
        sb.pack(side='right', fill='y')
        t = tk.Text(frame, wrap='word', yscrollcommand=sb.set,
                    font=('Consolas', 10), bg='#fafafa', relief='flat')
        t.pack(fill='both', expand=True)
        sb.config(command=t.yview)
        return t

    @staticmethod
    def _make_fig(parent, rows=1, cols=1, h=4.0):
        fig = Figure(figsize=(9, h), dpi=90, tight_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        axes = [fig.add_subplot(rows, cols, i + 1)
                for i in range(rows * cols)]
        return fig, canvas, axes

    @staticmethod
    def _make_split(parent):
        pw = ttk.PanedWindow(parent, orient='vertical')
        pw.pack(fill='both', expand=True)
        top = ttk.Frame(pw)
        bot = ttk.Frame(pw)
        pw.add(top, weight=1)
        pw.add(bot, weight=2)
        return top, bot

    # -------------------------------------------------------------------------
    # Tab builders
    # -------------------------------------------------------------------------
    def _build_overview_tab(self):
        self._ov_text = self._make_text(self._tabs['Overview'])

    def _build_model_tab(self):
        top, bot = self._make_split(self._tabs['Model & Sim'])
        self._model_text = self._make_text(top)
        self._model_fig, self._model_canvas, self._model_axes = \
            self._make_fig(bot, 1, 2, 3.8)

    def _build_fault_tab(self):
        top, bot = self._make_split(self._tabs['Fault Current'])
        self._fault_text = self._make_text(top)
        self._fault_fig, self._fault_canvas, self._fault_axes = \
            self._make_fig(bot, 1, 2, 3.8)

    def _build_protection_tab(self):
        top, bot = self._make_split(self._tabs['Protection'])
        self._prot_text = self._make_text(top)
        self._prot_fig, self._prot_canvas, self._prot_axes = \
            self._make_fig(bot, 1, 1, 3.8)

    def _build_speed_tab(self):
        top, bot = self._make_split(self._tabs['Speed Control'])
        self._speed_text = self._make_text(top)
        self._speed_fig, self._speed_canvas, self._speed_axes = \
            self._make_fig(bot, 1, 1, 4.2)

    def _build_thermal_tab(self):
        top, bot = self._make_split(self._tabs['Thermal & Econ'])
        self._therm_text = self._make_text(top)
        self._therm_fig, self._therm_canvas, self._therm_axes = \
            self._make_fig(bot, 1, 2, 3.8)

    def _build_harmonics_tab(self):
        top, bot = self._make_split(self._tabs['Harmonics'])
        self._harm_text = self._make_text(top)
        self._harm_fig, self._harm_canvas, self._harm_axes = \
            self._make_fig(bot, 1, 2, 3.8)

    def _build_validation_tab(self):
        top, bot = self._make_split(self._tabs['Validation'])
        self._val_text = self._make_text(top)
        self._val_fig, self._val_canvas, self._val_axes = \
            self._make_fig(bot, 1, 2, 3.8)

    # -------------------------------------------------------------------------
    # Central update dispatcher
    # -------------------------------------------------------------------------
    def update(self, *_):
        r = solve_motor(
            self.v_poles.get(), self.v_freq.get(),
            self.v_T_useful.get(), self.v_T_fric.get(),
            self.v_rotor_cpm.get(), self.v_stator_W.get(),
            self.v_vline.get(), self.v_pf.get())
        self._r = r
        self._summary_lbl.config(
            text=('Ns=%s rpm  |  Nr=%s rpm  |  s=%s%%  |  '
                  'P_shaft=%s W  |  eta=%s%%  |  I_line=%s A' % (
                      '%.1f' % r['Ns'],
                      '%.1f' % r['Nr'],
                      '%.3f' % (r['s'] * 100),
                      '%.1f' % r['P_shaft'],
                      '%.2f' % r['eff'],
                      '%.2f' % r['I_line'],
                  )))
        self._upd_overview(r)
        self._upd_model(r)
        self._upd_fault(r)
        self._upd_protection(r)
        self._upd_speed(r)
        self._upd_thermal(r)
        self._upd_harmonics(r)
        self._upd_validation(r)

    # -------------------------------------------------------------------------
    # Tab update methods
    # -------------------------------------------------------------------------
    @staticmethod
    def _set_text(widget, lines):
        widget.delete('1.0', 'end')
        widget.insert('1.0', '\n'.join(lines))

    def _upd_overview(self, r):
        rcpm = self.v_rotor_cpm.get()
        lines = [
            '=' * 70,
            '  3-PHASE INDUCTION MOTOR - COMPLETE STEP-BY-STEP SOLUTION',
            '=' * 70,
            '',
            'GIVEN DATA:',
            '  Number of poles P           = %d' % r['poles'],
            '  Supply frequency f          = %.1f Hz' % r['freq'],
            '  Useful (shaft) torque T_u   = %.2f Nm' % r['T_useful'],
            '  Friction + core-loss torque = %.2f Nm' % r['T_fric'],
            '  Rotor EMF frequency         = %.4f Hz  (%.1f cycles/min)' % (
                r['f_rotor'], rcpm),
            '  Stator total loss           = %.1f W' % r['stator_loss'],
            '',
            '-' * 70,
            'STEP 1 - Synchronous Speed',
            '  Ns = 120 x f / P = 120 x %.1f / %d' % (r['freq'], r['poles']),
            '     = %.4f rpm' % r['Ns'],
            '  omega_s = 2*pi*Ns/60 = %.6f rad/s' % r['omega_s'],
            '',
            'STEP 2 - Slip',
            '  Rotor EMF frequency = %.1f cpm = %.4f Hz' % (rcpm, r['f_rotor']),
            '  s = f_rotor / f_supply = %.4f / %.1f' % (r['f_rotor'], r['freq']),
            '    = %.8f  (%.6f %%)' % (r['s'], r['s'] * 100),
            '  Nr = Ns(1-s) = %.4f x %.8f = %.6f rpm' % (
                r['Ns'], 1 - r['s'], r['Nr']),
            '  omega_r = 2*pi*Nr/60 = %.6f rad/s' % r['omega_r'],
            '',
            'STEP 3 - Shaft Power Output',
            '  P_shaft = T_useful x omega_r',
            '          = %.2f x %.6f' % (r['T_useful'], r['omega_r']),
            '          = %.6f W' % r['P_shaft'],
            '          = %.4f kW' % (r['P_shaft'] / 1000),
            '',
            'STEP 4 - Air-gap Power (Rotor Input)',
            '  T_total = T_useful + T_fric = %.2f + %.2f = %.2f Nm' % (
                r['T_useful'], r['T_fric'], r['T_total']),
            '  P_airgap = T_total x omega_s = %.2f x %.6f' % (
                r['T_total'], r['omega_s']),
            '           = %.6f W' % r['P_airgap'],
            '  [Check via P_mech/(1-s): %.4f/%.6f = %.4f W]' % (
                r['P_mech'], 1 - r['s'], r['P_mech'] / (1 - r['s'])),
            '',
            '-' * 70,
            'ANSWERS',
            '',
            '(a)  ROTOR COPPER LOSS:',
            '     P_rotor_cu = s x P_airgap',
            '               = %.8f x %.6f' % (r['s'], r['P_airgap']),
            '               = %.6f W' % r['P_rotor_cu'],
            '               ~ %.2f W' % r['P_rotor_cu'],
            '',
            '(b)  INPUT TO THE MOTOR:',
            '     P_input = P_airgap + Stator_loss',
            '             = %.6f + %.1f' % (r['P_airgap'], r['stator_loss']),
            '             = %.6f W' % r['P_input'],
            '             ~ %.2f W = %.4f kW' % (
                r['P_input'], r['P_input'] / 1000),
            '',
            '(c)  EFFICIENCY:',
            '     eta = P_shaft / P_input x 100',
            '         = %.4f / %.4f x 100' % (r['P_shaft'], r['P_input']),
            '         = %.6f %%' % r['eff'],
            '         ~ %.2f %%' % r['eff'],
            '',
            '-' * 70,
            'COMPLETE POWER FLOW:',
            '  P_input          = %.2f W' % r['P_input'],
            '  - Stator loss    = %.2f W' % r['stator_loss'],
            '  P_airgap         = %.2f W' % r['P_airgap'],
            '  - Rotor Cu loss  = %.2f W' % r['P_rotor_cu'],
            '  P_mechanical     = %.2f W' % r['P_mech'],
            '  - Fric+core loss = %.2f W' % r['P_fric_w'],
            '  P_shaft (output) = %.2f W' % r['P_shaft'],
            '  Line current I   = %.4f A  (at PF=%.3f)' % (
                r['I_line'], r['pf']),
            '=' * 70,
            '',
            'EXPLANATION OF KEY CONCEPTS:',
            '',
            'Slip (s): fractional difference between synchronous and rotor',
            '  speed. At s=0 the rotor runs at synchronous speed (no torque);',
            '  at s=1 rotor is stationary. Full-load slip is typically 2-8%.',
            '',
            'Air-gap power: electromagnetic power crossing the air gap.',
            '  P_airgap = T_em x omega_s regardless of slip.',
            '  It splits: (1-s) fraction -> mechanical, s fraction -> heat.',
            '',
            'Rotor copper loss: I^2 R loss in rotor windings.',
            '  The 3-way identity:  P_rotor_cu : P_mech : P_airgap = s:(1-s):1',
            '',
            'Efficiency: only the useful shaft output counts, not P_mech,',
            '  because frictional and core losses are also subtracted.',
        ]
        self._set_text(self._ov_text, lines)

    def _upd_model(self, r):
        lines = [
            'MATHEMATICAL MODEL - 3-PHASE INDUCTION MOTOR',
            '=' * 58,
            '',
            '1. SYNCHRONOUS SPEED',
            '   Ns [rpm] = 120 * f / P',
            '',
            '2. SLIP',
            '   s = (Ns - Nr) / Ns  =  f_rotor / f_supply',
            '',
            '3. PER-PHASE EQUIVALENT CIRCUIT (IEEE T-circuit):',
            '   V1 = I1*(R1 + jX1) + E1',
            '   I2 = E2 / (R2/s + jX2)  (rotor referred to stator)',
            '   Magnetising branch: E1 = I_m * (R_m || jX_m)',
            '',
            '4. POWER RELATIONS',
            '   P_airgap  = 3*|I2|^2 * R2/s  = Te * omega_s',
            '   P_rotor_cu= 3*|I2|^2 * R2    = s  * P_airgap',
            '   P_mech    = (1-s) * P_airgap  = Te * omega_r',
            '',
            '5. ELECTROMAGNETIC TORQUE',
            '   Te = P_airgap / omega_s',
            '      = 3*V1^2*(R2/s) / [omega_s*((R1+R2/s)^2+(X1+X2)^2)]',
            '',
            '6. DYNAMIC MODEL (mechanical):',
            '   J * d(omega_r)/dt = Te - TL - B * omega_r',
            '   (J = inertia, B = viscous damping, TL = load torque)',
            '',
            '7. TORQUE-SLIP (Kloss approximation):',
            '   T(s) = 2*Tmax / (s/sm + sm/s)',
            '   sm = slip at max torque ~ R2/X2',
            '',
            'CURRENT OPERATING POINT:',
            '  omega_r = %.4f rad/s  (%.2f rpm)' % (r['omega_r'], r['Nr']),
            '  omega_s = %.4f rad/s  (%.2f rpm)' % (r['omega_s'], r['Ns']),
            '  s       = %.6f  (%.4f %%)' % (r['s'], r['s'] * 100),
            '  T_em    = %.2f Nm' % r['T_total'],
        ]
        self._set_text(self._model_text, lines)

        s_arr, T_arr = slip_torque_curve(
            r['poles'], r['freq'], r['T_total'], r['s'])
        ax1, ax2 = self._model_axes
        ax1.clear()
        ax1.plot(s_arr * 100.0, T_arr, 'b-', linewidth=2)
        ax1.axvline(r['s'] * 100, color='r', linestyle='--', linewidth=1.5,
                    label='Op. s=%.2f%%' % (r['s'] * 100))
        ax1.axhline(r['T_total'], color='g', linestyle='--', linewidth=1.5,
                    label='T=%.1f Nm' % r['T_total'])
        ax1.scatter([r['s'] * 100], [r['T_total']],
                    color='red', s=80, zorder=5)
        ax1.set_xlabel('Slip (%)')
        ax1.set_ylabel('Torque (Nm)')
        ax1.set_title('Torque-Slip Characteristic')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)

        labels = ['P_input', 'Stator\nLoss', 'P_airgap',
                  'Rotor Cu', 'P_mech', 'Fric+\nCore', 'P_shaft']
        vals = [r['P_input'], r['stator_loss'], r['P_airgap'],
                r['P_rotor_cu'], r['P_mech'], r['P_fric_w'], r['P_shaft']]
        colors = ['#2196F3', '#F44336', '#4CAF50',
                  '#FF9800', '#9C27B0', '#795548', '#00BCD4']
        ax2.clear()
        ax2.bar(labels, vals, color=colors, edgecolor='white')
        ax2.set_ylabel('Power (W)')
        ax2.set_title('Power Flow')
        ax2.tick_params(axis='x', labelsize=8)
        ax2.grid(True, axis='y', alpha=0.3)
        for i, v in enumerate(vals):
            ax2.text(i, v + max(vals) * 0.012, '%.0f' % v,
                     ha='center', fontsize=7)
        self._model_canvas.draw()

    def _upd_fault(self, r):
        V_ph = r['v_line'] / math.sqrt(3.0)
        I_rated = r['I_line']
        Z1 = complex(0.5, 1.2)
        Z2 = complex(0.4, 0.8)
        Z0 = complex(1.0, 3.0)
        I_3ph, I_slg, I_ll = fault_currents(r['v_line'], Z1, Z2, Z0)
        Z_sub = V_ph / (6.0 * I_rated) if I_rated > 0 else 1.0
        I_motor_fault = V_ph / Z_sub if Z_sub > 0 else 0.0
        lines = [
            'FAULT CURRENT ANALYSIS (Symmetrical Components)',
            '=' * 58,
            '',
            'SYSTEM DATA:',
            '  Line voltage   = %.1f V' % r['v_line'],
            '  Phase voltage  = %.2f V' % V_ph,
            '  Rated current  = %.4f A' % I_rated,
            '',
            'SEQUENCE IMPEDANCES (estimated per phase):',
            '  Z1 (positive) = %s  |Z1|=%.4f ohm' % (Z1, abs(Z1)),
            '  Z2 (negative) = %s  |Z2|=%.4f ohm' % (Z2, abs(Z2)),
            '  Z0 (zero)     = %s  |Z0|=%.4f ohm' % (Z0, abs(Z0)),
            '',
            '3-PHASE SYMMETRICAL FAULT:',
            '  I_3ph = V_ph/|Z1| = %.2f/%.4f = %.2f A' % (
                V_ph, abs(Z1), I_3ph),
            '  Peak (with DC offset) = %.2f A' % (I_3ph * 1.414 * 1.5),
            '',
            'SINGLE LINE-TO-GROUND (SLG) FAULT:',
            '  I_SLG = 3*V_ph/|Z1+Z2+Z0| = %.2f A' % I_slg,
            '',
            'LINE-TO-LINE (LL) FAULT:',
            '  I_LL = sqrt(3)*V_ph/|Z1+Z2| = %.2f A' % I_ll,
            '',
            'MOTOR CONTRIBUTION (sub-transient):',
            '  ~ 6x rated impedance: I_fault ~ %.2f A' % I_motor_fault,
            '  Decays to zero in ~100-200 ms',
            '',
            'FAULT SEVERITY RANKING:',
            '  3-phase: %.2f A (%.1fx I_rated)' % (I_3ph, I_3ph / max(I_rated, 0.001)),
            '  SLG:     %.2f A (%.1fx I_rated)' % (I_slg, I_slg / max(I_rated, 0.001)),
            '  LL:      %.2f A (%.1fx I_rated)' % (I_ll, I_ll / max(I_rated, 0.001)),
        ]
        self._set_text(self._fault_text, lines)

        ax1, ax2 = self._fault_axes
        ax1.clear()
        ftypes = ['3-Phase\n(sym)', 'SLG', 'Line-Line']
        fvals = [I_3ph, I_slg, I_ll]
        ax1.bar(ftypes, fvals,
                color=['#F44336', '#FF9800', '#2196F3'], edgecolor='white')
        ax1.set_ylabel('Fault Current (A)')
        ax1.set_title('Fault Current Comparison')
        ax1.grid(True, axis='y', alpha=0.3)
        for i, v in enumerate(fvals):
            ax1.text(i, v + max(fvals) * 0.02, '%.1f A' % v,
                     ha='center', fontsize=9)

        t_f = np.linspace(0, 0.5, 500)
        T_sub_t, T_tr_t = 0.05, 0.20
        I_ft = (I_motor_fault * np.exp(-t_f / T_sub_t) +
                0.5 * I_motor_fault * np.exp(-t_f / T_tr_t))
        ax2.clear()
        ax2.plot(t_f * 1000, I_ft, 'r-', linewidth=2,
                 label='Motor contribution')
        ax2.axhline(I_rated, color='b', linestyle='--', linewidth=1.5,
                    label='I_rated=%.1f A' % I_rated)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('Motor Fault Current Decay')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        self._fault_canvas.draw()

    def _upd_protection(self, r):
        I_rated = r['I_line']
        I_pu = max(1.15 * I_rated, 0.01)
        I_inst = 8.0 * I_rated
        I_arr, t_EI, t_NI = protection_curves(I_pu, TMS=1.0)
        lines = [
            'PROTECTION COORDINATION',
            '=' * 58,
            '',
            'MOTOR RATED CURRENT: %.4f A' % I_rated,
            '',
            '1. OVERLOAD RELAY (IEC Class 10):',
            '   Trip setting = 1.15 x I_rated = %.4f A' % I_pu,
            '   Class 10 trips <=10 s at 6x setting (start constraint)',
            '',
            '2. OVERCURRENT RELAY (OCR) - IEC Standard Inverse:',
            '   Pickup I_p = %.4f A,  TMS = 1.0' % I_pu,
            '   Normal Inverse:   t = TMS*0.14 / ((I/Ip)^0.02 - 1)',
            '   Ext. Inverse:     t = TMS*80   / ((I/Ip)^2   - 1)',
            '',
            '3. INSTANTANEOUS ELEMENT (50):',
            '   Setting = 8 x I_rated = %.4f A' % I_inst,
            '   Avoids start-inrush operation (~6x I_rated)',
            '',
            'RELAY TYPES FOR MOTOR PROTECTION:',
            '  50/51 - Overcurrent (primary protection)',
            '  49    - Thermal overload relay',
            '  87M   - Differential (motors > 1 MVA)',
            '  46    - Negative sequence (unbalance/single phase)',
            '  27    - Undervoltage',
            '  59    - Overvoltage',
            '',
            'Starting current ~ %.1f A for 5-15 s' % (6 * I_rated),
            'Discrimination: relay must ride through start period.',
        ]
        self._set_text(self._prot_text, lines)

        ax = self._prot_axes[0]
        ax.clear()
        ax.loglog(I_arr, t_EI, 'r-', linewidth=2, label='Ext. Inverse (IEC)')
        ax.loglog(I_arr, t_NI, 'b-', linewidth=2,
                  label='Normal Inverse (IEC)')
        ax.axvline(I_pu, color='g', linestyle='--', linewidth=1.5,
                   label='I_pickup=%.1f A' % I_pu)
        ax.axvline(I_inst, color='purple', linestyle=':', linewidth=1.5,
                   label='I_inst=%.1f A' % I_inst)
        ax.axvline(6.0 * I_rated, color='orange', linestyle='-.', linewidth=1.5,
                   label='I_start~%.1f A' % (6 * I_rated))
        ax.set_xlabel('Current (A)')
        ax.set_ylabel('Trip Time (s)')
        ax.set_title('Protection Coordination Curves')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim(0.01, 1000)
        self._prot_canvas.draw()

    def _upd_speed(self, r):
        omega_ref = r['omega_r']
        T_step = self.v_T_step.get()
        Kp = self.v_Kp.get()
        Ki = self.v_Ki.get()
        Kd = self.v_Kd.get()
        t_pid, w_pid = pid_sim(Kp, Ki, Kd, omega_ref, T_step)
        t_fuz, w_fuz = fuzzy_sim(omega_ref, T_step)
        t_ol = t_pid.copy()
        w_ol = np.where(t_ol < 1.0, omega_ref, omega_ref - T_step * 0.015)
        lines = [
            'SPEED CONTROLLER ANALYSIS',
            '=' * 58,
            '',
            'Reference  omega_ref = %.4f rad/s  (%.2f rpm)' % (
                omega_ref, r['Nr']),
            'Load step torque at t=1s = %.1f Nm' % T_step,
            '',
            'MOTOR MECHANICAL PARAMETERS (assumed):',
            '  J = 0.5 kg.m^2  (rotor inertia)',
            '  B = 0.1 N.m.s/rad  (viscous friction)',
            '',
            'PID CONTROLLER:',
            '  Kp=%.2f,  Ki=%.2f,  Kd=%.2f' % (Kp, Ki, Kd),
            '  u(t) = Kp*e + Ki*integral(e) + Kd*(de/dt)',
            '  Output clipped to +-500 Nm (anti-windup)',
            '',
            'FUZZY LOGIC CONTROLLER (Mamdani-type):',
            '  Inputs: speed error e, rate de/dt',
            '  Membership: NB, NM, Z, PM, PB (triangular)',
            '  25-rule base; defuzz: centre of gravity (COG)',
            '',
            'OPEN-LOOP (V/f):',
            '  No feedback: speed droops linearly with load',
            '  Steady-state error = slip * omega_s',
            '',
            'COMPARISON:',
            '  PID    -> fast precise tracking; model-dependent',
            '  Fuzzy  -> robust to nonlinearity; no explicit model',
            '  Open L -> simplest; not for tight speed regulation',
        ]
        self._set_text(self._speed_text, lines)

        ax = self._speed_axes[0]
        ax.clear()
        ax.plot(t_pid, w_pid, 'b-', linewidth=2, label='PID')
        ax.plot(t_fuz, w_fuz, 'r--', linewidth=2, label='Fuzzy')
        ax.plot(t_ol, w_ol, 'g:', linewidth=2, label='Open Loop')
        ax.axhline(omega_ref, color='k', linestyle=':', linewidth=1.5,
                   label='Reference')
        ax.axvline(1.0, color='gray', linestyle='--', linewidth=1.0,
                   alpha=0.7, label='Load step')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (rad/s)')
        ax.set_title('Speed Response - Step Load %.0f Nm at t=1 s' % T_step)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(max(0.0, omega_ref * 0.7), omega_ref * 1.15)
        self._speed_canvas.draw()

    def _upd_thermal(self, r):
        P_loss = r['P_input'] - r['P_shaft']
        t_th, T_th = thermal_model(P_loss)
        T_ss = float(T_th[-1])
        hours = self.v_hours.get()
        tariff = self.v_tariff.get()
        ann_kWh = r['P_input'] / 1000.0 * hours
        ann_cost = ann_kWh * tariff
        out_kWh = r['P_shaft'] / 1000.0 * hours
        loss_kWh = ann_kWh - out_kWh
        savings = (ann_kWh * (1 - r['eff'] / 100) -
                   ann_kWh * (1 - 91.0 / 100)) * tariff
        lines = [
            'THERMAL & ECONOMIC ANALYSIS',
            '=' * 58,
            '',
            'THERMAL MODEL (1st-order lumped):',
            '  C_th * d(DeltaT)/dt = P_loss - DeltaT/R_th',
            '  Steady state: DeltaT_ss = P_loss * R_th',
            '  Time constant tau = R_th * C_th (= 200 s here)',
            '',
            'LOSS BREAKDOWN:',
            '  Stator loss    = %.2f W' % r['stator_loss'],
            '  Rotor Cu loss  = %.2f W' % r['P_rotor_cu'],
            '  Fric+core loss = %.2f W' % r['P_fric_w'],
            '  Total losses   = %.2f W' % P_loss,
            '',
            'TEMPERATURE RESULTS:',
            '  Ambient T_amb        = 25.0 C',
            '  Steady-state DeltaT  = %.1f K' % (T_ss - 25.0),
            '  Winding temperature  = %.1f C' % T_ss,
            '  Insulation classes:  B=130C  F=155C  H=180C',
            '  Margin to class F    = %.1f C' % (155 - T_ss),
            '',
            'ECONOMIC ANALYSIS:',
            '  Operating h/year     = %.0f h' % hours,
            '  Tariff               = $%.3f/kWh' % tariff,
            '  Annual input energy  = %.2f kWh' % ann_kWh,
            '  Useful energy (out)  = %.2f kWh' % out_kWh,
            '  Energy wasted        = %.2f kWh' % loss_kWh,
            '  Annual energy cost   = $%.2f' % ann_cost,
            '  20-year total cost   = $%.2f' % (ann_cost * 20),
            '',
            'IE EFFICIENCY CLASSIFICATION (IEC 60034-30):',
            '  Current eta = %.2f %%' % r['eff'],
            '  IE1 ~86%  IE2 ~89%  IE3 ~91%  IE4 ~93%',
            '  Upgrade IE1->IE3 saves ~$%.2f/year' % max(savings, 0),
        ]
        self._set_text(self._therm_text, lines)

        ax1, ax2 = self._therm_axes
        ax1.clear()
        ax1.plot(t_th, T_th, 'r-', linewidth=2, label='Winding temp')
        ax1.axhline(155, color='orange', linestyle='--', linewidth=1.5,
                    label='Class F (155C)')
        ax1.axhline(130, color='g', linestyle='--', linewidth=1.5,
                    label='Class B (130C)')
        ax1.axhline(25, color='b', linestyle=':', linewidth=1.0,
                    label='Ambient (25C)')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Temperature (C)')
        ax1.set_title('Motor Winding Temperature Rise')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.clear()
        ax2.pie([out_kWh, loss_kWh],
                labels=['Useful\n%.0f kWh' % out_kWh,
                        'Losses\n%.0f kWh' % loss_kWh],
                colors=['#4CAF50', '#F44336'],
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 9})
        ax2.set_title('Annual Energy\nTotal: %.0f kWh  Cost: $%.0f' % (
            ann_kWh, ann_cost))
        self._therm_canvas.draw()

    def _upd_harmonics(self, r):
        ords, pct, I_h, I_rms, thd_calc = harmonic_spectrum(r['I_line'])
        pf_disp = r['pf']
        pf_true = pf_disp / math.sqrt(1.0 + (thd_calc / 100.0) ** 2)
        lines = [
            'HARMONICS & POWER QUALITY ANALYSIS',
            '=' * 58,
            '',
            'FUNDAMENTAL CURRENT I1 = %.4f A' % r['I_line'],
            'Calculated THD_I       = %.2f %%' % thd_calc,
            '',
            'HARMONIC SPECTRUM (typical VFD-fed motor):',
            '  %-8s%-14s%-15s' % ('Order', '% of I1', 'Magnitude (A)'),
            '  ' + '-' * 38,
        ]
        for o, p, i in zip(ords, pct, I_h):
            lines.append('  %-8d%-14.1f%-15.4f' % (o, p, i))
        lines += [
            '',
            'POWER QUALITY:',
            '  I_rms                   = %.4f A' % I_rms,
            '  Displacement PF (cos phi)= %.4f' % pf_disp,
            '  True PF                  = %.4f' % pf_true,
            '  PF degradation           = %.2f %%' % (
                (pf_disp - pf_true) / pf_disp * 100),
            '',
            '  Stray load loss (est.) ~ %.2f W' % (
                0.015 * (1 + (thd_calc / 100) ** 2) * r['P_input']),
            '',
            'IEEE 519-2022 LIMITS:',
            '  I_sc/I_L < 20  -> THD <= 5%',
            '  I_sc/I_L 20-50 -> THD <= 8%',
            '  Status: THD=%.1f %% -> %s' % (
                thd_calc,
                'COMPLIANT' if thd_calc <= 8 else 'EXCEEDS LIMIT - FILTER NEEDED'),
            '',
            'MITIGATION:',
            '  5th/7th tuned passive LC filter',
            '  Active Power Filter (APF)',
            '  12-pulse or 18-pulse rectifier',
            '  Line reactor (reduces harmonic + protects VFD)',
        ]
        self._set_text(self._harm_text, lines)

        ax1, ax2 = self._harm_axes
        ax1.clear()
        ax1.bar(ords, I_h, color='steelblue', edgecolor='white', width=0.6)
        ax1.set_xlabel('Harmonic Order')
        ax1.set_ylabel('Current (A)')
        ax1.set_title('Harmonic Spectrum  (THD=%.1f%%)' % thd_calc)
        ax1.set_xticks(ords)
        ax1.grid(True, axis='y', alpha=0.3)
        for o, i in zip(ords, I_h):
            ax1.text(o, i + max(I_h) * 0.02, '%.3f' % i,
                     ha='center', fontsize=8)

        t_w = np.linspace(0, 2.0 / r['freq'], 1000)
        i_wave = np.zeros(len(t_w))
        for o, ih in zip(ords, I_h):
            i_wave += ih * np.sin(2 * PI * o * r['freq'] * t_w)
        ax2.clear()
        ax2.plot(t_w * 1000, i_wave, 'b-', linewidth=1.5)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('Synthesised Current Waveform (2 cycles)')
        ax2.grid(True, alpha=0.3)
        self._harm_canvas.draw()

    def _upd_validation(self, r):
        bal1 = r['P_input'] - r['stator_loss'] - r['P_rotor_cu'] - r['P_mech']
        bal2 = r['P_mech'] - r['P_fric_w'] - r['P_shaft']
        bal3 = r['P_rotor_cu'] - r['s'] * r['P_airgap']
        check_balance = lambda v: 'PASS' if abs(v) < 0.01 else 'FAIL'
        lines = [
            'MODEL VALIDATION & RESULTS VERIFICATION',
            '=' * 58,
            '',
            '1. POWER BALANCE CHECKS (must be ~0):',
            '   P_in - P_stator - P_rotor_cu - P_mech',
            '   = %.6f - %.6f - %.6f - %.6f' % (
                r['P_input'], r['stator_loss'],
                r['P_rotor_cu'], r['P_mech']),
            '   = %.2e W  [%s]' % (bal1, check_balance(bal1)),
            '',
            '   P_mech - P_fric - P_shaft',
            '   = %.6f - %.6f - %.6f' % (
                r['P_mech'], r['P_fric_w'], r['P_shaft']),
            '   = %.2e W  [%s]' % (bal2, check_balance(bal2)),
            '',
            '   Rotor_Cu = s * P_airgap  check',
            '   delta = %.2e W  [%s]' % (bal3, check_balance(bal3)),
            '',
            '2. SANITY CHECKS:',
            '   0 < s < 1:     s=%.6f  [%s]' % (
                r['s'], 'OK' if 0 < r['s'] < 1 else 'FAIL'),
            '   Nr < Ns:       %.2f < %.2f  [%s]' % (
                r['Nr'], r['Ns'],
                'OK' if r['Nr'] < r['Ns'] else 'FAIL'),
            '   0 < eta < 100: %.4f%%  [%s]' % (
                r['eff'], 'OK' if 0 < r['eff'] < 100 else 'FAIL'),
            '   P_shaft < P_in: [%s]' % (
                'OK' if r['P_shaft'] < r['P_input'] else 'FAIL'),
            '',
            '3. TEXTBOOK REFERENCE (default inputs):',
            '   poles=6, f=50, T_u=160, T_f=10, cpm=120, stator=800 W',
            '',
            '   Expected:  Ns=1000 rpm  s=4%  Nr=960 rpm',
            '   P_shaft~16085W  Rotor_Cu~712W  P_in~18602W  eta~86.47%',
            '',
            '   Computed:',
            '   Ns=%.4f rpm  s=%.6f  Nr=%.4f rpm' % (
                r['Ns'], r['s'], r['Nr']),
            '   P_shaft=%.2f W  Rotor_Cu=%.2f W' % (
                r['P_shaft'], r['P_rotor_cu']),
            '   P_input=%.2f W  eta=%.4f %%' % (r['P_input'], r['eff']),
            '',
            '4. REFERENCES:',
            '   Chapman - Electric Machinery Fundamentals (5th ed.)',
            '   Fitzgerald, Kingsley, Umans - Electric Machinery (6th ed.)',
            '   IEC 60034-1  Motor ratings; IEC 60909 Fault currents',
            '   IEEE 519-2022  Harmonic limits',
        ]
        self._set_text(self._val_text, lines)

        ax1, ax2 = self._val_axes
        ax1.clear()
        stages = ['Input', 'After\nStator', 'Air\ngap',
                  'After\nRotor Cu', 'Mech', 'After\nFric', 'Shaft']
        pows = [r['P_input'],
                r['P_input'] - r['stator_loss'],
                r['P_airgap'],
                r['P_mech'],
                r['P_mech'],
                r['P_mech'] - r['P_fric_w'],
                r['P_shaft']]
        ax1.barh(stages, pows, color='steelblue', edgecolor='white')
        ax1.set_xlabel('Power (W)')
        ax1.set_title('Power Flow Through Motor')
        ax1.grid(True, axis='x', alpha=0.3)
        for i, v in enumerate(pows):
            ax1.text(v + max(pows) * 0.005, i, '%.0f' % v,
                     va='center', fontsize=8)

        slips = np.linspace(0.005, 0.20, 300)
        effs = []
        for ss in slips:
            Nr_s = r['Ns'] * (1.0 - ss)
            omega_rs = 2.0 * PI * Nr_s / 60.0
            P_sh = r['T_useful'] * omega_rs
            P_in = r['T_total'] * r['omega_s'] + r['stator_loss']
            effs.append(P_sh / P_in * 100.0 if P_in > 0 else 0.0)
        ax2.clear()
        ax2.plot(slips * 100.0, effs, 'g-', linewidth=2)
        ax2.axvline(r['s'] * 100, color='r', linestyle='--', linewidth=1.5,
                    label='Op. s=%.2f%%  eta=%.1f%%' % (
                        r['s'] * 100, r['eff']))
        ax2.set_xlabel('Slip (%)')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('Efficiency vs Slip')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        self._val_canvas.draw()

    # -------------------------------------------------------------------------
    # Control buttons
    # -------------------------------------------------------------------------
    def _start(self):
        self.v_status.set('Running ...')
        self.update()

    def _stop(self):
        self.v_status.set('Stopped')

    def _reset(self):
        self.v_poles.set(6)
        self.v_freq.set(50.0)
        self.v_T_useful.set(160.0)
        self.v_T_fric.set(10.0)
        self.v_rotor_cpm.set(120.0)
        self.v_stator_W.set(800.0)
        self.v_vline.set(400.0)
        self.v_pf.set(0.87)
        self.v_Kp.set(5.0)
        self.v_Ki.set(2.0)
        self.v_Kd.set(0.5)
        self.v_T_step.set(50.0)
        self.v_tariff.set(0.12)
        self.v_hours.set(8000.0)
        self.v_status.set('Reset to default values')
        self.update()


if __name__ == '__main__':
    root = tk.Tk()
    app = MotorApp(root)
    root.mainloop()
