import tkinter as tk
from tkinter import ttk
from math import sqrt

HP_TO_W = 746.0

class MotorApp:
    def __init__(self, root):
        self.root = root
        root.title('Induction Motor Advanced Analysis')
        root.geometry('1220x820')
        root.minsize(980, 680)
        self.v_line = tk.DoubleVar(value=400)
        self.freq = tk.DoubleVar(value=50)
        self.poles = tk.IntVar(value=6)
        self.hp = tk.DoubleVar(value=20)
        self.speed = tk.DoubleVar(value=995)
        self.pf = tk.DoubleVar(value=0.87)
        self.stator_cu = tk.DoubleVar(value=1500)
        self.controller = tk.StringVar(value='PID')
        self.status = tk.StringVar(value='Ready')
        self.build_ui()
        self.update_results()

    def solve(self):
        poles = max(2, int(round(self.poles.get() / 2)) * 2)
        Ns = 120 * self.freq.get() / poles
        s = max(0.0, min((Ns - self.speed.get()) / Ns, 0.999))
        Pm = self.hp.get() * HP_TO_W
        rotor_cu = (s / (1 - s)) * Pm if s < 0.999 else 0.0
        rotor_input = Pm + rotor_cu
        stator_input = rotor_input + self.stator_cu.get()
        current = stator_input / (sqrt(3) * self.v_line.get() * self.pf.get())
        return Ns, s, rotor_cu, stator_input, current

    def add_slider(self, parent, row, label, var, frm, to):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=6, pady=6)
        ttk.Scale(parent, variable=var, from_=frm, to=to, orient='horizontal', command=lambda e: self.update_results()).grid(row=row, column=1, sticky='ew', padx=6, pady=6)
        ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=2, padx=6, pady=6)

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill='both', expand=True)
        left = ttk.LabelFrame(outer, text='Input parameters and control')
        left.pack(side='left', fill='y', padx=(0, 10))
        left.grid_columnconfigure(1, weight=1)
        fields = [
            ('Line voltage (V)', self.v_line, 300, 500),
            ('Frequency (Hz)', self.freq, 40, 60),
            ('Poles', self.poles, 2, 12),
            ('Developed power (HP)', self.hp, 5, 50),
            ('Running speed (rpm)', self.speed, 500, 1495),
            ('Power factor', self.pf, 0.5, 1.0),
            ('Stator copper loss (W)', self.stator_cu, 100, 4000),
        ]
        for r, (lab, var, frm, to) in enumerate(fields):
            self.add_slider(left, r, lab, var, frm, to)
        ttk.Label(left, text='Controller').grid(row=7, column=0, sticky='w', padx=6, pady=6)
        ttk.Combobox(left, textvariable=self.controller, values=['Open Loop', 'PID', 'Fuzzy'], state='readonly').grid(row=7, column=1, columnspan=2, sticky='ew', padx=6, pady=6)
        btns = ttk.Frame(left)
        btns.grid(row=8, column=0, columnspan=3, sticky='ew', padx=6, pady=6)
        ttk.Button(btns, text='Start', command=lambda: self.status.set('Simulation placeholder started')).pack(side='left', fill='x', expand=True, padx=3)
        ttk.Button(btns, text='Stop', command=lambda: self.status.set('Simulation stopped')).pack(side='left', fill='x', expand=True, padx=3)
        ttk.Button(btns, text='Reset', command=self.reset).pack(side='left', fill='x', expand=True, padx=3)
        ttk.Label(left, textvariable=self.status).grid(row=9, column=0, columnspan=3, sticky='w', padx=6, pady=6)

        right = ttk.Frame(outer)
        right.pack(side='left', fill='both', expand=True)
        ttk.Label(right, text='Detailed solution and advanced electrical engineering tabs', font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(0, 8))
        self.summary = ttk.Label(right, text='', wraplength=760, justify='left')
        self.summary.pack(fill='x', pady=(0, 8))
        nb = ttk.Notebook(right)
        nb.pack(fill='both', expand=True)
        self.text = {}
        for name in ['Overview', 'Model', 'Fault Current', 'Protection', 'Speed Control', 'Thermal & Economic', 'Harmonics', 'Validation']:
            frame = ttk.Frame(nb, padding=10)
            nb.add(frame, text=name)
            t = tk.Text(frame, wrap='word')
            t.pack(fill='both', expand=True)
            self.text[name] = t

    def reset(self):
        self.v_line.set(400); self.freq.set(50); self.poles.set(6); self.hp.set(20)
        self.speed.set(995); self.pf.set(0.87); self.stator_cu.set(1500)
        self.status.set('Reset complete')
        self.update_results()

    def update_results(self):
        Ns, s, rotor_cu, stator_input, current = self.solve()
        self.summary.config(text=f'Synchronous speed = {Ns:.2f} rpm | Slip = {100*s:.3f}% | Rotor copper loss = {rotor_cu:.2f} W | Line current = {current:.2f} A')
        texts = {
            'Overview': f'Given solution:\nSlip = {100*s:.3f}%\nRotor copper loss = {rotor_cu:.2f} W\nLine current = {current:.2f} A',
            'Model': 'Electrical model:\nNs = 120f/P\ns = (Ns-N)/Ns\nPrcu = [s/(1-s)] Pm\nMechanical model: J dω/dt = Te - TL - Bω',
            'Fault Current': f'Use Vphase / Zeq for an approximate symmetrical fault-current study.\nMotor feeder current estimate = {current:.2f} A',
            'Protection': f'Overload pickup ≈ {1.15*current:.2f} A\nInstantaneous pickup ≈ {8*current:.2f} A',
            'Speed Control': 'Compare open-loop, PID, and fuzzy strategies under a step-load disturbance.',
            'Thermal & Economic': f'Estimated input power = {stator_input:.2f} W\nUse this with operating hours and tariff for annual cost.',
            'Harmonics': 'True power factor decreases when THDi increases: PFtrue = PF / sqrt(1 + THDi^2).',
            'Validation': f'Power-balance residual = {stator_input - self.stator_cu.get() - rotor_cu - self.hp.get()*HP_TO_W:.6f} W',
        }
        for key, value in texts.items():
            self.text[key].delete('1.0', 'end')
            self.text[key].insert('1.0', value)

root = tk.Tk()
app = MotorApp(root)
root.mainloop()
