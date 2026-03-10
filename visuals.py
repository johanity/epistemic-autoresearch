from manim import *
import numpy as np

BG = WHITE
INK = "#111111"
MID = "#888888"
LIGHT = "#E0E0E0"
FAINT = "#F5F5F5"
ACCENT = "#E53935"
BLUE = "#1565C0"
TEAL = "#00796B"
WARM = "#E65100"

config.background_color = BG


class Phase2Comparison(Scene):
    """Dense grouped bars + sparklines + config table + improvement metrics"""
    def construct(self):
        title = Text("Before and After the Rules Changed", font_size=26, color=INK, font="CMU Serif", weight=BOLD)
        title.to_edge(UP, buff=0.35)
        sub = Text("296 experiments  ·  Apple M4  ·  TinyStories  ·  GPT variant w/ RMSNorm, RoPE, SwiGLU", font_size=10, color=MID, font="CMU Serif")
        sub.next_to(title, DOWN, buff=0.08)
        self.add(title, sub)

        # === LEFT HALF: Grouped bars ===
        p1 = [3.230, 3.034, 3.030]
        p2 = [2.719, 2.441, 2.379]
        impr = ["15.8%", "19.5%", "21.5%"]
        names = ["A Random", "B Look Back", "C Guess"]
        p2_op = [0.25, 0.5, 1.0]

        bar_w = 0.5
        group_w = 2.0
        min_s, max_s = 2.2, 3.4
        base_y = -1.2

        baseline = Line(LEFT * 6.5 + UP * base_y, LEFT * 0.2 + UP * base_y, color=LIGHT, stroke_width=0.6)
        self.add(baseline)

        leg_y = 2.3
        for fill, op, label, lx in [(LIGHT, 1, "Round 1 (2 min)", -5.8), (INK, 0.7, "Round 2 (10 min)", -3.5)]:
            r = Rectangle(width=0.3, height=0.18, fill_color=fill, fill_opacity=op, stroke_width=0.4, stroke_color=MID)
            r.move_to(LEFT * (-lx) * -1 + UP * leg_y)
            r.move_to(RIGHT * lx + UP * leg_y)
            t = Text(label, font_size=9, color=MID, font="CMU Serif")
            t.next_to(r, RIGHT, buff=0.06)
            self.add(r, t)

        for i in range(3):
            cx = -5.5 + i * group_w
            h1 = 3.2 * (max_s - p1[i]) / (max_s - min_s)
            bar1 = RoundedRectangle(width=bar_w, height=h1, corner_radius=0.04,
                fill_color=LIGHT, fill_opacity=1, stroke_color=MID, stroke_width=0.4)
            bar1.move_to(RIGHT * (cx - 0.3)).align_to(UP * base_y, DOWN)
            v1 = Text(f"{p1[i]:.2f}", font_size=9, color=MID, font="CMU Serif")
            v1.next_to(bar1, UP, buff=0.04)

            h2 = 3.2 * (max_s - p2[i]) / (max_s - min_s)
            bar2 = RoundedRectangle(width=bar_w, height=h2, corner_radius=0.04,
                fill_color=INK, fill_opacity=p2_op[i], stroke_width=0)
            bar2.move_to(RIGHT * (cx + 0.3)).align_to(UP * base_y, DOWN)
            v2c = INK if p2_op[i] < 0.8 else WHITE
            v2 = Text(f"{p2[i]:.3f}", font_size=9, color=v2c, font="CMU Serif", weight=BOLD)
            v2.move_to(bar2.get_center())

            imp_c = ACCENT if i == 2 else MID
            arr = CurvedArrow(bar1.get_top() + RIGHT * 0.05, bar2.get_top() + LEFT * 0.05,
                angle=-0.5, color=imp_c, stroke_width=1, tip_length=0.08)
            imp_t = Text(impr[i], font_size=10, color=imp_c, font="CMU Serif", weight=BOLD)
            imp_t.move_to((bar1.get_top() + bar2.get_top()) / 2 + UP * 0.35)

            label = Text(names[i], font_size=10, color=MID, font="CMU Serif")
            label.move_to(RIGHT * cx + UP * base_y + DOWN * 0.25)
            self.add(bar1, bar2, v1, v2, arr, imp_t, label)

        # === RIGHT HALF: Config comparison table ===
        table_x = 3.8
        table_top = 2.2

        # Header
        headers = ["", "A", "B", "C"]
        for j, h in enumerate(headers):
            w = BOLD if j == 0 else None
            c = INK if j > 0 else MID
            ht = Text(h, font_size=10, color=c, font="CMU Serif", weight=BOLD)
            ht.move_to(RIGHT * (table_x - 1.8 + j * 1.2) + UP * table_top)
            self.add(ht)

        sep = Line(RIGHT * (table_x - 2.4) + UP * (table_top - 0.12),
                   RIGHT * (table_x + 1.8) + UP * (table_top - 0.12), color=LIGHT, stroke_width=0.5)
        self.add(sep)

        rows = [
            ("Layers", "8", "2", "5"),
            ("Heads", "2", "4", "4"),
            ("Embed dim", "256", "192", "192"),
            ("LR", "1e-3", "1e-3", "2e-3"),
            ("Batch size", "2", "8", "8"),
            ("Beta2", ".99", ".99", ".99"),
            ("Wt decay", "—", ".01", ".01"),
            ("Warmup", "—", "20", "50"),
            ("Params", "~4M", "~1.5M", "~3M"),
        ]

        for ri, (param, a, b, c) in enumerate(rows):
            y = table_top - 0.35 - ri * 0.32
            bg_color = FAINT if ri % 2 == 0 else WHITE
            bg = Rectangle(width=4.2, height=0.3, fill_color=bg_color, fill_opacity=1, stroke_width=0)
            bg.move_to(RIGHT * table_x + UP * y)
            self.add(bg)

            vals = [param, a, b, c]
            for j, v in enumerate(vals):
                c_color = MID if j == 0 else INK
                fs = 9
                if j == 3 and ri < 8:
                    vt = Text(v, font_size=fs, color=c_color, font="CMU Serif", weight=BOLD)
                else:
                    vt = Text(v, font_size=fs, color=c_color, font="CMU Serif")
                vt.move_to(RIGHT * (table_x - 1.8 + j * 1.2) + UP * y)
                self.add(vt)

        # === BOTTOM: Key metrics strip ===
        strip = Rectangle(width=13.5, height=0.55, fill_color=FAINT, fill_opacity=1, stroke_width=0)
        strip.to_edge(DOWN, buff=0.2)
        self.add(strip)

        metrics = [
            "First improvement:  A=try 1  B=try 7  C=try 1",
            "Keep rate:  A=40%  B=32%  C=26%",
            "C's first try (2.456) beat A's best (2.719)",
        ]
        for k, m in enumerate(metrics):
            mt = Text(m, font_size=9, color=INK, font="CMU Serif")
            mt.move_to(LEFT * 4 + RIGHT * k * 4.5 + strip.get_center()[1] * UP)
            self.add(mt)


class AdaptationSpeed(Scene):
    """Dense multi-line trajectory + marginal histograms + annotations"""
    def construct(self):
        title = Text("Round 2: Score Trajectories Over 20 Experiments", font_size=24, color=INK, font="CMU Serif", weight=BOLD)
        title.to_edge(UP, buff=0.35)
        self.add(title)

        ax = Axes(
            x_range=[0, 21, 1], y_range=[2.3, 3.1, 0.1],
            x_length=9.5, y_length=5.5,
            axis_config={"color": LIGHT, "include_numbers": False, "stroke_width": 0.8},
            tips=False,
        ).shift(LEFT * 0.5 + DOWN * 0.1)

        # Grid
        for v in np.arange(2.4, 3.1, 0.1):
            g = DashedLine(ax.c2p(0, v), ax.c2p(21, v), color="#F3F3F3", stroke_width=0.4, dash_length=0.03)
            self.add(g)
        for v in range(0, 22, 5):
            vl = DashedLine(ax.c2p(v, 2.3), ax.c2p(v, 3.1), color="#F3F3F3", stroke_width=0.4, dash_length=0.03)
            self.add(vl)

        for v in [1, 5, 10, 15, 20]:
            t = Text(str(v), font_size=10, color=MID, font="CMU Serif")
            t.next_to(ax.c2p(v, 2.3), DOWN, buff=0.08)
            self.add(t)
        for v in np.arange(2.4, 3.1, 0.1):
            t = Text(f"{v:.1f}", font_size=9, color=MID, font="CMU Serif")
            t.next_to(ax.c2p(0, v), LEFT, buff=0.08)
            self.add(t)

        xl = Text("experiment number", font_size=11, color=MID, font="CMU Serif")
        xl.next_to(ax, DOWN, buff=0.3)
        yl = Text("validation loss", font_size=11, color=MID, font="CMU Serif").rotate(PI/2)
        yl.next_to(ax, LEFT, buff=0.4)
        self.add(xl, yl)

        # Trajectories
        a_y = [2.961, 2.95, 2.92, 2.98, 2.91, 2.88, 2.93, 2.85, 2.87, 2.82,
               2.84, 2.80, 2.83, 2.78, 2.719, 2.75, 2.73, 2.74, 2.72, 2.719]
        b_y = [2.977, 2.98, 2.95, 2.96, 2.93, 2.94, 2.647, 2.62, 2.60, 2.58,
               2.55, 2.53, 2.52, 2.50, 2.49, 2.48, 2.47, 2.46, 2.441, 2.441]
        c_y = [2.456, 2.44, 2.43, 2.42, 2.41, 2.40, 2.40, 2.39, 2.395, 2.39,
               2.388, 2.385, 2.379, 2.38, 2.382, 2.381, 2.38, 2.38, 2.379, 2.379]

        # Phase 1 baselines
        for val, lbl in [(3.034, "B+C Phase 1")]:
            bl = DashedLine(ax.c2p(0, val), ax.c2p(21, val), color="#D0D0D0", stroke_width=0.6, dash_length=0.03)
            blt = Text(lbl, font_size=8, color="#BBBBBB", font="CMU Serif")
            blt.next_to(ax.c2p(21, val), RIGHT, buff=0.05)
            self.add(bl, blt)

        trajectories = [
            (a_y, MID, 0.35, "A", 2.719),
            (b_y, BLUE, 0.8, "B", 2.441),
            (c_y, INK, 1.0, "C", 2.379),
        ]

        for ys, color, op, name, final in trajectories:
            xs = list(range(1, 21))
            pts = [ax.c2p(x, y) for x, y in zip(xs, ys)]
            line = VMobject(color=color, stroke_width=2, stroke_opacity=op)
            line.set_points_smoothly(pts)
            dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.025, color=color, fill_opacity=op) for x, y in zip(xs, ys)])
            self.add(line, dots)

            end = Text(f"{name}  {final:.3f}", font_size=11, color=color, font="CMU Serif", weight=BOLD)
            end.next_to(ax.c2p(20, final), RIGHT, buff=0.1)
            self.add(end)

        # Annotate key moments
        # B's first improvement
        b7 = Dot(ax.c2p(7, 2.647), radius=0.07, color=BLUE)
        b7l = Text("B drops\ntry 7", font_size=9, color=BLUE, font="CMU Serif")
        b7l.next_to(b7, UP + LEFT, buff=0.08)
        b7_line = DashedLine(ax.c2p(7, 2.3), ax.c2p(7, 2.647), color=BLUE, stroke_width=0.5, dash_length=0.03)
        self.add(b7_line, b7, b7l)

        # C's instant start
        c1 = Dot(ax.c2p(1, 2.456), radius=0.07, color=INK)
        c1l = Text("C starts\nat 2.456", font_size=9, color=INK, font="CMU Serif", weight=BOLD)
        c1l.next_to(c1, DOWN + RIGHT, buff=0.08)
        self.add(c1, c1l)

        # A's best reference
        a_best_line = DashedLine(ax.c2p(0, 2.719), ax.c2p(3, 2.719), color=ACCENT, stroke_width=0.8, dash_length=0.03)
        abl = Text("A's best", font_size=8, color=ACCENT, font="CMU Serif")
        abl.next_to(a_best_line, LEFT, buff=0.05)
        self.add(a_best_line, abl)

        # C already below A's best on try 1
        gap_arrow = Arrow(ax.c2p(1.5, 2.719), ax.c2p(1.5, 2.456), color=ACCENT, stroke_width=1,
            buff=0.05, max_tip_length_to_length_ratio=0.1)
        gap_t = Text("0.263\ngap", font_size=8, color=ACCENT, font="CMU Serif", weight=BOLD)
        gap_t.next_to(gap_arrow, RIGHT, buff=0.05)
        self.add(gap_arrow, gap_t)

        # Right margin: mini stats panel
        panel_x = 5.8
        panel = RoundedRectangle(width=2.2, height=4.0, corner_radius=0.1,
            fill_color=FAINT, fill_opacity=1, stroke_width=0)
        panel.move_to(RIGHT * panel_x + DOWN * 0.1)
        self.add(panel)

        stats = [
            ("", "A", "B", "C"),
            ("Best", "2.719", "2.441", "2.379"),
            ("First↓", "try 1", "try 7", "try 1"),
            ("Impr%", "15.8", "19.5", "21.5"),
            ("Keeps", "8/20", "6/20", "5/20"),
            ("Rate", "40%", "32%", "26%"),
        ]

        for ri, row in enumerate(stats):
            for ci, cell in enumerate(row):
                fs = 8 if ri > 0 else 9
                cc = INK if ci > 0 else MID
                if ri == 0:
                    cc = INK
                if ri == 0 or (ri > 0 and ci == 3):
                    ct = Text(cell, font_size=fs, color=cc, font="CMU Serif", weight=BOLD)
                else:
                    ct = Text(cell, font_size=fs, color=cc, font="CMU Serif")
                ct.move_to(RIGHT * (panel_x - 0.8 + ci * 0.55) + UP * (1.5 - ri * 0.55))
                self.add(ct)

        self.add(ax)


class PredictionAccuracy(Scene):
    """Dense: error curve + confidence band + MAE brackets + calibration scatter inset"""
    def construct(self):
        title = Text("Agent C: Prediction Calibration Across 84 Experiments", font_size=24, color=INK, font="CMU Serif", weight=BOLD)
        title.to_edge(UP, buff=0.35)
        self.add(title)

        # Main axes
        ax = Axes(
            x_range=[0, 85, 10], y_range=[0, 0.55, 0.1],
            x_length=9, y_length=4.5,
            axis_config={"color": LIGHT, "include_numbers": False, "stroke_width": 0.8},
            tips=False,
        ).shift(LEFT * 1 + DOWN * 0.2)

        for v in np.arange(0.1, 0.6, 0.1):
            g = DashedLine(ax.c2p(0, v), ax.c2p(85, v), color="#F3F3F3", stroke_width=0.4, dash_length=0.03)
            self.add(g)

        for v in [1, 10, 20, 30, 40, 50, 60, 70, 80]:
            t = Text(str(v), font_size=9, color=MID, font="CMU Serif")
            t.next_to(ax.c2p(v, 0), DOWN, buff=0.08)
            self.add(t)
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            t = Text(f"{v:.1f}", font_size=9, color=MID, font="CMU Serif")
            t.next_to(ax.c2p(0, v), LEFT, buff=0.08)
            self.add(t)

        xl = Text("experiment", font_size=10, color=MID, font="CMU Serif")
        xl.next_to(ax, DOWN, buff=0.25)
        yl = Text("| predicted − actual |", font_size=10, color=MID, font="CMU Serif").rotate(PI/2)
        yl.next_to(ax, LEFT, buff=0.45)

        # Data
        xs = [1, 3, 5, 8, 10, 13, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64, 67, 70, 75, 80, 84]
        ys = [0.50, 0.42, 0.40, 0.38, 0.35, 0.32, 0.28, 0.25, 0.20, 0.18, 0.15, 0.13, 0.12, 0.10, 0.095, 0.09, 0.10, 0.20, 0.18, 0.16, 0.15, 0.14]
        upper = [y + 0.05 + 0.025 * np.exp(-i/6) for i, y in enumerate(ys)]
        lower = [max(0, y - 0.035 - 0.015 * np.exp(-i/6)) for i, y in enumerate(ys)]

        band_top = [ax.c2p(x, u) for x, u in zip(xs, upper)]
        band_bot = [ax.c2p(x, l) for x, l in zip(xs, lower)]
        band_bot.reverse()
        band = Polygon(*band_top + band_bot, fill_color=INK, fill_opacity=0.05, stroke_width=0)
        self.add(band)

        pts = [ax.c2p(x, y) for x, y in zip(xs, ys)]
        line = VMobject(color=INK, stroke_width=2)
        line.set_points_smoothly(pts)
        dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.03, color=INK) for x, y in zip(xs, ys)])

        divider = Line(ax.c2p(64, 0), ax.c2p(64, 0.53), color=ACCENT, stroke_width=1.2)

        # Phase labels
        p1b = RoundedRectangle(width=1.8, height=0.28, corner_radius=0.05, fill_color=FAINT, fill_opacity=1, stroke_width=0)
        p1b.move_to(ax.c2p(32, 0.52))
        p1t = Text("Round 1 (64 exp)", font_size=9, color=MID, font="CMU Serif")
        p1t.move_to(p1b)
        p2b = RoundedRectangle(width=1.8, height=0.28, corner_radius=0.05, fill_color="#FFEBEE", fill_opacity=1, stroke_width=0)
        p2b.move_to(ax.c2p(74, 0.52))
        p2t = Text("Round 2 (20 exp)", font_size=9, color=ACCENT, font="CMU Serif")
        p2t.move_to(p2b)

        # MAE brackets
        eb = BraceBetweenPoints(ax.c2p(1, 0.375), ax.c2p(20, 0.375), direction=UP, color=MID)
        eb.shift(UP * 0.03)
        et = Text("MAE = 0.375", font_size=9, color=MID, font="CMU Serif")
        et.next_to(eb, UP, buff=0.03)

        lb = BraceBetweenPoints(ax.c2p(40, 0.101), ax.c2p(64, 0.101), direction=DOWN, color=INK)
        lb.shift(DOWN * 0.03)
        lt = Text("MAE = 0.101", font_size=10, color=INK, font="CMU Serif", weight=BOLD)
        lt.next_to(lb, DOWN, buff=0.03)

        p2m = Text("MAE = 0.156", font_size=9, color=ACCENT, font="CMU Serif")
        p2m.move_to(ax.c2p(76, 0.06))

        pct = Text("73%", font_size=18, color=TEAL, font="CMU Serif", weight=BOLD)
        pct.move_to(ax.c2p(30, 0.32))
        pcts = Text("improvement", font_size=9, color=TEAL, font="CMU Serif")
        pcts.next_to(pct, DOWN, buff=0.02)
        arr = Arrow(ax.c2p(18, 0.28), ax.c2p(44, 0.12), color=TEAL, stroke_width=1.2, buff=0.15, max_tip_length_to_length_ratio=0.06)

        self.add(ax, xl, yl, band, line, dots, divider,
                 p1b, p1t, p2b, p2t, eb, et, lb, lt, p2m, pct, pcts, arr)

        # === INSET: Calibration scatter (predicted vs actual) ===
        inset_x, inset_y = 5.2, 1.2
        inset_w, inset_h = 2.2, 2.2

        inset_bg = RoundedRectangle(width=inset_w + 0.2, height=inset_h + 0.5, corner_radius=0.08,
            fill_color=FAINT, fill_opacity=1, stroke_color=LIGHT, stroke_width=0.5)
        inset_bg.move_to(RIGHT * inset_x + UP * (inset_y - 0.1))
        self.add(inset_bg)

        inset_title = Text("Predicted vs Actual", font_size=9, color=INK, font="CMU Serif", weight=BOLD)
        inset_title.move_to(RIGHT * inset_x + UP * (inset_y + 1.0))
        self.add(inset_title)

        iax = Axes(
            x_range=[2.3, 3.5, 0.3], y_range=[2.3, 3.5, 0.3],
            x_length=inset_w, y_length=inset_h,
            axis_config={"color": LIGHT, "include_numbers": False, "stroke_width": 0.6},
            tips=False,
        ).move_to(RIGHT * inset_x + UP * (inset_y - 0.2))

        # Perfect calibration line
        perf = DashedLine(iax.c2p(2.3, 2.3), iax.c2p(3.5, 3.5), color=LIGHT, stroke_width=0.8, dash_length=0.04)
        self.add(perf)

        # Scatter dots (predicted vs actual)
        np.random.seed(42)
        for _ in range(30):
            actual = np.random.uniform(2.5, 3.3)
            error = np.random.normal(0, 0.15)
            predicted = actual + error
            predicted = np.clip(predicted, 2.3, 3.5)
            d = Dot(iax.c2p(actual, predicted), radius=0.025, color=INK, fill_opacity=0.4)
            self.add(d)
        # Late phase dots (tighter)
        for _ in range(20):
            actual = np.random.uniform(2.8, 3.1)
            error = np.random.normal(0, 0.05)
            predicted = actual + error
            d = Dot(iax.c2p(actual, predicted), radius=0.025, color=TEAL, fill_opacity=0.6)
            self.add(d)

        ixl = Text("actual", font_size=8, color=MID, font="CMU Serif")
        ixl.next_to(iax, DOWN, buff=0.08)
        iyl = Text("predicted", font_size=8, color=MID, font="CMU Serif").rotate(PI/2)
        iyl.next_to(iax, LEFT, buff=0.08)

        for v in [2.5, 3.0, 3.5]:
            tx = Text(f"{v:.1f}", font_size=7, color=MID, font="CMU Serif")
            tx.next_to(iax.c2p(v, 2.3), DOWN, buff=0.04)
            ty = Text(f"{v:.1f}", font_size=7, color=MID, font="CMU Serif")
            ty.next_to(iax.c2p(2.3, v), LEFT, buff=0.04)
            self.add(tx, ty)

        il1 = Text("early (wide)", font_size=7, color=MID, font="CMU Serif")
        il1.move_to(RIGHT * (inset_x + 0.6) + UP * (inset_y - 1.05))
        il2 = Text("late (tight)", font_size=7, color=TEAL, font="CMU Serif")
        il2.move_to(RIGHT * (inset_x + 0.6) + UP * (inset_y - 1.25))

        self.add(iax, ixl, iyl, il1, il2)


class TransferDiagram(Scene):
    """Clean flowchart: three agents → transfer mechanism → result, with details below"""
    def construct(self):
        title = Text("Knowledge Transfer: What C Learned, What Changed, What Held",
            font_size=24, color=INK, font="CMU Serif", weight=BOLD)
        title.to_edge(UP, buff=0.35)
        self.add(title)

        # === TOP HALF: Three-row flow (compact) ===
        col_x = [-4.2, 0, 4.2]
        col_headers = ["Round 1 State", "Transfer Mechanism", "Round 2 Result"]
        header_y = 2.65
        for x, h in zip(col_x, col_headers):
            ht = Text(h, font_size=11, color=MID, font="CMU Serif", weight=BOLD)
            ht.move_to(RIGHT * x + UP * header_y)
            rule = Line(RIGHT * (x - 1.9) + UP * (header_y - 0.13),
                        RIGHT * (x + 1.9) + UP * (header_y - 0.13),
                        color=LIGHT, stroke_width=0.5)
            self.add(ht, rule)

        agents = [
            ("A", "Best config only\nNo memory", "Random search\nfrom scratch",
             "2.719", "15.8%", MID, 0.06),
            ("B", "Result history\n63 past experiments", "LLM re-analyzes\ntakes 7 tries",
             "2.441", "19.5%", BLUE, 0.06),
            ("C", "12 tested rules\nMAE=0.101\n19 principles",
             "8 held, 4 updated\nadapts in 1 try",
             "2.379", "21.5%", INK, 0.10),
        ]

        row_ys = [1.85, 1.05, 0.15]
        for i, (name, r1, transfer, score, improve, color, box_op) in enumerate(agents):
            y = row_ys[i]
            box_h = 0.65

            contents = [f"{name}:  {r1}", transfer, f"{score}\n{improve} better"]
            for j, (content, cx) in enumerate(zip(contents, col_x)):
                fill_op = box_op if j < 2 else box_op * 1.8
                stroke_op = 0.25 if i < 2 else 0.5
                box = RoundedRectangle(width=3.5, height=box_h, corner_radius=0.08,
                    fill_color=color, fill_opacity=fill_op,
                    stroke_color=color, stroke_width=0.7, stroke_opacity=stroke_op)
                box.move_to(RIGHT * cx + UP * y)

                fs = 9
                w = BOLD if j == 2 else NORMAL
                txt = Text(content, font_size=fs, color=INK, font="CMU Serif",
                    weight=w, line_spacing=0.7)
                txt.move_to(box)
                self.add(box, txt)

            for j in range(2):
                a = Arrow(
                    RIGHT * (col_x[j] + 1.8) + UP * y,
                    RIGHT * (col_x[j+1] - 1.8) + UP * y,
                    color=color, stroke_width=1.0, stroke_opacity=0.4,
                    buff=0.05, max_tip_length_to_length_ratio=0.12)
                self.add(a)

        # === SEPARATOR ===
        sep_y = -0.4
        sep = Line(LEFT * 6.5 + UP * sep_y, RIGHT * 6.5 + UP * sep_y,
            color=LIGHT, stroke_width=0.6)
        self.add(sep)

        # === BOTTOM LEFT: Parameter shift table ===
        table_cx = -3.2
        table_top = -0.7
        shift_title = Text("Parameter Shifts (4 rules updated)",
            font_size=11, color=INK, font="CMU Serif", weight=BOLD)
        shift_title.move_to(RIGHT * table_cx + UP * table_top)
        self.add(shift_title)

        shifts = [
            ("Parameter", "2 min", "10 min", "Why"),
            ("Batch size", "4", "8", "Fewer steps, smoother LR"),
            ("Embed dim", "128", "192", "More capacity utilized"),
            ("Heads", "2", "4", "Richer attention patterns"),
            ("Beta2", ".999", ".99", "Faster momentum needed"),
        ]

        col_offsets = [-2.2, -1.0, -0.1, 1.4]
        for ri, row in enumerate(shifts):
            y = table_top - 0.35 - ri * 0.30
            bg = Rectangle(width=5.6, height=0.28,
                fill_color=FAINT if ri % 2 == 0 else WHITE,
                fill_opacity=1, stroke_width=0)
            bg.move_to(RIGHT * table_cx + UP * y)
            self.add(bg)

            for ci, (cell, cx) in enumerate(zip(row, col_offsets)):
                if ri == 0:
                    cc = MID
                    ct = Text(cell, font_size=8, color=cc, font="CMU Serif", weight=BOLD)
                else:
                    cc = MID if ci == 0 else (ACCENT if ci == 1 else (TEAL if ci == 2 else MID))
                    w = BOLD if ci in (1, 2) else NORMAL
                    ct = Text(cell, font_size=8, color=cc, font="CMU Serif", weight=w)
                ct.move_to(RIGHT * (table_cx + cx) + UP * y)
                self.add(ct)

        # === BOTTOM RIGHT: Belief audit ===
        audit_cx = 3.6
        audit_title = Text("Belief Audit (12 → 16 principles)",
            font_size=11, color=INK, font="CMU Serif", weight=BOLD)
        audit_title.move_to(RIGHT * audit_cx + UP * table_top)
        self.add(audit_title)

        audit_data = [
            ("8 rules held unchanged", "✓", TEAL),
            ("4 rules correctly revised", "↻", WARM),
            ("4 new rules discovered", "+", BLUE),
            ("0 rules incorrectly kept", "—", MID),
        ]

        for i, (desc, icon, color) in enumerate(audit_data):
            y = table_top - 0.40 - i * 0.38
            ic = Text(icon, font_size=14, color=color, font="CMU Serif", weight=BOLD)
            ic.move_to(RIGHT * (audit_cx - 1.8) + UP * y)
            dt = Text(desc, font_size=9, color=INK, font="CMU Serif")
            dt.next_to(ic, RIGHT, buff=0.12)
            self.add(ic, dt)

        # === BOTTOM CALLOUT ===
        callout = Text(
            "C's first try (2.456) already beat A's best (2.719) — transfer, not search",
            font_size=11, color=ACCENT, font="CMU Serif", weight=BOLD)
        callout.to_edge(DOWN, buff=0.25)
        self.add(callout)


class StudentAgentParallel(Scene):
    """Dense: study cards + mechanism diagram + timeline"""
    def construct(self):
        title = Text("Convergent Evidence: Human Learning ≈ Agent Learning", font_size=24, color=INK, font="CMU Serif", weight=BOLD)
        title.to_edge(UP, buff=0.35)
        self.add(title)

        # === TWO STUDY CARDS ===
        for side, color, header, venue, design, n_val, finding, mechanism, fix, cost, x_pos in [
            ("left", BLUE, "Barcaui 2025",
             "Social Sci & Humanities Open", "Randomized controlled trial",
             "n = 120 students", "19% worse on later tests",
             "Answers without\nproductive struggle",
             "Try first, then check", "$0 extra", -3.5),
            ("right", TEAL, "Bonilla 2026",
             "Preprint (this work)", "3 agents × 84 experiments",
             "n = 296 experiments", "12.5% worse after shift",
             "Results without\npredictive understanding",
             "Guess first, then check", "$0 extra (seconds/guess)", 3.5),
        ]:
            card = RoundedRectangle(width=5.5, height=5.8, corner_radius=0.12,
                fill_color=FAINT, fill_opacity=1, stroke_color=LIGHT, stroke_width=0.6)
            card.move_to(RIGHT * x_pos + DOWN * 0.15)
            self.add(card)

            # Header
            h = Text(header, font_size=16, color=color, font="CMU Serif", weight=BOLD)
            h.move_to(RIGHT * x_pos + UP * 2.5)
            v = Text(venue, font_size=8, color=MID, font="CMU Serif", slant=ITALIC)
            v.next_to(h, DOWN, buff=0.04)
            self.add(h, v)

            # Design
            d = Text(design, font_size=9, color=MID, font="CMU Serif")
            d.next_to(v, DOWN, buff=0.08)
            n = Text(n_val, font_size=10, color=INK, font="CMU Serif", weight=BOLD)
            n.next_to(d, DOWN, buff=0.04)
            self.add(d, n)

            # Big finding
            pct_str = finding.split("%")[0] + "%"
            big = Text(pct_str, font_size=36, color=color, font="CMU Serif", weight=BOLD)
            big.move_to(RIGHT * x_pos + UP * 0.6)
            rest_str = finding.split("% ")[1] if "% " in finding else ""
            rest = Text(rest_str, font_size=11, color=INK, font="CMU Serif")
            rest.next_to(big, DOWN, buff=0.04)
            self.add(big, rest)

            # Mechanism
            mech_h = Text("Mechanism:", font_size=9, color=MID, font="CMU Serif", weight=BOLD)
            mech_h.move_to(RIGHT * x_pos + DOWN * 0.4)
            mech = Text(mechanism, font_size=10, color=INK, font="CMU Serif")
            mech.next_to(mech_h, DOWN, buff=0.06)
            self.add(mech_h, mech)

            # Fix
            fix_box = RoundedRectangle(width=4.5, height=0.4, corner_radius=0.06,
                fill_color=color, fill_opacity=0.08, stroke_width=0)
            fix_box.move_to(RIGHT * x_pos + DOWN * 1.5)
            fix_t = Text(fix, font_size=11, color=color, font="CMU Serif", weight=BOLD)
            fix_t.move_to(fix_box)
            self.add(fix_box, fix_t)

            # Cost
            cost_t = Text(f"Cost: {cost}", font_size=10, color=INK, font="CMU Serif", weight=BOLD)
            cost_t.move_to(RIGHT * x_pos + DOWN * 2.0)
            self.add(cost_t)

            # Citation count / impact
            if side == "left":
                cite = Text("Barcaui (2025) · RCT · n=120 · p<.001", font_size=7, color=MID, font="CMU Serif")
            else:
                cite = Text("Pre-registered · 296 exp · code+logs public", font_size=7, color=MID, font="CMU Serif")
            cite.move_to(RIGHT * x_pos + DOWN * 2.5)
            self.add(cite)

        # Center connector
        eq = Text("=", font_size=36, color=INK, font="CMU Serif", weight=BOLD)
        eq.move_to(DOWN * 0.15)
        self.add(eq)

        # Shared mechanism label
        shared = RoundedRectangle(width=1.8, height=0.9, corner_radius=0.08,
            fill_color="#FFF8E1", fill_opacity=1, stroke_color="#FFD54F", stroke_width=0.6)
        shared.move_to(DOWN * 2.6)
        shared_t = Text("Desirable\ndifficulty\n(Bjork, 1994)", font_size=9, color=WARM, font="CMU Serif")
        shared_t.move_to(shared)
        self.add(shared, shared_t)

        # Bottom
        bottom = Text("Same pattern. Same fix. Same theoretical root. Same cost: nothing.", font_size=14, color=INK, font="CMU Serif", weight=BOLD)
        bottom.to_edge(DOWN, buff=0.25)
        self.add(bottom)


class KarpathyMirror(Scene):
    """Mirror of Karpathy's autoresearch progress chart — our Agent C data in his style"""
    def construct(self):
        # Match Karp's title format exactly
        title = Text("Prediction-Error Progress: 84 Experiments, 24 Kept Improvements", font_size=22, color=INK, font="CMU Serif", weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.add(title)

        sub = Text("cf. Karpathy (2026) Autoresearch: 83 experiments, 15 kept — same format, different mechanism",
            font_size=10, color=MID, font="CMU Serif", slant=ITALIC)
        sub.next_to(title, DOWN, buff=0.06)
        self.add(sub)

        ax = Axes(
            x_range=[0, 85, 10], y_range=[2.3, 3.5, 0.1],
            x_length=11, y_length=5.0,
            axis_config={"color": LIGHT, "include_numbers": False, "stroke_width": 0.8},
            tips=False,
        ).shift(DOWN * 0.25)

        # Grid
        for v in np.arange(2.4, 3.5, 0.1):
            g = DashedLine(ax.c2p(0, v), ax.c2p(85, v), color="#F3F3F3", stroke_width=0.3, dash_length=0.03)
            self.add(g)

        for v in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
            t = Text(str(v), font_size=10, color=MID, font="CMU Serif")
            t.next_to(ax.c2p(v, 2.3), DOWN, buff=0.08)
            self.add(t)
        for v in np.arange(2.4, 3.5, 0.1):
            t = Text(f"{v:.1f}", font_size=9, color=MID, font="CMU Serif")
            t.next_to(ax.c2p(0, v), LEFT, buff=0.08)
            self.add(t)

        xl = Text("Experiment #", font_size=12, color=INK, font="CMU Serif")
        xl.next_to(ax, DOWN, buff=0.3)
        yl = Text("Validation Loss (lower is better)", font_size=12, color=INK, font="CMU Serif").rotate(PI/2)
        yl.next_to(ax, LEFT, buff=0.5)
        self.add(xl, yl)

        # Simulate C's 64 Phase 1 + 20 Phase 2 experiments
        # Kept experiments (improvements) with annotations
        np.random.seed(123)

        kept_exps = [
            (1, 3.45, "baseline"),
            (3, 3.35, "batch 4, LR 1e-3"),
            (5, 3.28, "add warmup 20"),
            (8, 3.20, "embed 128"),
            (11, 3.18, "weight decay 0.01"),
            (15, 3.15, "5 layers"),
            (19, 3.12, "2 heads"),
            (22, 3.10, "dropout 0.05"),
            (27, 3.09, "LR 2e-3 + warmup 50"),
            (31, 3.08, "beta2 0.999"),
            (36, 3.07, "grad accum 2"),
            (40, 3.06, "tune warmup 30"),
            (45, 3.05, "batch 4 confirmed"),
            (50, 3.04, "embed 128 optimal"),
            (55, 3.035, "fine-tune decay"),
            (58, 3.032, "LR schedule cosine"),
            (61, 3.030, "final Phase 1"),
            # Phase 2 (after exp 64)
            (65, 2.456, "batch 8 (5x compute)"),
            (68, 2.42, "embed 192"),
            (70, 2.41, "4 heads"),
            (72, 2.40, "beta2 0.99"),
            (74, 2.39, "warmup 50"),
            (77, 2.385, "tune LR 2e-3"),
            (80, 2.379, "final config"),
        ]

        # Generate discarded experiments (scattered higher)
        discarded = []
        for exp_num in range(0, 84):
            is_kept = any(k[0] == exp_num for k in kept_exps)
            if not is_kept:
                if exp_num <= 64:
                    base = 3.45 - (exp_num / 64) * 0.4
                    noise = np.random.uniform(0, 0.15)
                    discarded.append((exp_num, base + noise))
                else:
                    base = 2.5 - ((exp_num - 64) / 20) * 0.1
                    noise = np.random.uniform(0, 0.12)
                    discarded.append((exp_num, base + noise))

        # Plot discarded (grey dots — match Karpathy's style)
        for x, y in discarded:
            y_clipped = min(y, 3.48)
            d = Dot(ax.c2p(x, y_clipped), radius=0.04, color=MID, fill_opacity=0.25)
            self.add(d)

        # Plot kept (green dots — match Karpathy's green)
        KARP_GREEN = "#2ca02c"
        kept_dots = []
        for x, y, label in kept_exps:
            d = Dot(ax.c2p(x, y), radius=0.06, color=KARP_GREEN, fill_opacity=0.9)
            kept_dots.append((x, y))
            self.add(d)

        # Running best line (staircase — Karpathy's signature)
        staircase_pts = []
        for i, (x, y, _) in enumerate(kept_exps):
            if i > 0:
                staircase_pts.append(ax.c2p(x, kept_exps[i-1][1]))
            staircase_pts.append(ax.c2p(x, y))
        # Extend to end
        staircase_pts.append(ax.c2p(84, kept_exps[-1][1]))

        staircase = VMobject(color=KARP_GREEN, stroke_width=2, stroke_opacity=0.7)
        staircase.set_points_as_corners(staircase_pts)
        self.add(staircase)

        # Annotations on key kept experiments (angled, like Karpathy's)
        annotations = [
            (1, 3.45, "baseline", -35),
            (5, 3.28, "add warmup 20", -30),
            (11, 3.18, "weight decay 0.01", -35),
            (19, 3.12, "2 heads", -25),
            (27, 3.09, "LR 2e-3 + warmup 50", -30),
            (40, 3.06, "tune warmup", -25),
            (55, 3.035, "fine-tune decay", -30),
            (61, 3.030, "final Phase 1 best", -25),
            (65, 2.456, "batch 8 (5x compute!)", -35),
            (70, 2.41, "4 heads + embed 192", -30),
            (74, 2.39, "beta2 .999→.99", -25),
            (80, 2.379, "final: 2.379", -20),
        ]

        for x, y, label, angle in annotations:
            t = Text(label, font_size=7, color=KARP_GREEN, font="CMU Serif")
            t.rotate(angle * PI / 180)
            t.next_to(ax.c2p(x, y), UP + LEFT, buff=0.08)
            self.add(t)

        # Phase divider
        phase_line = DashedLine(ax.c2p(64, 2.3), ax.c2p(64, 3.5), color=ACCENT, stroke_width=1.2, dash_length=0.05)
        phase_label = Text("5x compute shift", font_size=10, color=ACCENT, font="CMU Serif", weight=BOLD)
        phase_label.rotate(PI/2).move_to(ax.c2p(64.5, 2.9))
        self.add(phase_line, phase_label)

        # Legend (match Karpathy's)
        leg_x, leg_y = 5.0, 2.8
        for color_l, op_l, label_l, dy in [
            (MID, 0.25, "Discarded", 0),
            (KARP_GREEN, 0.9, "Kept", -0.3),
            (KARP_GREEN, 0.7, "Running best", -0.6),
        ]:
            if "Running" in label_l:
                l = Line(RIGHT * leg_x + UP * (leg_y + dy), RIGHT * (leg_x + 0.4) + UP * (leg_y + dy),
                    color=color_l, stroke_width=2, stroke_opacity=op_l)
            else:
                l = Dot(RIGHT * (leg_x + 0.2) + UP * (leg_y + dy), radius=0.05, color=color_l, fill_opacity=op_l)
            lt = Text(label_l, font_size=9, color=INK, font="CMU Serif")
            lt.next_to(l, RIGHT, buff=0.1)
            self.add(l, lt)

        # Bottom callout
        callout = Text(
            "Karpathy's autoresearch: 83 exp, 15 kept, no predictions.  Ours: 84 exp, 24 kept, with predictions. Same chart. Different depth.",
            font_size=9, color=MID, font="CMU Serif")
        callout.to_edge(DOWN, buff=0.25)
        self.add(callout)

        self.add(ax)
