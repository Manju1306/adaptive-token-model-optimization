"""Graphical abstract for the KATMO manuscript. Numbers are the full-run results (Tables 1-2); the step allocation shown
is the most frequent choice per step type on GSM8K at lambda* (Figure 2). Regenerate: python make_ga.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "Liberation Sans"})  # metric-compatible with Arial (Elsevier-accepted font family)
fig = plt.figure(figsize=(13.28, 5.31), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

def box(x, y, w, h, fc="#ffffff", ec=MUTED, lw=1.2, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK2))

ax.text(2, 37.2, "KATMO: knowledge-based, step-level choice of model tier and token budget for LLM agents",
        fontsize=14.5, weight="bold", color=INK, va="center")

# 1 agent task
box(2, 9, 17, 22); ax.text(10.5, 28.8, "1  Agent task", ha="center", fontsize=11.5, weight="bold", color=INK)
ax.text(10.5, 25.4, "Five steps per task", ha="center", fontsize=9.6, color=INK2)
for i, t in enumerate(["plan", "reason", "tool", "verify", "answer"]):
    ax.text(10.5, 22.6 - i*2.3, t, ha="center", fontsize=9.6, color=INK2)
ax.text(10.5, 10.2, "GSM8K and HotpotQA", ha="center", fontsize=8.6, color=MUTED)
arrow(19.3, 20, 24.2, 20)

# 2 hybrid decision layer
box(24.5, 9, 25, 22); ax.text(37, 28.8, "2  Hybrid decision layer", ha="center", fontsize=11.5, weight="bold", color=INK)
box(26.2, 17.2, 21.6, 9.0, fc="#eaf2fc", ec=BLUE)
ax.text(37, 24.2, "Knowledge base", ha="center", fontsize=10.4, weight="bold", color=BLUE)
ax.text(37, 21.2, "6 logged rules set the\nadmissible tier and budget", ha="center", fontsize=9.2, color=INK2, va="center", linespacing=1.3)
box(26.2, 10.3, 21.6, 5.9, fc="#fdeee8", ec=ORANGE)
ax.text(37, 14.3, "Learned estimator", ha="center", fontsize=10.4, weight="bold", color=ORANGE)
ax.text(37, 11.9, "task success per configuration", ha="center", fontsize=9.0, color=INK2)
arrow(49.8, 20, 54.7, 20)

# 3 per-step choice
box(55, 9, 24, 22); ax.text(67, 28.8, "3  Choose per step", ha="center", fontsize=11.5, weight="bold", color=INK)
ax.text(67, 25.6, "minimise  cost − λ × P(success)", ha="center", fontsize=9.6, color=INK, style="italic")
choice = [("plan", "32B @128"), ("reason", "32B @1024"), ("tool", "32B @1024"), ("verify", "32B @1024"), ("answer", "32B @128")]
for i, (s, c) in enumerate(choice):
    y = 22.0 - i*2.3
    ax.text(58.2, y, s, ha="left", va="center", fontsize=9.2, color=INK2)
    ax.add_patch(FancyBboxPatch((66.4, y-1.0), 11.2, 2.0, boxstyle="round,pad=0,rounding_size=0.5",
                                fc="#cde2fb" if "128" in c else BLUE, ec="none"))
    ax.text(72.0, y, c, ha="center", va="center", fontsize=8.8, color=INK if "128" in c else "white")
ax.text(67, 10.0, "most frequent choice on GSM8K at λ*", ha="center", fontsize=8.0, color=MUTED)
arrow(79.3, 20, 83.9, 20)

# 4 result
box(84.2, 9, 14, 22); ax.text(91.2, 28.8, "4  Result", ha="center", fontsize=11.5, weight="bold", color=INK)
ax.text(91.2, 25.0, "86.5%", ha="center", fontsize=19, weight="bold", color=AQUA)
ax.text(91.2, 22.6, "of the cost of the\n32B reference", ha="center", va="center", fontsize=8.6, color=INK2, linespacing=1.25)
ax.text(91.2, 18.6, "80.0% vs 80.7%", ha="center", fontsize=11.5, weight="bold", color=INK)
ax.text(91.2, 16.4, "task success\n(p = 0.80)", ha="center", va="center", fontsize=8.6, color=INK2, linespacing=1.25)
ax.text(91.2, 11.9, "300 test tasks,\n3 open-weight models", ha="center", va="center", fontsize=8.0, color=MUTED, linespacing=1.25)

# bottom strip: honest caveat
ax.add_patch(Rectangle((2, 2.2), 96.2, 4.2, fc="#f6f5f2", ec="none"))
ax.text(50, 4.3, "Modest gain: a static 32B model with a 128-token cap costs 82.0% at 79.0% success; the saving appears mainly on HotpotQA",
        ha="center", va="center", fontsize=9.6, color=INK2)
fig.savefig("graphical_abstract.png", dpi=300, facecolor="white")
fig.savefig("graphical_abstract.pdf", facecolor="white")
from PIL import Image
im = Image.open("graphical_abstract.png"); print(im.size)
