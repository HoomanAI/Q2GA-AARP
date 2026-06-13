"""Generate detailed Q2GA component flowcharts (PNG, white background) for
the methodology document.

Figures produced (in figures/flowcharts/):
  - q2ga_master_flow.png        : how the detailed diagrams relate to one
                                   generation of the Q2GA loop
  - q2ga_chromosome_collapse.png: quantum chromosome representation &
                                   population collapse (Make P(t))
  - q2ga_observation_state.png  : RL observation state O_t,i = (g, h, f)
  - q2ga_action_selection.png   : epsilon-greedy action selection & Q-table
  - q2ga_rotation_gate.png      : adaptive quantum rotation gate U(t)
  - q2ga_reward_update.png      : reward calculation & Q-table update
  - qga_vs_q2ga_rotation.png    : QGA (fixed-angle) vs Q2GA (adaptive) gate
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "flowcharts")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
})

WHITE = "#FFFFFF"
BLUE = "#A9C4EB"
ORANGE = "#FFCC80"
GREY = "#E6E6E6"


def new_axes(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    return fig, ax


def box(ax, cx, cy, w, h, text, fc=WHITE, fontsize=9.5, boxstyle="round,pad=0.02,rounding_size=0.08"):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=boxstyle,
                        fc=fc, ec="black", lw=1.2)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)
    return dict(top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                 left=(cx - w / 2, cy), right=(cx + w / 2, cy), c=(cx, cy))


def ellipse(ax, cx, cy, w, h, text, fc=BLUE, fontsize=10):
    e = Ellipse((cx, cy), w, h, fc=fc, ec="black", lw=1.2)
    ax.add_patch(e)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)
    return dict(top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                 left=(cx - w / 2, cy), right=(cx + w / 2, cy), c=(cx, cy))


def diamond(ax, cx, cy, w, h, text, fc=WHITE, fontsize=9):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    p = Polygon(pts, closed=True, fc=fc, ec="black", lw=1.2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)
    return dict(top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                 left=(cx - w / 2, cy), right=(cx + w / 2, cy), c=(cx, cy))


def arrow(ax, p1, p2, text=None, fontsize=8.5, ha="left", offset=(0.1, 0.0)):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, color="black", lw=1.2)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + offset[0], my + offset[1], text, fontsize=fontsize, ha=ha, va="center")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 0. Master flow: how the detailed diagrams relate to one generation
# ===========================================================================
def fig_master_flow():
    fig, ax = new_axes((7, 9), (0, 10), (0, 14))

    n1 = ellipse(ax, 5, 13, 4, 1, "Generation t\n(start)")
    n2 = box(ax, 5, 11.2, 7, 1.4, "Stage 1 -- Collapse Q(t) into P(t),\nevaluate fitness  (see Fig. 2)", fc=GREY)
    n3 = box(ax, 5, 9.2, 7, 1.4, "Stage 2 -- Build RL observation state\nO_t,i = (g, h, f) for each individual\n(see Fig. 3)", fc=ORANGE)
    n4 = box(ax, 5, 7.2, 7, 1.4, "Stage 3 -- epsilon-greedy action\nselection from Q-table  (see Fig. 4)", fc=ORANGE)
    n5 = box(ax, 5, 5.2, 7, 1.4, "Stage 4 -- Apply adaptive rotation gate\nU(t) to selected genes/qubits\n(see Fig. 5)", fc=GREY)
    n6 = box(ax, 5, 3.2, 7, 1.4, "Stage 5 -- Re-sample, evaluate,\ncompute reward, update Q-table and\nbest-so-far solution  (see Fig. 6)", fc=ORANGE)
    n7 = ellipse(ax, 5, 1.2, 4, 1, "t = t + 1\n(loop)")

    for a, b_ in [(n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, n6), (n6, n7)]:
        arrow(ax, a["bottom"], b_["top"])

    arrow(ax, (1.3, 1.2), (1.3, 11.2))
    ax.add_patch(FancyArrowPatch((1.3, 11.2), n2["left"], arrowstyle="-|>", mutation_scale=14, color="black", lw=1.2))
    ax.text(0.9, 6.2, "repeat for t = 0 .. T-1", fontsize=9, rotation=90, ha="center", va="center")
    arrow(ax, n7["left"], (1.3, 1.2))

    ax.set_title("Q2GA: relation between detailed component diagrams and the\n"
                  "main RL-assisted rotation loop (one generation)", fontsize=12)
    save(fig, "q2ga_master_flow")


# ===========================================================================
# 1. Quantum chromosome representation and population collapse (Make P(t))
# ===========================================================================
def fig_chromosome_collapse():
    fig, ax = new_axes((7, 9), (0, 10), (0, 13.5))

    n1 = ellipse(ax, 5, 12.5, 6, 1, "Q-chromosome i in Q(t)\n(alpha_{i,j}, beta_{i,j}), j=1..n_qubits,\nalpha^2+beta^2=1")
    n2 = box(ax, 5, 10.6, 7.5, 1.4, "For each qubit j:\nsample bit  x_{i,j} = 1 with probability beta_{i,j}^2,\nelse x_{i,j} = 0   (quantum measurement)")
    n3 = box(ax, 5, 8.7, 6.5, 1.0, "Collapsed bit string\nb_i = (x_{i,1}, ..., x_{i,n_qubits})")
    n4 = box(ax, 5, 6.8, 7.5, 1.4, "Decode: bits_to_real(b_i)\n-> real-valued AARP chromosome\n(vehicle-route / sequencing genes in [0,1])")
    n5 = box(ax, 5, 4.9, 7.5, 1.4, "Decode chromosome into vehicle routes\n(decode()) and simulate dispatch\n(simulate())")
    n6 = box(ax, 5, 3.0, 7.5, 1.4, "Evaluate fitness(b_i) = w1*C1+w2*C2+w3*C3\n+ penalty1+penalty2+infeasibility\n-> stored as P(t) row i")
    n7 = box(ax, 5, 1.1, 6.5, 1.0, "If fitness(b_i) < best_fitness so far:\nupdate best_bits, best_fitness", fc=ORANGE)

    for a, b_ in [(n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, n6), (n6, n7)]:
        arrow(ax, a["bottom"], b_["top"])

    ax.text(8.7, 10.6, "repeated for\ni = 1..pop_size", fontsize=8.5, ha="center", style="italic")
    ax.set_title("Q2GA Stage 1: Quantum chromosome representation and\n"
                  "population collapse (\"Make P(t) by observing Q(t)\")", fontsize=12)
    save(fig, "q2ga_chromosome_collapse")


# ===========================================================================
# 2. Observation state O_t,i = (g, h, f)
# ===========================================================================
def fig_observation_state():
    fig, ax = new_axes((9, 9), (0, 13), (0, 12.5))

    title_y = 12
    ax.text(6.5, title_y, "Q2GA Stage 2: RL observation state $O_{t,i} = (g, h, f)$", ha="center", fontsize=12)

    # g branch
    g1 = box(ax, 2.2, 10.2, 4.0, 1.2, "Population mean fitness\nat generation t:\nmean_fit(t)", fc=ORANGE)
    g2 = box(ax, 2.2, 8.4, 4.0, 1.2, "Compare mean_fit(t) to the\naverage of the last\nGEN_WINDOW=5 generations", fc=ORANGE)
    g3 = diamond(ax, 2.2, 6.3, 4.2, 1.6, "mean_fit(t) <\nrunning average?", fc=ORANGE)
    g4a = box(ax, 1.0, 4.3, 1.8, 0.9, "g = 1\n(improving)", fc=ORANGE)
    g4b = box(ax, 3.4, 4.3, 1.8, 0.9, "g = 0\n(not improving)", fc=ORANGE)

    arrow(ax, g1["bottom"], g2["top"])
    arrow(ax, g2["bottom"], g3["top"])
    arrow(ax, g3["bottom"], g4a["top"], text="yes", offset=(-0.35, 0.1))
    arrow(ax, (g3["c"][0] + 0.6, g3["c"][1] - 0.7), g4b["top"], text="no", offset=(0.05, 0.1))

    # h branch
    h1 = box(ax, 6.5, 10.2, 4.2, 1.2, "Hamming distance of\nchromosome i to every other\nindividual in P(t)")
    h2 = box(ax, 6.5, 8.4, 4.2, 1.2, "Average Hamming distance\nfor individual i:\nham_i = mean_{k!=i} d(b_i,b_k)")
    h3 = box(ax, 6.5, 6.3, 4.2, 1.4, "Assign ham_i to a quartile bin\nover {ham_1,...,ham_pop}\nh in {0,1,2,3}  (HAM_BINS=4)")

    arrow(ax, h1["bottom"], h2["top"])
    arrow(ax, h2["bottom"], h3["top"])

    # f branch
    f1 = box(ax, 11.0, 10.2, 4.0, 1.2, "Fitness of\nchromosome i:\nfitness_i")
    f2 = box(ax, 11.0, 8.4, 4.0, 1.4, "Assign fitness_i to a quartile\nbin over {fitness_1,...,fitness_pop}\nf in {0,1,2,3}  (FIT_BINS=4)")

    arrow(ax, f1["bottom"], f2["top"])

    # combine
    combo = box(ax, 6.5, 1.8, 9.0, 1.6,
                 "$O_{t,i} = (g, h, f)$ -- discrete state\n"
                 "g in {0,1}, h in {0,1,2,3}, f in {0,1,2,3}\n"
                 "=> 2 x 4 x 4 = 32 possible states, used to index Q(O,a)")

    arrow(ax, (g4a["c"][0], g4a["bottom"][1]), (combo["left"][0] + 1.0, combo["top"][1]))
    arrow(ax, (g4b["c"][0], g4b["bottom"][1]), (combo["left"][0] + 1.0, combo["top"][1]))
    arrow(ax, h3["bottom"], (combo["c"][0], combo["top"][1]))
    arrow(ax, f2["bottom"], (combo["right"][0] - 1.0, combo["top"][1]))

    save(fig, "q2ga_observation_state")


# ===========================================================================
# 3. Action selection (epsilon-greedy) and Q-table lookup
# ===========================================================================
def fig_action_selection():
    fig, ax = new_axes((8, 10), (0, 11), (0, 13.5))

    n1 = box(ax, 5.5, 12.5, 8.5, 1.2,
             "$O_{t,i} = (g,h,f)$ from Fig. 3, and decayed\n"
             "$\\epsilon(t) = \\epsilon_{start} + (\\epsilon_{end}-\\epsilon_{start}) \\cdot t/(T-1)$", fc=ORANGE)
    n2 = diamond(ax, 5.5, 10.4, 4.5, 1.7, "rand() < epsilon(t) ?", fc=ORANGE)
    n3a = box(ax, 2.3, 8.2, 4.0, 1.2, "Exploration:\nchoose random action\na in {1,...,5}", fc=ORANGE)
    n3b = box(ax, 8.7, 8.2, 4.0, 1.2, "Exploitation:\na = argmax_a  Q(O_{t,i}, a)", fc=ORANGE)
    n4 = box(ax, 5.5, 6.0, 8.5, 1.6,
             "Action a in {1,...,5} sets:\n"
             "  n_rotate = max(1, round(a/5 * n_genes))   (genes to perturb)\n"
             "  multiplier = a   (rotation magnitude scale, 1x..5x base angle)")
    n5 = box(ax, 5.5, 3.8, 8.5, 1.2,
             "Randomly select n_rotate genes (without\nreplacement) from the n_genes real-valued genes")
    n6 = box(ax, 5.5, 1.6, 8.5, 1.4,
             "Pass {selected genes, multiplier, direction rule}\nto the adaptive rotation gate U(t)  (Fig. 5)", fc=GREY)

    arrow(ax, n1["bottom"], n2["top"])
    arrow(ax, (n2["c"][0] - 0.8, n2["c"][1] - 0.7), n3a["top"], text="yes", offset=(-0.3, 0.1))
    arrow(ax, (n2["c"][0] + 0.8, n2["c"][1] - 0.7), n3b["top"], text="no", offset=(0.05, 0.1))
    arrow(ax, n3a["bottom"], (n4["left"][0] + 1.5, n4["top"][1]))
    arrow(ax, n3b["bottom"], (n4["right"][0] - 1.5, n4["top"][1]))
    arrow(ax, n4["bottom"], n5["top"])
    arrow(ax, n5["bottom"], n6["top"])

    ax.set_title("Q2GA Stage 3: epsilon-greedy action selection\nfrom the Q-table", fontsize=12)
    save(fig, "q2ga_action_selection")


# ===========================================================================
# 4. Adaptive rotation gate U(t)
# ===========================================================================
def fig_rotation_gate():
    fig, ax = new_axes((8, 10), (0, 11), (0, 14.5))

    n1 = box(ax, 5.5, 13.5, 8.5, 1.2,
             "Inputs (per individual i, from Fig. 4):\n"
             "selected genes -> qubit indices, multiplier a, best_bits")
    n2 = box(ax, 5.5, 11.6, 8.5, 1.2,
             "For each selected qubit j:\ncompute sign(alpha_{i,j} * beta_{i,j})\n(if zero, choose +-1 at random)")
    n3 = diamond(ax, 5.5, 9.2, 5.0, 1.9, "Compare collapsed bit\nx_{i,j} to best-so-far\nbit  bb_j")
    n3a = box(ax, 1.6, 9.2, 2.6, 1.4, "x=0, bb=1:\nbase_dir = sign", fc=GREY)
    n3b = box(ax, 9.4, 9.2, 2.6, 1.4, "x=1, bb=0:\nbase_dir = -sign", fc=GREY)
    n3c = box(ax, 5.5, 6.8, 3.4, 1.0, "x = bb:\nbase_dir = 0")

    arrow(ax, n1["bottom"], n2["top"])
    arrow(ax, n2["bottom"], n3["top"])
    arrow(ax, n3["left"], n3a["right"], text="case 0->1", offset=(0, 0.25))
    arrow(ax, n3["right"], n3b["left"], text="case 1->0", offset=(0, 0.25))
    arrow(ax, n3["bottom"], n3c["top"], text="x=bb", offset=(0.3, 0.1))

    n4 = box(ax, 5.5, 4.9, 8.5, 1.4,
             "With probability random_dir_frac (default 0.02):\n"
             "replace direction with a uniformly random +-1\n"
             "(stochastic exploration of rotation direction)", fc=ORANGE)
    n5 = box(ax, 5.5, 3.0, 8.5, 1.2,
             "theta_{i,j} = base_delta_theta * multiplier * direction\n"
             "(base_delta_theta = 0.05*pi by default)")
    n6 = box(ax, 5.5, 1.1, 8.5, 1.4,
             "Apply rotation matrix R(theta_{i,j}) to (alpha_{i,j}, beta_{i,j}),\n"
             "renormalise so alpha'^2+beta'^2=1\n"
             "=> updated qubits form Q(t+1) row i")

    arrow(ax, n3a["bottom"], (n4["left"][0] + 1.5, n4["top"][1]))
    arrow(ax, n3b["bottom"], (n4["right"][0] - 1.5, n4["top"][1]))
    arrow(ax, n3c["bottom"], (n4["c"][0], n4["top"][1]))
    arrow(ax, n4["bottom"], n5["top"])
    arrow(ax, n5["bottom"], n6["top"])

    ax.set_title("Q2GA Stage 4: Adaptive quantum rotation gate U(t)\n"
                  "(action / operator applied to Q(t))", fontsize=12)
    save(fig, "q2ga_rotation_gate")


# ===========================================================================
# 5. Reward calculation and Q-table update
# ===========================================================================
def fig_reward_update():
    fig, ax = new_axes((8, 10), (0, 11), (0, 13.5))

    n1 = box(ax, 5.5, 12.5, 8.5, 1.2,
             "Updated qubits (alpha_i', beta_i') from Fig. 5\n"
             "(individual i's row of Q(t+1))")
    n2 = box(ax, 5.5, 10.6, 8.5, 1.2,
             "Sample new collapsed bit string b_i' from\n(alpha_i', beta_i'); decode -> new_chromosome_i")
    n3 = box(ax, 5.5, 8.7, 8.5, 1.0,
             "Evaluate new_fitness_i = fitness(b_i')")
    n4 = box(ax, 5.5, 6.8, 8.5, 1.2,
             "reward = (old_fitness_i - new_fitness_i) * 100\n/ |old_fitness_i|", fc=ORANGE)
    n5 = box(ax, 5.5, 4.9, 8.5, 1.4,
             "counts(O_{t,i}, a) += 1\n"
             "Q(O_{t,i}, a) <- Q(O_{t,i}, a) + lr * (reward - Q(O_{t,i}, a))\n"
             "(lr = learning_rate, default 0.5)", fc=ORANGE)
    n6 = diamond(ax, 5.5, 2.7, 5.5, 1.9, "new_fitness_i <\nbest_fitness so far?")
    n7a = box(ax, 1.6, 0.9, 2.8, 1.0, "Update\nbest_bits,\nbest_fitness", fc=ORANGE)
    n7b = box(ax, 9.4, 0.9, 2.8, 1.0, "Keep current\nbest_bits,\nbest_fitness")

    for a, b_ in [(n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, n6)]:
        arrow(ax, a["bottom"], b_["top"])
    arrow(ax, n6["left"], n7a["top"], text="yes", offset=(0, 0.25))
    arrow(ax, n6["right"], n7b["top"], text="no", offset=(0, 0.25))

    ax.text(5.5, -0.4, "Repeat for the next individual i+1; after all i,\n"
                       "save best individual of P(t) and proceed to t = t+1",
            ha="center", fontsize=9.5, style="italic")

    ax.set_title("Q2GA Stage 5: Reward calculation, Q-table update,\nand best-solution tracking", fontsize=12)
    save(fig, "q2ga_reward_update")


# ===========================================================================
# 6. QGA (fixed-angle) vs Q2GA (adaptive) rotation gate
# ===========================================================================
def fig_qga_vs_q2ga():
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.set_axis_off()

    # QGA panel
    ax = axes[0]
    a1 = box(ax, 5, 11, 8.5, 1.0, "All n_qubits qubits of every individual")
    a2 = box(ax, 5, 9.3, 8.5, 1.4, "Compare every collapsed bit x_j to\nbest-so-far bit bb_j")
    a3 = box(ax, 5, 7.4, 8.5, 1.6, "Direction: toward best genome only\n(sign(alpha*beta), case 0->1 / 1->0)\nNo random-direction term")
    a4 = box(ax, 5, 5.5, 8.5, 1.2, "theta = DELTA_THETA (fixed = 0.05*pi)\nfor every rotated qubit, every generation")
    a5 = box(ax, 5, 3.4, 8.5, 1.6, "Apply R(theta) to all qubits,\nrenormalise -> Q(t+1)\n(no learning, no per-individual choice)")
    a6 = box(ax, 5, 1.2, 8.5, 1.2, "Same fixed update rule applied\nuniformly, generation after generation", fc=GREY)
    for x, y in [(a1, a2), (a2, a3), (a3, a4), (a4, a5), (a5, a6)]:
        arrow(ax, x["bottom"], y["top"])
    ax.set_title("QGA: fixed-angle rotation gate", fontsize=12)

    # Q2GA panel
    ax = axes[1]
    b1 = box(ax, 5, 11, 8.5, 1.0, "Only n_rotate selected genes' qubits,\nchosen per individual i", fc=ORANGE)
    b2 = box(ax, 5, 9.3, 8.5, 1.4, "Compare collapsed bit x_j to\nbest-so-far bit bb_j (same as QGA)")
    b3 = box(ax, 5, 7.4, 8.5, 1.8, "Direction: mostly toward best genome,\nbut with probability random_dir_frac\nreplaced by a random +-1 (exploration)", fc=ORANGE)
    b4 = box(ax, 5, 5.3, 8.5, 1.4, "theta = base_delta_theta * multiplier * dir\nmultiplier in {1,...,5} chosen by RL agent", fc=ORANGE)
    b5 = box(ax, 5, 3.2, 8.5, 1.6, "Apply R(theta) to selected qubits only,\nrenormalise -> Q(t+1) row i")
    b6 = box(ax, 5, 1.0, 8.5, 1.4, "RL agent (Q-table) learns, from observed\nrewards, which (state -> action) choices\nimprove fitness fastest", fc=ORANGE)
    for x, y in [(b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6)]:
        arrow(ax, x["bottom"], y["top"])
    ax.set_title("Q2GA: adaptive (RL-selected) rotation gate", fontsize=12)

    fig.suptitle("Comparison: QGA's fixed-angle rotation gate vs. Q2GA's\n"
                  "adaptive, RL-controlled rotation gate", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "qga_vs_q2ga_rotation.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_master_flow()
    fig_chromosome_collapse()
    fig_observation_state()
    fig_action_selection()
    fig_rotation_gate()
    fig_reward_update()
    fig_qga_vs_q2ga()
    print("Flowcharts written to", OUT)


if __name__ == "__main__":
    main()
