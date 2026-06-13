# Q2GA: Reinforcement Learning-Assisted Quantum-Inspired Genetic Algorithm for Cyber Risk-Aware Autonomous Ambulance Allocation and Routing

This repository contains the implementation, mathematical formulation, case studies, and experimental results presented in the paper:

> **"Reinforcement Learning-Assisted Quantum-Inspired Genetic Algorithm for Cyber Risk-Aware Autonomous Ambulance Allocation and Routing"**

---<p align="center">
  <img src="figures/Q2GA Hooman Razavi.png" width="800">
</p>

## Overview

Emergency medical response systems require rapid and reliable ambulance allocation and routing to minimize response times and improve patient outcomes. The emergence of autonomous vehicles offers new opportunities for intelligent emergency response; however, the increasing integration of cyber-physical systems introduces cybersecurity risks that may disrupt communication, routing decisions, and service availability.

This project proposes a novel **Q2GA (Q-Learning Assisted Quantum-Inspired Genetic Algorithm)** framework for solving the cyber risk-aware autonomous ambulance allocation and routing problem. The framework formulates the problem as a multi-objective optimization model that simultaneously:

- Minimizes emergency response time
- Maximizes emergency coverage
- Improves allocation fairness
- Reduces energy consumption
- Increases patient survival likelihood
- Enhances resilience against cyber-induced service disruptions

---

## Key Contributions

- **Novel mathematical model** for cyber risk-aware autonomous ambulance allocation and routing.
- **Integration of reinforcement learning (Q-Learning)** with a Quantum-Inspired Genetic Algorithm.
- **Dynamic parameter tuning** of quantum rotation gates through reward-driven learning.
- **Adaptive exploration-exploitation balance** during optimization.
- **Cyber risk-aware decision-making framework** for autonomous emergency response systems.
- **Comprehensive evaluation** through realistic emergency response case studies.

---

## Repository Structure

```text
Q2GA-Cyber-EMS/
│
├── src/                  # Source code
├── data/                 # Case study datasets
├── methodology/          # Mathematical formulations
├── docs/                 # Flowcharts and supporting documents
├── figures/              # Generated figures and visualizations
├── results/              # Experimental results
├── paper/                # Manuscript and supplementary materials
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Methodology

The proposed Q2GA framework integrates **Reinforcement Learning (RL)** and a **Quantum-Inspired Genetic Algorithm (QGA)** to solve the cyber risk-aware autonomous ambulance allocation and routing problem.

### 1. Reinforcement Learning (Q-Learning)

The reinforcement learning component continuously learns from the optimization process and dynamically adjusts algorithm parameters based on observed rewards.

Key functions include:

- Adaptive parameter tuning
- Reward-driven learning
- State-action evaluation
- Dynamic optimization guidance
- Improved exploration-exploitation balance

### 2. Quantum-Inspired Genetic Algorithm (QGA)

The QGA component performs population-based optimization using quantum-inspired representations and operators.

Key features include:

- Quantum bit (Q-bit) representation
- Rotation gate updates
- Probabilistic solution generation
- Population-based evolutionary search
- Efficient exploration of large solution spaces

### 3. Q2GA Hybrid Optimization Framework

The proposed Q2GA framework combines the strengths of reinforcement learning and quantum-inspired evolutionary optimization.

The reinforcement learning agent dynamically adjusts QGA rotation gate parameters based on optimization performance, enabling:

- Faster convergence
- Enhanced solution quality
- Adaptive search behavior
- Improved robustness under dynamic conditions
- Better handling of cyber-induced disruptions

### 4. Cyber Risk-Aware Emergency Response Modeling

The optimization model explicitly incorporates cyber risk factors that may affect emergency response operations, including:

- Communication failures
- Network disruptions
- Service availability degradation
- Infrastructure vulnerabilities
- Cyber-induced routing uncertainties

---

## Case Studies

The proposed framework is evaluated using three real-world-inspired emergency response scenarios:

### Case Study 1: Ottawa, Canada

A metropolitan emergency medical response network representing urban ambulance allocation and routing challenges under varying emergency demand patterns and cyber risk conditions.

### Case Study 2: Calgary, Canada

A large-scale urban emergency response scenario focusing on ambulance allocation efficiency, response time optimization, and resilience against communication disruptions.

### Case Study 3: Los Angeles Wildfire Emergency Response (January 2025)

A wildfire emergency response scenario based on the January 2025 Los Angeles wildfire events. This case study investigates ambulance allocation and routing under large-scale emergency conditions characterized by:

- Rapidly changing demand patterns
- Infrastructure disruptions
- Road network constraints
- Resource scarcity
- Increased cyber-physical system vulnerabilities

The wildfire case study demonstrates the ability of Q2GA to support resilient emergency response operations during complex multi-hazard events.



## Results

Experimental results demonstrate that Q2GA:

- Reduces emergency response times
- Improves ambulance resource utilization
- Enhances fairness in emergency prioritization
- Increases critical-area coverage
- Improves resilience against cyber threats
- Outperforms baseline optimization methods

---

## Keywords

**Reinforcement Learning**, **Q-Learning**, **Quantum-Inspired Genetic Algorithm**, **Q2GA**, **Autonomous Ambulances**, **Emergency Response Systems**, **Vehicle Routing Problem**, **Cyber Risk Assessment**, **Cyber-Physical Systems**, **Multi-Objective Optimization**, **Smart Healthcare**, **Autonomous Vehicles**

---

## Citation

If you use this repository in your research, please cite the associated publication.

```bibtex
@article{Q2GA2026,
  title={Reinforcement Learning-Assisted Quantum-Inspired Genetic Algorithm for Cyber Risk-Aware Autonomous Ambulance Allocation and Routing},
  author={Author(s)},
  year={2026}
}
```

---

## License

This project is released under the MIT License.
