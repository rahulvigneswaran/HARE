# HARE: Human-in-the-Loop Algorithmic Recourse

This repository contains the code and scripts to reproduce the results from our paper **"HARE: Human-in-the-Loop Algorithmic Recourse"**, accepted at **TMLR 2025**.


[Sai Srinivas Kancheti](https://ksais.github.io/)‡<sup>1</sup>, [Rahul Vigneswaran](https://rahulvigneswaran.github.io/)‡<sup>1</sup>, [Bamdev Mishra](https://bamdevmishra.in/)<sup>2</sup>, [Vineeth N Balasubramanian](https://people.iith.ac.in/vineethnb/)<sup>1</sup>

>‡ contributed equally to this work. 

<sup>1</sup>Indian Institute of Technology Hyderabad, India  
<sup>2</sup>Microsoft, India  

📄 Reviewed on OpenReview: [https://openreview.net/forum?id=56EBglCFvx](https://openreview.net/forum?id=56EBglCFvx)

---

## 🔹 Abstract

Machine learning models are seeing increasing use as decision making systems in domains such as education, finance and healthcare. It is desirable that these models are trustworthy to the end-user, by ensuring fairness, transparency and reliability of decisions. In this work, we consider a key aspect of responsible and transparent AI models -- actionable explanations, viz. the ability of such models to provide recourse to end users adversely affected by their decisions. While algorithmic recourse has seen a variety of efforts in recent years, there have been very few efforts on exploring personalized recourse for a given user. Two users with the same feature profile may prefer vastly different recourses. The limited work in this direction hitherto rely on one-time feature preferences provided by a user. Instead, we present a human-in-the-loop formulation of algorithmic recourse that can incorporate both relative and absolute human feedback for a given test instance. We show that our formulation can extend any existing recourse generating method, enabling the generation of recourses that are satisfactory to the user. We perform experiments on 3 benchmark datasets on top of 6 popular baseline recourse methods where we observe that our framework performs significantly better on simulated user preferences.

---

## 🚀 Reproducing Results

To reproduce the results from our paper, follow these steps:

### 1️⃣ Install Dependencies

```bash
conda env create -f environment.yml
```

### 2️⃣ Run Experiments

To reproduce Table 1 from the paper, execute:

```bash
bash src/scripts/table1.sh
```

For other tables and experiments, modify the script accordingly:

```bash
bash src/scripts/<table_name>.sh
```
### 3️⃣ Results
Results are saved in `logs` directory.

---

## 📌 Citation

If you find our work useful, please cite our paper:

```bibtex
@article{HARE2025,
  title={HARE: Human-in-the-Loop Algorithmic Recourse},
  author={Sai Srinivas Kancheti and Rahul Vigneswaran and Bamdev Mishra and Vineeth N Balasubramanian},
  journal={TMLR},
  year={2025}
}
```
---

## 🛠️ Issues & Support

If you encounter any issues or bugs, please raise an issue on our GitHub repository. We welcome feedback and contributions to improve this work!
