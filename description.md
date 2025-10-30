# Artemis Health

Our system uses a **neck-mounted sensor** that sends data **every minute** to monitor animal health and behavior.

---

##  Sensor Data Description

| **Data Name**        | **Technical Meaning**                                                             | **Analytical Use**                                                                                 |
| -------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Temperature (°C)** | Measures the animal’s body temperature using a thermal sensor on the neck collar. | Detects **fever**, **heat stress**, **estrus (fertility)**, and general **physiological changes**. |
| **Fxa**              | Acceleration along the **X-axis** (longitudinal axis, forward–backward).          | Indicates **forward movement** such as walking or running.                                         |
| **Mya**              | Acceleration along the **Y-axis** (lateral axis, side-to-side).                   | Reflects **lateral movements**, useful for detecting chewing or ruminating.                        |
| **Rza**              | Acceleration along the **Z-axis** (vertical axis, up–down).                       | Indicates **body orientation** — helps identify standing, lying, or posture changes.               |
| **Sxg**              | Angular velocity around the **X-axis** (roll).                                    | Captures **rolling or twisting** motion of the neck or body.                                       |
| **Lyg**              | Angular velocity around the **Y-axis** (pitch).                                   | Captures **up-and-down** head movements, such as during feeding or ruminating.                     |
| **Dzg**              | Angular velocity around the **Z-axis** (yaw).                                     | Represents **horizontal head rotations** or **orientation changes**.                               |

---

##  Three Main Data Analysis Layers

### **1. Physical Behavior Layer**

**Goal:**
Automatically recognize physical activity, posture, and daily movement patterns of the animal.

**Sub-analyses:**

* Body posture detection (lying, standing, walking)
* Activity level and rest duration (movement intensity, inactivity periods)
* Detection of specific behavioral patterns such as **ruminating**, **feeding**, or **stress-related** motion

---

### **2. Physiological Analysis Layer**

**Goal:**
Monitor internal body changes using temperature and motion data to assess health conditions.

**Sub-analyses:**

* Temperature pattern analysis (average, sudden rise or drop)
* Daily **circadian rhythm** of body temperature (loss of rhythm may indicate illness)
* Correlation between **temperature** and **activity** (e.g., high temperature + low movement → fever)
* Long-term trend tracking to determine whether the animal is **recovering** or **deteriorating**

---

### **3. Health Intelligence and Early Warning Layer**

**Goal:**
Combine behavioral and physiological data to produce **automated health assessments and alerts**.

**Sub-analyses:**

* **Instant Health Detection:**

  * Fever alert → temperature > 39.5 °C **and** reduced motion
  * Heat stress → high temperature **with** high activity
  * Prolonged inactivity alert
* **Health Trend Monitoring:**
  Multi-day evaluation of temperature and activity changes
* **Estrus & Pregnancy Detection:**

  * Estrus → short-term temperature rise (0.3–0.6 °C) with increased activity
  * Pregnancy → stable temperature and gradual reduction in activity after estrus
* **Health Scoring (0–100):**
  Based on temperature stability, activity, and behavioral patterns
* **Automated Alert System:**
  Generates alerts for **fever**, **heat stress**, **estrus**, or **sensor malfunction**

---

##  Summary of Layers

| **Layer**                   | **Focus**                  | **Description**                                    |
| --------------------------- | -------------------------- | -------------------------------------------------- |
| **1 – Behavior**            | *What the animal is doing* | Posture and movement recognition                   |
| **2 – Physiology**          | *How the body is reacting* | Body temperature and internal condition monitoring |
| **3 – Health Intelligence** | *What may happen next*     | Predicts disease, recovery, pregnancy, or stress   |

