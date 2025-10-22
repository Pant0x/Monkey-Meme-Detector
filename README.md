# 🐒 Monkey Meme Detector

A fun, real-time computer vision project that detects **hand gestures** and **facial expressions** using your webcam — and switches between monkey meme reactions depending on your pose.

Built with **OpenCV** and **MediaPipe**, the detector recognizes 3 main gestures:
- 🧠 **Thinking** (when you touch your mouth)
- 💡 **Idea** (when you raise a finger)
- 😁 **Smile** (when you smile)
- 😐 **Normal** (default state)

Each gesture dynamically triggers a different meme image on the right side of the screen.

---

## ⚙️ Features

- Real-time webcam detection (face + hand)
- Dynamic meme switching
- On-screen labels and bounding box
- Simple and clean UI
- Works fully offline

---

## 🧩 Requirements

Make sure you have Python **3.8+** installed.  
Then install dependencies:

```bash
pip install -r requirements.txt
