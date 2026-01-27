---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.0
kernelspec:
  display_name: 'Python 3'
  language: python3
  name: python3
---

# About the Course and Instructor

:::{caution} Disclaimer
This is an experimental format. The content of this page is being adapted from Professor Roman's Lectures in Fall 2025 and Spring 2026. Claude 4.5 and Gemini 3.0 Pro are used to clean up the audio transcripts and adapt them for this format. AI may make mistakes. If you into any issues please open an `Issue` or send an email to the TA [Sasank](mailto:sasank.g@duke.edu)
:::

## 1.0 Introduction and Course Overview

### 1.1 Welcome and Instructor Introduction

Welcome to CYBERSEC 520: Artificial Intelligence in Cyber Operations. This first module serves as our launchpad for the semester. We will begin with introductions to get to know each other, establish the logistical framework for the course, and review the essential tools and technologies we will use. By the end of this session, you will have a clear roadmap for what to expect and the foundational knowledge needed to begin our journey into the practical application of AI in cybersecurity.

**Instructor Bio: Michael Roman**

My name is Michael Roman, but please, call me Mike. My own path into the world of AI was not a direct one; it was driven by a deeply personal experience. My interest began with a missed medical diagnosis. When my oldest son was just under three years old, a radiologist looked at his scan and reported nothing of note. However, a persistent pediatrician, trained at Johns Hopkins, suspected something was wrong and pushed for a specialist. It turned out my son had two giant tumors and was diagnosed with stage 5 cancer.

That experience, while terrifying, sparked a critical question for me: how could AI be used to augment human expertise and improve outcomes? How can we create systems that help a doctor in a small commuter town perform at the level of a top specialist at a world-class hospital? Thankfully, my son is now a healthy 11-year-old, but that question set me on the path that brought me here. My initial curiosity was in health, and my move into cybersecurity was driven by opportunity, but the core motivation remains the same: using technology to solve complex, high-stakes problems.

My professional background is a blend of academia, government service, and industry leadership:

*   **Academic Background**: I received my undergraduate degree from Wake Forest University and earned a PhD in Physics from NC State University.
*   **Government Experience**: I spent a decade with the Department of Defense working on advanced concept development, with projects ranging from computational electromagnetics to high-energy laser weapon systems.
*   **Industry Leadership**:
    *   **OOKOS**: I served as the CEO of an AI cybersecurity company that developed technology for continuous user authentication based on their "behavioral fingerprint"—the unique way each person interacts with a network.
    *   **MAXISIQ**: I was the head of research and engineering for this defense contractor, leading a 40-person division focused on applying emerging technologies like AI to cyber operations for clients in the Department of Defense (DoD), Department of Justice (DOJ), and the Intelligence Community.
    *   **Current Role**: I am currently transitioning to a new leadership role at another AI company. Due to an NDA, I can't share the details just yet, but it's an exciting new venture in this space.

My teaching philosophy is simple: ask questions. If you have a question, it's almost certain someone else in the room has the same one. We all come from different backgrounds, and there is no expectation that everyone starts with the same knowledge. This classroom is an interactive and collaborative environment. The more you engage, the more you will learn—not just from me, but from each other.

### 1.2 Class Profile and Learning Goals

A key strength of this program is the diverse professional and academic experience each student brings. Our class introductions revealed a rich collection of backgrounds, which will be a significant asset as we tackle collaborative projects this semester. The cohort includes professionals with direct experience in supply chain security for a major PC manufacturer, streaming media project leadership for global events like the Super Bowl, and front-line work as Security Operations Center (SOC) consultants. This industry experience is complemented by strong academic foundations in computer science and applied cryptography.

While your backgrounds are varied, your reasons for taking this course share a common theme: a desire to bridge the gap between the theory of artificial intelligence and its practical application in cybersecurity. Based on your introductions, the primary learning objectives for this cohort are:

*   **Practical Python & Coding Skills**: Many of you, especially those without a formal computer science background, are eager to gain more hands-on experience and confidence in coding with Python, the lingua franca of machine learning.
*   **Hands-On ML Project Experience**: A recurring goal was to move beyond high-level concepts and theoretical knowledge to actually build, train, and evaluate machine learning models for real-world scenarios.
*   **AI Applications in Cybersecurity**: You want to understand precisely how AI and ML are being deployed today to solve tangible cybersecurity challenges, from threat detection to policy generation.

These goals align perfectly with the course's hands-on, project-based philosophy. We will be spending the majority of our time not just talking about AI, but building with it.

### 1.3 Course Logistics and Required Tools

To ensure a productive and collaborative semester, it's important to establish a clear operational framework. This section covers the class structure, our approach to collaboration, and the specific software and frameworks that will form our technical toolkit for all assignments and projects.

**Class Structure and Collaboration**

*   **Collaborative Work**: I strongly encourage you to work in pairs on most projects and assignments. The goal is to foster a learning environment where you can help each other. Pairing a student with a computer science background with one from a different discipline can be incredibly effective.
*   **Real-World Simulation**: This collaborative approach is designed to mimic the team-based environments you will encounter in the professional world. In a real development team, you are rarely working in isolation.
*   **Class Schedule**: Our sessions are scheduled until 5:50 PM, but I aim to conclude around 5:30 PM. Experience shows that focus wanes after about two and a half hours. My goal is to be as impactful as possible in the time we have. We will also have a break in the middle of each class to recharge.

**Technical Toolkit**

We will rely on a standardized set of tools to ensure consistency and focus our energy on machine learning concepts rather than on troubleshooting installations. This approach mirrors industry best practices where teams standardize on a common tech stack.

*   **Presentation Software**: For the final project, you will need to create a presentation. Any standard tool like PowerPoint, Keynote, or Google Slides is perfectly acceptable.
*   **Programming Environment: Jupyter Notebooks**
    *   All of our assignments and hands-on labs will be completed in Jupyter Notebooks.
    *   I strongly recommend using **Google Colab** as your primary platform. It requires no local installation, runs entirely in the browser, and provides free access to powerful hardware like GPUs, which is essential for training more complex models. Alternatives like Kaggle and Gradient are also available.
*   **Programming Language: Python**
    *   Python is the exclusive programming language for this course.
    *   This is the industry-standard language for machine learning for a simple reason: its vast ecosystem of libraries. Tasks that would have required extensive custom code just 5-10 years ago can now often be accomplished by importing a library and calling a single function.
*   **Core Frameworks**:
    *   **Scikit-learn**: This will be our primary framework for foundational machine learning tasks. It is user-friendly, powerful, and widely used in both academia and commercial tools like Splunk.
    *   **PyTorch**: For deep learning, we will use PyTorch. In recent years, it has surpassed TensorFlow to become the leading framework in the research community, meaning it has excellent support and a vibrant community.

## Hands-on Assignment: Get to Know You
- **Goal**: Map in-class participation questions to an online submission.
- [Assignment Link](../assignments/index.md)


