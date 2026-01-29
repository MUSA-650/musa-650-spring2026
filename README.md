# Geospatial Machine Learning in Remote Sensing (MUSA 650)

![](public/cover.jpg)

[Overview](#overview-and-objectives) | [Schedule](#schedule) | [Assignments](#assignments) | [Software](#software) | [Academic Integrity](#academic-integrity-and-ai-use)

[![Join Slack](https://custom-icon-badges.demolab.com/badge/Join_Slack-4A154B?logo=slack&logoColor=fff)](https://join.slack.com/t/musa6502026/shared_invite/zt-3monprxnr-5wpkj_4nNYGq2tiBG4qFaw) |  [![Class Zoom Link](https://img.shields.io/badge/Class_Zoom_Link-2D8CFF?logo=zoom&logoColor=white)](https://pennmedicine.zoom.us/j/91607482543)

## Overview and Objectives

Satellite remote sensing is the science of converting raw aerial imagery into actionable intelligence about the built and natural environment. In city planning, this includes applications like building footprint extraction, land cover classification, and change detection. This course will provide you with the foundation necessary to apply machine learning algorithms to satellite imagery using a modern, cloud-native spatial data stack. We will cover supervised and unsupervised learning, model selection, feature engineering, and performance evaluation, as well as deep learning approaches using convolutional neural networks for semantic image segmentation. The course combines lectures, hands-on labs, real-world case studies, and guest speakers to give you a sense of the scope of the field.

The main goal of this class is to give you the conceptual foundation and practical skills to apply machine learning to remote sensing problems independently. You will learn to access and prepare satellite imagery using modern cloud-native tools such as the STAC API, define problems, select appropriate algorithms, build machine learning models, and validate them on new datasets. Given the ready availability of AI-generated code, we emphasize understanding why methods work over memorizing syntax. Our priority is ensuring you can adapt to new tools and methods as the field evolves.

### Instructors

| Name | Email | Office Hours |
|------|-------|-------------|
| [Guray Erus](https://www.med.upenn.edu/cbica/aibil/gerus.html), PhD | [guray.erus@pennmedicine.upenn.edu](mailto:guray.erus@pennmedicine.upenn.edu) | Friday 3-5 PM (or by appointment in case of conflict) |
| [Nissim Lebovits](https://nlebovits.github.io/), MCP | [nissimlebovits@proton.me](mailto:nissimlebovits@proton.me) | By appointment |


## Schedule

MUSA 650 meets in person on Thursday from 8:30am-11:29am in Meyerson B13. In person participation is required. Online participation is an option only in case of a valid excuse with approval.

The course will be divided into two halves. In the first half of the semester, we will focus on building a strong foundational understanding of remote sensing and machine learning. The second half of the semester will delve into more advanced and specific use cases, such as deep learning and data pipelines.

Most weeks, we will divide class time evenly between lectures and labs. Lectures will focus on a conceptual overview of the material, while labs will offer hands-on time to work on pairs or groups on assignments designed to build remote sensing skills applied to planning-specific use case.


| Week # | Date | Topic | Assignment Due | Lab |
|--------|------|-------|------------|-------|
| 1 | 1/15 | Course intro, overview of remote sensing and applications in city planning | | | 
| 2 | 1/22 | Data sources and IO (Landsat, Sentinel, `pystac-client`, etc.) | | [Landsat ML Cookbook: Foundations](week02/README.md) |
| 3 | 1/29 | Non-ML approaches to remote sensing (e.g., indices, change detection, time series) | | [EoPF: Sardinia Wildfires](week03/README.md) | 
| 4 | 2/5 | Introduction to machine learning (unsupervised and supervised) | | [Landsat ML Cookbook: Spectral Clustering](week04/README.md) |
| 5 | 2/12 | Supervised learning for land cover classification, Part 1: training data, cross-validation, method and model selection | | [Week 05](week05/README.md) |
| 6 | 2/19 | Supervised learning for land cover classification, Part 2: validation, evaluation, multi-class classification, parameter optimization | | [Week 06](week06/README.md) |
| 7 | 2/26 | | Midterm Exam | [Google Earth Engine: Land Cover Classification](week07/README.md) |
| Spring Break | | | |
| 8 | 3/12 | Fundamentals of deep learning | | | 
| 9 | 3/19 | Recent advances in deep learning: literature review and paper discussion | [Final Project Proposal](assignments/FINAL_PROJECT_PROPOSAL.md) | | 
| 10 | 3/26 | Convolutional neural networks for image classification; UNet architecture for semantic segmentation | | |
| 11 | 4/2 | Case study: deep learning applications (Guest: Tristan) | [EuroSat](assignments/EUROSAT.md) | | 
| 12 | 4/9 | Case study: predictive modeling using deep learning (Guest: TBD) | | |
| 13 | 4/16 | Big data and machine learning: techniques, tools, challenges, future directions | Final Exam | |
| 14 | 4/23 | Final project presentations | [Final Project](assignments/FINAL_PROJECT.md) | |

## Assignments
There are two exams and two projects over the course of the semester. The midterm and final exams are each worth 20%. The EuroSat assignment is worth 20%. The final project is worth 40% total: 10% for the proposal and 30% for the final submission. For more information, please see the rubrics included with each assignment.

Homework can be started at any time but is due by the end of class by the date indicated on the syllabus. Unexcused late homework will not be accepted.

**All assignments must be completed in groups of 2-3 students.** 
One member of the group should submit on behalf of everyone.

## Software

This course relies on the use of Python and various related packages. All software is open-source and freely available. We will use common tools to facilitate collaboration and ease of use. You are expected to use VSCode, Git, and `uv`during the course in order to make sure that we're all using consistent environments and minimize dependency issues and other kinds of software problems.

### Installation

1. Install [VSCode](https://code.visualstudio.com/download).
2. Install [Git Bash](https://git-scm.com/install/).
3. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
4. Create a fork of this repository.
5. Clone your fork to your local machine in the location of your choice.
6. In VSCode, open the cloned repository. Open a Git Bash terminal and run `uv sync` to install the dependencies.
7. That's it! You're now set up.

## Academic Integrity and AI Use

All students are expected to comply with with Penn's [Code of Academic Integrity](http://www.upenn.edu/academicintegrity/ai_codeofacademicintegrity.html), and obvious copying is prohibited. That said, this course relies extensively on [pair programming](https://www.codecademy.com/resources/blog/what-is-pair-programming/) and aims to prepare you for real-world work. So, you are encouraged to learn how to collaborate effectively and use the internet and other sources to support your work, so long as you attribute it clearly when adapting large chunks of code or other material from other sources. In cases where this is unclear, please make sure to attribute your source.

Relatedly, we have no issue with using AI tools to help with coding (note: you have all have [free GitHub Copilot access as long as you are students](https://education.github.com/pack)). That said, when using any of these tools, you are subject to the standard citation guidelines. Also, be aware that AI tools are subject to mistakes, especially as you get deeper into specialized technical tools. If you are too reliant on these tools, you run the risk of wasting time debugging nonsense code, or not understanding the underlying strucure of what you're writing. Ultimately, it's in your best interest to learn to use AI tools as _tools_--not as replacements for critical thinking--and to learn how to use them _intelligently and effectively_.

