# CYBERSEC 520: Machine Learning in Cybersecurity

The course website can be accessed over here - [https://mproman.github.io/CYBERSEC520/intro.html](https://mproman.github.io/CYBERSEC520/intro.html)

This repository contains the source code for the "Machine Learning in Cybersecurity" online course book. This material is designed to supplement the official course content.

**Note to Students:** The **complete and official course content** is available on **Canvas**. Please refer to Canvas for valid assignments, due dates, and official announcements.

## Support & Feedback

If you encounter issues with this website (e.g., broken links, rendering errors) or the notebook code, please:

1.  **Raise an Issue:** Open a [GitHub Issue](https://github.com/course/repo/issues) on this repository.
2.  **Contact the TA:** Email **Sasank Garimella** at [sasank.g@duke.edu](mailto:sasank.g@duke.edu).
3.  **Contact the Professor:** Reach out to **Dr.Michael Roman**.

## For Students

**How to Use This Course:**
1.  **Read Online:** The most up-to-date content is hosted on **Canvas**.
2.  **Interact:** The course includes interactive Jupyter Notebooks which you can view directly on the website.
3.  **Download:** To run the notebooks locally, please download the `.ipynb` files from **Canvas** or directly from this repository.

## For Instructors & Developers

### Prerequisites
*   Python 3.11+
*   [`uv`](https://github.com/astral-sh/uv) (Package Manager)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```

2.  **Create environment & install dependencies:**
    ```bash
    uv venv
    uv pip install -r requirements.txt
    ```

### Building the Book
To build the HTML documentation locally:
```bash
uv run jupyter-book build .
```
The output will be in `_build/html/index.html`.

### Cleaning
To remove the build directory:
```bash
uv run jupyter-book clean .
```

