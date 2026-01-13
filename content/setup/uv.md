# Setting up Python with UV

We recommend using [uv](https://github.com/astral-sh/uv) for managing Python versions and dependencies in this course. It is extremely fast and simplifies the workflow.

## Installation

### Windows (PowerShell)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, restart your terminal to ensure `uv` is in your PATH.

## Setting up the Environment

1.  **Clone the Repository** and navigate to the project folder.
2.  **Create a Virtual Environment**:
    ```bash
    uv venv
    ```
    This creates a `.venv` folder with a fresh Python installation.

3.  **Activate the Environment** (Optional but recommended):
    - Windows: `.venv\Scripts\activate`
    - macOS/Linux: `source .venv/bin/activate`

4.  **Install Dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

## Common Commands

- **Run a script**: `uv run script.py`
- **Add a package**: `uv pip install pandas`
- **Check installed packages**: `uv pip freeze`
- **Update dependencies**: `uv pip compile requirements.in -o requirements.txt` (if using requirements.in)
